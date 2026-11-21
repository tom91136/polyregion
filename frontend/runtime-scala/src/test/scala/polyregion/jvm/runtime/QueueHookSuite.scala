package polyregion.jvm.runtime

import java.lang.ref.WeakReference
import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch, Semaphore, TimeUnit}

class QueueHookSuite extends munit.FunSuite {

  private def waitFor(latch: CountDownLatch, what: String): Unit =
    if (!latch.await(30, TimeUnit.SECONDS)) throw IllegalStateException(s"timed out: $what")

  private class Stop {
    val arrived = CountDownLatch(1)
    val resume  = CountDownLatch(1)
    def hold(): Unit = {
      arrived.countDown()
      waitFor(resume, "never resumed")
    }
    def awaitArrival(): Unit = waitFor(arrived, "never arrived")
  }

  private object Seam {
    private val actions = ConcurrentHashMap[(String, String), () => Unit]()

    def on(thread: String, point: String)(action: => Unit): Unit = actions.put((thread, point), () => action)

    def install(): Unit = Queue.hook = (point: String) =>
      Seq(Thread.currentThread.getName, "*").flatMap(t => Option(actions.get((t, point)))).foreach(_())

    def clear(): Unit = {
      Queue.hook = (_: String) => ()
      actions.clear()
    }
  }

  private def thread(name: String)(body: => Unit): Thread = {
    val t = Thread((() => body): Runnable, name)
    t.setDaemon(true)
    t.start()
    t
  }

  private def awaitState(t: Thread, states: Thread.State*): Thread.State = {
    val deadline = System.nanoTime + TimeUnit.SECONDS.toNanos(10)
    val wanted   = states.toSet
    while (!wanted.contains(t.getState) && System.nanoTime < deadline) Thread.sleep(1)
    t.getState
  }

  private def direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)

  private case class Row(label: String, device: Option[DeviceInfo])

  private val rows = {
    val all = Context.platforms().flatMap(_.devices)
    List(
      Row("host", all.find(d => d.platformKind == 1 && d.moduleFormat == 2)),
      Row("nvidia", all.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0))
    )
  }

  private def withContext(row: Row)(f: Context => Unit): Unit = row.device match {
    case None => assume(cond = false, s"no ${row.label} device in the catalogue")
    case Some(device) =>
      val ctx = Context.create(device)
      Seam.install()
      try f(ctx)
      finally {
        Seam.clear()
        ctx.close()
      }
  }

  private def await(start: Completion => Unit): Option[Throwable] = {
    val latch   = CountDownLatch(1)
    val failure = AtomicReference[Throwable]()
    start { f =>
      failure.set(f); latch.countDown()
    }
    waitFor(latch, "completion never ran")
    Option(failure.get)
  }

  private def register(ctx: Context, buffer: ByteBuffer): Unit = {
    assertEquals(
      await { c =>
        ctx.queue.invalidate(buffer, c); ()
      },
      None
    )
    assertNotEquals(ctx.queue.devicePointer(buffer), 0L)
  }

  rows.foreach { row =>
    test(s"[${row.label}] release waits for an acquire parked between lookup and pending increment") {
      withContext(row) { ctx =>
        val buffer = direct(64)
        register(ctx, buffer)
        val stop      = Stop()
        val completed = CountDownLatch(1)
        val failure   = AtomicReference[Throwable]()
        Seam.on("acquirer", "acquire.found")(stop.hold())
        val acquirer = thread("acquirer") {
          ctx.queue.invalidate(
            buffer,
            f => {
              failure.set(f); completed.countDown()
            }
          )
        }
        stop.awaitArrival()
        val releaser = thread("releaser")(ctx.queue.release(buffer))
        val state    = awaitState(releaser, Thread.State.BLOCKED, Thread.State.TERMINATED)
        try
          assertEquals(
            state,
            Thread.State.BLOCKED,
            "release ran while an acquire sat between its lookup and its pending increment"
          )
        finally stop.resume.countDown()
        waitFor(completed, "invalidate never completed")
        assertEquals(Option(failure.get), None)
        releaser.join(30000)
        acquirer.join(30000)
        assert(!releaser.isAlive && !acquirer.isAlive)
        assertEquals(ctx.queue.devicePointer(buffer), 0L)
      }
    }

    test(s"[${row.label}] concurrent releases of one buffer free it once") {
      withContext(row) { ctx =>
        val buffer = direct(64)
        register(ctx, buffer)
        val before   = ctx.queue.stats().releases
        val stop     = Stop()
        val outcomes = ConcurrentLinkedQueue[String]()
        Seam.on("copier", "copy.begin")(stop.hold())
        val copier = thread("copier")(ctx.queue.invalidate(buffer, _ => ()))
        stop.awaitArrival()
        val releasers = (0 until 2).map { i =>
          thread(s"releaser-$i") {
            try {
              ctx.queue.release(buffer)
              outcomes.add("released")
            } catch { case _: IllegalArgumentException => outcomes.add("rejected") }
          }
        }
        val states = releasers.map(awaitState(_, Thread.State.WAITING, Thread.State.TERMINATED))
        try
          assertEquals(
            states.toList,
            List(Thread.State.WAITING, Thread.State.WAITING),
            "both releases wait on the pending copy"
          )
        finally stop.resume.countDown()
        releasers.foreach(_.join(30000))
        copier.join(30000)
        assert(releasers.forall(!_.isAlive) && !copier.isAlive)
        assertEquals(outcomes.toArray.map(_.toString).sorted.toList, List("rejected", "released"))
        assertEquals(ctx.queue.stats().releases - before, 1L)
      }
    }

    test(s"[${row.label}] allocating for a new buffer cannot deadlock with a copy that settles inline") {
      withContext(row) { ctx =>
        val settling = direct(64)
        val fresh    = direct(64)
        register(ctx, settling)
        val nativeLock = Semaphore(1)
        val holding    = CountDownLatch(1)
        val allocating = CountDownLatch(1)
        Seam.on("copier", "copy.begin") {
          nativeLock.acquire()
          holding.countDown()
          waitFor(allocating, "allocation never started")
        }
        Seam.on("copier", "copy.end")(nativeLock.release())
        Seam.on("allocator", "allocate") {
          allocating.countDown()
          nativeLock.acquire()
          nativeLock.release()
        }
        val copier = thread("copier")(ctx.queue.invalidate(settling, _ => ()))
        waitFor(holding, "copy never started")
        val allocator = thread("allocator")(ctx.queue.invalidate(fresh, _ => ()))
        copier.join(10000)
        allocator.join(10000)
        val stuck  = copier.isAlive || allocator.isAlive
        val states = s"copier=${copier.getState} allocator=${allocator.getState}"
        if (stuck) nativeLock.release()
        copier.join(30000)
        allocator.join(30000)
        assert(!stuck, s"deadlock: $states")
        assertNotEquals(ctx.queue.devicePointer(fresh), 0L)
      }
    }
  }

  rows.filter(_.label == "host").foreach { row =>
    test(s"[${row.label}] completions run in the order the copies settled, not the order they were issued") {
      withContext(row) { ctx =>
        val first, second = direct(64)
        register(ctx, first)
        register(ctx, second)
        val stop  = Stop()
        val order = java.util.Collections.synchronizedList(java.util.ArrayList[String]())
        val done  = CountDownLatch(2)
        Seam.on("slow", "settled")(stop.hold())
        val slow = thread("slow") {
          ctx.queue.invalidate(
            first,
            _ => {
              order.add("first"); done.countDown()
            }
          )
        }
        stop.awaitArrival()
        ctx.queue.invalidate(
          second,
          _ => {
            order.add("second"); done.countDown()
          }
        )
        stop.resume.countDown()
        waitFor(done, "completions never ran")
        slow.join(30000)
        assertEquals(order.toArray.map(_.toString).toList, List("second", "first"))
      }
    }
  }

  private val copyPaths = List(true, false)

  private def gcClears(ref: WeakReference[ByteBuffer]): Boolean =
    Iterator.range(0, 40).exists { _ =>
      System.gc()
      Thread.sleep(25)
      Option(ref.get).isEmpty
    }

  rows.foreach { row =>
    copyPaths.foreach { internal =>
      val path = if (internal) "internal" else "public"
      test(s"[${row.label}] a $path copy keeps its buffer reachable until the completion runs") {
        withContext(row) { ctx =>
          val device  = ctx.malloc(64)
          val stop    = Stop()
          val where   = AtomicReference[Thread]()
          val ref     = AtomicReference[WeakReference[ByteBuffer]]()
          val settled = CountDownLatch(1)
          Seam.on("*", "copy.completing") {
            where.set(Thread.currentThread)
            stop.hold()
          }
          val caller = thread("caller") {
            val buffer = direct(64)
            ref.set(WeakReference(buffer))
            val cb: Completion = _ => settled.countDown()
            if (internal) ctx.copyToDeviceDirect(buffer, device, 64, cb) else ctx.copyToDevice(buffer, device, 64, cb)
          }
          stop.awaitArrival()
          val inline = where.get eq caller
          try
            if (!inline) {
              caller.join(30000)
              assert(!caller.isAlive)
              assert(!gcClears(ref.get), "the buffer was collected while its copy was pending")
            }
          finally stop.resume.countDown()
          waitFor(settled, "completion never ran")
          assume(!inline, "the completion runs inline on the calling thread, which keeps the buffer reachable")
          ctx.free(device)
        }
      }
    }
  }
}
