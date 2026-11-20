package polyregion

import polyregion.ast.{CodeGen, MsgPack, PolyAST as p}
import polyregion.jvm.compiler as ct
import polyregion.jvm.runtime as rt
import polyregion.scalalang.{blocking, future}

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{CompletableFuture, CountDownLatch, TimeUnit}
import scala.concurrent.Await
import scala.concurrent.duration.*

class QueueSuite extends munit.FunSuite {

  private val IntPtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)

  private def select(tpe: p.Type, n: String): (p.Named, p.Term.Select) = {
    val named = p.Named(n, tpe)
    named -> p.Term.Select(named, Nil, tpe)
  }

  private def bumpProgram: p.Program = {
    val (out, outRef) = select(IntPtr, "out")
    val (by, byRef)   = select(p.Type.IntS32, "by")
    val (gid, gidRef) = select(p.Type.IntU32, "gid")
    val (idx, idxRef) = select(p.Type.IntS32, "idx")
    val (cur, curRef) = select(p.Type.IntS32, "cur")
    val (nxt, nxtRef) = select(p.Type.IntS32, "nxt")
    val body = List(
      p.Stmt.Var(gid, Some(p.Expr.SpecOp(p.Spec.GpuGlobalIdx(p.Term.IntU32Const(0)))), false),
      p.Stmt.Var(idx, Some(p.Expr.Cast(gidRef, p.Type.IntS32)), false),
      p.Stmt.Var(cur, Some(p.Expr.Index(outRef, idxRef, p.Type.IntS32)), false),
      p.Stmt.Var(nxt, Some(p.Expr.IntrOp(p.Intr.Add(curRef, byRef, p.Type.IntS32))), false),
      p.Stmt.Update(outRef, idxRef, nxtRef),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val decl = p.FunctionDecl(
      p.Sym("bump"),
      Nil,
      None,
      List(p.Arg(out), p.Arg(by)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Offload
    )
    p.Program(
      Some(
        p.Function(
          decl,
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.OffloadEntry
        )
      ),
      Nil,
      Nil
    )
  }

  private def compile(target: ct.Target, arch: String): ct.Compilation = {
    val compiler = ct.Compiler.create()
    try
      compiler.compile(
        MsgPack.encode(CodeGen.programVersioned(bumpProgram)),
        false,
        ct.Options.of(target, arch),
        ct.Opt.O3.value
      )
    finally compiler.close()
  }

  private case class Row(label: String, device: Option[rt.DeviceInfo], target: ct.Target, arch: String)

  private val rows = {
    val all = rt.Context.platforms().flatMap(_.devices)
    val host = Row(
      "host",
      all.find(d => d.platformKind == 1 && d.moduleFormat == 2),
      ct.Target.LLVM_HOST,
      "native"
    )
    val nvidia = Row(
      "nvidia",
      all.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0),
      ct.Target.LLVM_NVPTX64,
      all
        .find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0)
        .map(d => s"sm_${d.cudaArchitectureMajor}${d.cudaArchitectureMinor}")
        .getOrElse("sm_80")
    )
    List(host, nvidia)
  }

  private def direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)

  private def ints(n: Int, f: Int => Int): ByteBuffer = {
    val b = direct(n * 4)
    (0 until n).foreach(i => b.putInt(i * 4, f(i)))
    b
  }

  private def contents(b: ByteBuffer): List[Int] = (0 until b.capacity / 4).map(i => b.getInt(i * 4)).toList

  private class Fixture(val ctx: rt.Context, val image: rt.ModuleImage) {
    val queue: rt.Queue = ctx.queue()

    def bump(pointer: Long, n: Int, by: Int): Unit = {
      val types = Array(rt.Type.PTR.value, rt.Type.INT.value)
      val data  = ByteBuffer.allocate(12).order(ByteOrder.nativeOrder).putLong(pointer).putInt(by).array
      assertEquals(ctx.select(Array(image)), 0)
      ctx.launch(Array(image), image.name, "bump", types, data, rt.Policy(rt.Dim3(n, 1, 1)))
    }

    def await(start: rt.Completion => Unit): Throwable = {
      val latch   = CountDownLatch(1)
      var failure = Option.empty[Throwable]
      start { f =>
        failure = Option(f); latch.countDown()
      }
      assert(latch.await(30, TimeUnit.SECONDS), "completion never ran")
      failure.orNull
    }
  }

  private def withRow(row: Row)(f: Fixture => Unit): Unit = row.device match {
    case None => assume(false, s"no ${row.label} device in the catalogue")
    case Some(device) =>
      val c     = compile(row.target, row.arch)
      val image = rt.ModuleImage(s"bump@${row.label}", row.target, c.features, c.program)
      val ctx   = rt.Context.create(device)
      try f(Fixture(ctx, image))
      finally ctx.close()
  }

  rows.foreach { row =>
    test(s"[${row.label}] invalidate, launch and sync round-trip a buffer") {
      withRow(row) { fx =>
        val buffer = ints(64, identity)
        val ptr    = fx.queue.invalidate(buffer, _ => ())
        assertNotEquals(ptr, 0L)
        fx.bump(ptr, 64, 3)
        assertEquals(fx.await(fx.queue.sync(buffer, _)), null)
        assertEquals(contents(buffer), (0 until 64).map(_ + 3).toList)
        fx.queue.release(buffer)
        assertEquals(fx.queue.devicePointer(buffer), 0L)
      }
    }

    test(s"[${row.label}] completions run once, in order, off the calling thread") {
      withRow(row) { fx =>
        val a, b   = ints(16, identity)
        val order  = java.util.Collections.synchronizedList(java.util.ArrayList[String]())
        val calls  = AtomicInteger()
        val caller = Thread.currentThread
        val done   = CountDownLatch(4)
        def record(name: String): rt.Completion = failure => {
          assertEquals(failure, null)
          assertNotEquals(Thread.currentThread, caller)
          order.add(name); calls.incrementAndGet(); done.countDown()
        }
        fx.queue.invalidate(a, record("invalidate a"))
        fx.queue.invalidate(b, record("invalidate b"))
        fx.queue.sync(a, record("sync a"))
        fx.queue.sync(b, record("sync b"))
        assert(done.await(30, TimeUnit.SECONDS))
        Thread.sleep(100)
        assertEquals(calls.get, 4)
        assertEquals(order.toArray.toList, List("invalidate a", "invalidate b", "sync a", "sync b"))
      }
    }

    test(s"[${row.label}] a second invalidate re-uploads the same allocation") {
      withRow(row) { fx =>
        val buffer = ints(8, identity)
        val first  = fx.queue.invalidate(buffer, _ => ())
        (0 until 8).foreach(i => buffer.putInt(i * 4, 100 + i))
        val second = fx.queue.invalidate(buffer, _ => ())
        assertEquals(second, first)
        fx.bump(second, 8, 1)
        assertEquals(fx.await(fx.queue.sync(buffer, _)), null)
        assertEquals(contents(buffer), (0 until 8).map(_ + 101).toList)
        fx.queue.release(buffer)
      }
    }

    test(s"[${row.label}] unregistered buffers are rejected") {
      withRow(row) { fx =>
        val buffer = ints(4, identity)
        intercept[IllegalArgumentException](fx.queue.sync(buffer, _ => ()))
        intercept[IllegalArgumentException](fx.queue.release(buffer))
        intercept[IllegalArgumentException](fx.queue.invalidate(ByteBuffer.allocate(16), _ => ()))
        fx.queue.invalidate(buffer, _ => ())
        fx.queue.release(buffer)
        intercept[IllegalArgumentException](fx.queue.release(buffer))
      }
    }

    test(s"[${row.label}] a closed context fails operations through the completion") {
      withRow(row) { fx =>
        val buffer = ints(4, identity)
        fx.ctx.close()
        val failure = fx.await(c => assertEquals(fx.queue.invalidate(buffer, c), 0L))
        assert(failure.isInstanceOf[rt.PolyregionRuntimeException], String.valueOf(failure))
        assert(failure.getMessage.endsWith(" is closed"), failure.getMessage)
      }
    }

    test(s"[${row.label}] a scope uploads once, writes back once, and launches share the copy") {
      withRow(row) { fx =>
        val n         = 32
        val launches  = 5
        val xs, ys    = Array.tabulate(n)(identity)
        val before    = fx.queue.stats()
        val ctxBefore = fx.ctx.stats()
        fx.ctx.resident(
          Array[AnyRef](xs, ys),
          (scope: rt.Queue.Scope) => {
            (0 until launches).foreach(_ => fx.bump(scope.pointer(xs), n, 1))
            fx.bump(scope.pointer(ys), n, 10)
          }
        )
        val after = fx.queue.stats()
        assertEquals(after.invalidations - before.invalidations, 2L)
        assertEquals(after.syncs - before.syncs, 2L)
        assertEquals(after.releases - before.releases, 2L)
        assertEquals(after.stagedIn - before.stagedIn, 2L)
        assertEquals(after.stagedOut - before.stagedOut, 2L)
        assertEquals(xs.toList, (0 until n).map(_ + launches).toList)
        assertEquals(ys.toList, (0 until n).map(_ + 10).toList)
        if (!fx.ctx.sharedAddressSpace) {
          val ctxAfter = fx.ctx.stats()
          assertEquals(ctxAfter.hostToDevice - ctxBefore.hostToDevice, 2L)
          assertEquals(ctxAfter.deviceToHost - ctxBefore.deviceToHost, 2L)
        }
        assertEquals(fx.queue.devicePointer(xs), 0L)
      }
    }

    test(s"[${row.label}] outside a scope every launch copies") {
      withRow(row) { fx =>
        val n        = 32
        val launches = 5
        val xs       = Array.tabulate(n)(identity)
        val before   = fx.queue.stats()
        (0 until launches).foreach { _ =>
          fx.ctx.resident(Array[AnyRef](xs), (scope: rt.Queue.Scope) => fx.bump(scope.pointer(xs), n, 1))
        }
        val after = fx.queue.stats()
        assertEquals(after.invalidations - before.invalidations, launches.toLong)
        assertEquals(after.syncs - before.syncs, launches.toLong)
        assertEquals(xs.toList, (0 until n).map(_ + launches).toList)
      }
    }

    test(s"[${row.label}] a scope writes back when the body throws") {
      withRow(row) { fx =>
        val n  = 16
        val xs = Array.tabulate(n)(identity)
        val e = intercept[IllegalStateException] {
          fx.ctx.resident(
            Array[AnyRef](xs),
            (scope: rt.Queue.Scope) => {
              fx.bump(scope.pointer(xs), n, 7)
              throw IllegalStateException("body")
            }
          )
        }
        assertEquals(e.getMessage, "body")
        assertEquals(xs.toList, (0 until n).map(_ + 7).toList)
        assertEquals(fx.queue.devicePointer(xs), 0L)
      }
    }

    test(s"[${row.label}] a host write inside a scope is overwritten by the write-back") {
      withRow(row) { fx =>
        val n  = 8
        val xs = Array.tabulate(n)(identity)
        fx.ctx.resident(
          Array[AnyRef](xs),
          (scope: rt.Queue.Scope) => {
            xs.indices.foreach(i => xs(i) = -1)
            fx.bump(scope.pointer(xs), n, 1)
          }
        )
        assertEquals(xs.toList, (0 until n).map(_ + 1).toList)
      }
    }

    test(s"[${row.label}] an explicit invalidate inside a scope uploads the host write") {
      withRow(row) { fx =>
        val n  = 8
        val xs = Array.tabulate(n)(identity)
        fx.ctx.resident(
          Array[AnyRef](xs),
          (scope: rt.Queue.Scope) => {
            xs.indices.foreach(i => xs(i) = 50 + i)
            scope.invalidate(xs)
            fx.bump(scope.pointer(xs), n, 1)
          }
        )
        assertEquals(xs.toList, (0 until n).map(_ + 51).toList)
      }
    }

    test(s"[${row.label}] a nested scope reuses the outer residents") {
      withRow(row) { fx =>
        val n      = 8
        val xs     = Array.tabulate(n)(identity)
        val before = fx.queue.stats()
        fx.ctx.resident(
          Array[AnyRef](xs),
          (outer: rt.Queue.Scope) => {
            val pointer = outer.pointer(xs)
            fx.ctx.resident(
              Array[AnyRef](xs),
              (inner: rt.Queue.Scope) => {
                assertEquals(inner.pointer(xs), pointer)
                fx.bump(inner.pointer(xs), n, 1)
              }
            )
            assertEquals(fx.queue.devicePointer(xs), pointer)
            fx.bump(pointer, n, 1)
          }
        )
        val after = fx.queue.stats()
        assertEquals(after.invalidations - before.invalidations, 1L)
        assertEquals(after.syncs - before.syncs, 1L)
        assertEquals(xs.toList, (0 until n).map(_ + 2).toList)
      }
    }

    test(s"[${row.label}] a scope over a direct buffer stages nothing") {
      withRow(row) { fx =>
        val n      = 8
        val buffer = ints(n, identity)
        val before = fx.queue.stats()
        fx.ctx.resident(Array[AnyRef](buffer), (scope: rt.Queue.Scope) => fx.bump(scope.pointer(buffer), n, 4))
        val after = fx.queue.stats()
        assertEquals(after.stagedIn - before.stagedIn, 0L)
        assertEquals(after.stagedOut - before.stagedOut, 0L)
        assertEquals(contents(buffer), (0 until n).map(_ + 4).toList)
      }
    }

    test(s"[${row.label}] a scope rejects unsupported residents") {
      withRow(row) { fx =>
        intercept[IllegalArgumentException](
          fx.ctx.resident(Array[AnyRef]("string"), (_: rt.Queue.Scope) => ())
        )
        intercept[IllegalArgumentException](
          fx.ctx.resident(Array[AnyRef](ByteBuffer.allocate(4)), (_: rt.Queue.Scope) => ())
        )
      }
    }

    test(s"[${row.label}] a public copy completes on the queue thread with the bytes moved") {
      withRow(row) { fx =>
        val source = ints(16, _ + 5)
        val back   = direct(64)
        val remote = fx.ctx.malloc(64)
        try {
          var thread = ""
          assertEquals(
            fx.await(c =>
              fx.ctx.copyToDevice(source, remote, 64, f => { thread = Thread.currentThread.getName; c.complete(f) })
            ),
            null
          )
          assertEquals(thread, "polyregion-queue")
          assertEquals(fx.await(c => fx.ctx.copyFromDevice(remote, back, 64, c)), null)
          assertEquals(contents(back), (0 until 16).map(_ + 5).toList)
        } finally fx.ctx.free(remote)
      }
    }

    test(s"[${row.label}] a completion that throws goes to the uncaught exception handler") {
      withRow(row) { fx =>
        val seen     = java.util.concurrent.LinkedBlockingQueue[Throwable]()
        val previous = Thread.getDefaultUncaughtExceptionHandler
        Thread.setDefaultUncaughtExceptionHandler((_, e) => seen.add(e))
        try {
          val buffer = ints(4, identity)
          fx.queue.invalidate(buffer, _ => throw IllegalStateException("completion"))
          val reported = seen.poll(30, TimeUnit.SECONDS)
          assert(reported != null && reported.getMessage == "completion", String.valueOf(reported))
          assertEquals(fx.await(fx.queue.sync(buffer, _)), null)
          fx.queue.release(buffer)
        } finally Thread.setDefaultUncaughtExceptionHandler(previous)
      }
    }

    test(s"[${row.label}] blocking on the queue from a completion fails fast") {
      withRow(row) { fx =>
        val buffer  = ints(4, identity)
        val outcome = java.util.concurrent.LinkedBlockingQueue[Either[Throwable, Unit]]()
        fx.queue.invalidate(
          buffer,
          _ =>
            outcome.add(
              try Right(blocking.sync(fx.ctx, buffer))
              catch { case e: Throwable => Left(e) }
            )
        )
        val result = outcome.poll(30, TimeUnit.SECONDS)
        assert(result != null, "the completion never returned")
        assert(result.left.exists(_.isInstanceOf[IllegalStateException]), String.valueOf(result))
        fx.queue.release(buffer)
      }
    }

    test(s"[${row.label}] a buffer is resident on one context at a time") {
      withRow(row) { fx =>
        val other  = rt.Context.create(row.device.get)
        val buffer = ints(4, identity)
        try {
          fx.queue.invalidate(buffer, _ => ())
          assertEquals(rt.Queue.ownerOf(buffer), fx.queue)
          val e = intercept[IllegalArgumentException](other.queue.invalidate(buffer, _ => ()))
          assert(e.getMessage.startsWith("Already resident on another context"), e.getMessage)
          fx.queue.release(buffer)
          assertEquals(rt.Queue.ownerOf(buffer), null)
          assertNotEquals(other.queue.invalidate(buffer, _ => ()), 0L)
          other.queue.release(buffer)
        } finally other.close()
      }
    }

    test(s"[${row.label}] concurrent invalidate, sync and release of shared buffers stay consistent") {
      withRow(row) { fx =>
        val buffers  = Array.fill(4)(ints(64, identity))
        val failures = java.util.concurrent.ConcurrentLinkedQueue[Throwable]()
        val threads = (0 until 8).map { t =>
          new Thread(
            (
                () =>
                  try
                    (0 until 300).foreach { i =>
                      val b = buffers((t + i) % buffers.length)
                      try
                        (i % 3) match {
                          case 0 => fx.queue.invalidate(b, _ => ())
                          case 1 => fx.queue.sync(b, _ => ())
                          case _ => fx.queue.release(b)
                        }
                      catch { case _: IllegalArgumentException => () }
                    }
                  catch { case e: Throwable => failures.add(e) }
            ): Runnable
          )
        }
        threads.foreach(_.start())
        threads.foreach(_.join(120000))
        assert(threads.forall(!_.isAlive), "a worker never finished")
        assertEquals(failures.toString, "[]")
        fx.queue.releaseAll()
        buffers.foreach(b => assertEquals(fx.queue.devicePointer(b), 0L))
        assertEquals(fx.await(fx.queue.invalidateAll(_)), null)
      }
    }

    test(s"[${row.label}] the blocking adapter completes and fails") {
      withRow(row) { fx =>
        val buffer = ints(8, identity)
        blocking.invalidate(fx.ctx, buffer)
        fx.bump(fx.queue.devicePointer(buffer), 8, 1)
        blocking.sync(fx.ctx, buffer)
        assertEquals(contents(buffer), (0 until 8).map(_ + 1).toList)
        blocking.release(fx.ctx, buffer)
        intercept[IllegalArgumentException](blocking.sync(fx.ctx, buffer))
        fx.ctx.close()
        intercept[rt.PolyregionRuntimeException](blocking.invalidate(fx.ctx, ints(4, identity)))
      }
    }

    test(s"[${row.label}] the future adapter completes and fails") {
      withRow(row) { fx =>
        val buffer = ints(8, identity)
        Await.result(future.invalidate(fx.ctx, buffer), 30.seconds)
        fx.bump(fx.queue.devicePointer(buffer), 8, 2)
        Await.result(future.sync(fx.ctx, buffer), 30.seconds)
        assertEquals(contents(buffer), (0 until 8).map(_ + 2).toList)
        future.release(fx.ctx, buffer)
        fx.ctx.close()
        intercept[rt.PolyregionRuntimeException](Await.result(future.invalidate(fx.ctx, ints(4, identity)), 30.seconds))
      }
    }

    test(s"[${row.label}] the CompletableFuture adapter completes and fails") {
      withRow(row) { fx =>
        val buffer = ints(8, identity)
        polyregion.javalang.QueueFutures.invalidate(fx.queue, buffer).get(30, TimeUnit.SECONDS)
        fx.bump(fx.queue.devicePointer(buffer), 8, 3)
        polyregion.javalang.QueueFutures.sync(fx.queue, buffer).get(30, TimeUnit.SECONDS)
        assertEquals(contents(buffer), (0 until 8).map(_ + 3).toList)
        fx.queue.release(buffer)
        fx.ctx.close()
        val failed: CompletableFuture[Void] = polyregion.javalang.QueueFutures.invalidate(fx.queue, ints(4, identity))
        val e = intercept[java.util.concurrent.ExecutionException](failed.get(30, TimeUnit.SECONDS))
        assert(e.getCause.isInstanceOf[rt.PolyregionRuntimeException], String.valueOf(e.getCause))
      }
    }
  }
}
