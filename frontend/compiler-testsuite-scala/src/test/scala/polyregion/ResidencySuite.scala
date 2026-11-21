package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*
import polyregion.scalalang.blocking.Host

import java.nio.{ByteBuffer, ByteOrder}

class ResidencySuite extends munit.FunSuite {

  private class CountingDoubles(backing: ByteBuffer, id: Long)
      extends Buffer.DoubleBuffer(backing, backing.asDoubleBuffer(), id) {
    var reads                                    = 0
    var writes                                   = 0
    override def apply(i: Int): Double           = { reads += 1; buffer.get(i) }
    override def update(i: Int, x: Double): Unit = { writes += 1; buffer.put(i, x); () }
  }

  private val ids = java.util.concurrent.atomic.AtomicLong(1000)

  private def counting(n: Int): CountingDoubles = {
    val buffer = new CountingDoubles(ByteBuffer.allocateDirect(n * 8).order(ByteOrder.nativeOrder), ids.getAndIncrement)
    (0 until n).foreach(i => buffer.buffer.put(i, i.toDouble))
    buffer
  }

  private def values(xs: CountingDoubles): List[Double] = (0 until xs.length).map(xs.buffer.get).toList

  private inline def bump(xs: Buffer[Double], n: Int): Unit = {
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0(Host.context, outcome = _) {
      var i = 0
      while (i < n) {
        xs(i) = xs(i) + 1.0
        i += 1
      }
      ()
    }
    outcome.fold(throw _, identity)
  }

  test("outside a scope every launch copies the elements both ways") {
    val n                        = 64
    val launches                 = 3
    val xs                       = counting(n)
    val captured: Buffer[Double] = xs
    (0 until launches).foreach(_ => bump(captured, n))
    assertEquals(values(xs), (0 until n).map(_.toDouble + launches).toList)
    assertEquals(xs.reads, launches * n)
    assertEquals(xs.writes, launches * n)
  }

  test("inside a scope launches copy nothing and the scope uploads and writes back once") {
    val n                        = 64
    val launches                 = 5
    val xs                       = counting(n)
    val captured: Buffer[Double] = xs
    val queue                    = Host.context.queue
    val before                   = queue.stats()
    Host.context.resident(captured) {
      (0 until launches).foreach(_ => bump(captured, n))
    }
    val after = queue.stats()
    assertEquals(values(xs), (0 until n).map(_.toDouble + launches).toList)
    assertEquals(xs.reads, 0)
    assertEquals(xs.writes, 0)
    assertEquals(after.invalidations - before.invalidations, 1L)
    assertEquals(after.syncs - before.syncs, 1L)
    assertEquals(queue.devicePointer(xs.backing), 0L)
  }

  test("a scope writes back and releases when the body throws") {
    val n                        = 16
    val xs                       = counting(n)
    val captured: Buffer[Double] = xs
    intercept[IllegalStateException] {
      Host.context.resident(captured) {
        bump(captured, n)
        throw IllegalStateException("body")
      }
    }
    assertEquals(values(xs), (0 until n).map(_.toDouble + 1).toList)
    assertEquals(Host.context.queue.devicePointer(xs.backing), 0L)
  }

  test("a launch inside a scope reads captures that are not resident through the copying path") {
    val n                        = 16
    val xs                       = counting(n)
    val ys                       = counting(n)
    val resident: Buffer[Double] = xs
    val copied: Buffer[Double]   = ys
    Host.context.resident(resident) {
      var outcome: Either[Throwable, Unit] = null
      compiletime.offload0(Host.context, outcome = _) {
        var i = 0
        while (i < n) {
          resident(i) = resident(i) + copied(i)
          i += 1
        }
        ()
      }
      outcome.fold(throw _, identity)
    }
    assertEquals(values(xs), (0 until n).map(_.toDouble * 2).toList)
    assertEquals(xs.reads, 0)
    assertEquals(ys.reads, n)
  }

  test("a launch on another context than the resident one fails instead of copying stale data") {
    val n                        = 8
    val xs                       = counting(n)
    val captured: Buffer[Double] = xs
    val host  = rt.Context.platforms().flatMap(_.devices).find(d => d.platformKind == 1 && d.moduleFormat == 2).get
    val other = rt.Context.create(host)
    try {
      var outcome: Either[Throwable, Unit] = null
      Host.context.resident(captured) {
        compiletime.offload0(other, outcome = _) {
          var i = 0
          while (i < n) {
            captured(i) = captured(i) + 1.0
            i += 1
          }
          ()
        }
      }
      assert(outcome.left.exists(_.getMessage.contains("is resident on")), String.valueOf(outcome))
      assertEquals(values(xs), (0 until n).map(_.toDouble).toList)
    } finally other.close()
  }

  private class CountingInts(backing: ByteBuffer, id: Long)
      extends Buffer.IntBuffer(backing, backing.asIntBuffer(), id) {
    var reads                                 = 0
    var writes                                = 0
    override def apply(i: Int): Int           = { reads += 1; buffer.get(i) }
    override def update(i: Int, x: Int): Unit = { writes += 1; buffer.put(i, x); () }
  }

  private inline def bumpInts(xs: Buffer[Int], n: Int): Unit = {
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0(Host.context, outcome = _) {
      var i = 0
      while (i < n) {
        xs(i) = xs(i) + 1
        i += 1
      }
      ()
    }
    outcome.fold(throw _, identity)
  }

  test("int buffers are resident across launches too") {
    val n  = 32
    val xs = new CountingInts(ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder), ids.getAndIncrement)
    (0 until n).foreach(i => xs.buffer.put(i, i))
    val captured: Buffer[Int] = xs
    Host.context.resident(captured) {
      (0 until 4).foreach(_ => bumpInts(captured, n))
    }
    assertEquals((0 until n).map(xs.buffer.get).toList, (0 until n).map(_ + 4).toList)
    assertEquals(xs.reads, 0)
    assertEquals(xs.writes, 0)
  }
}
