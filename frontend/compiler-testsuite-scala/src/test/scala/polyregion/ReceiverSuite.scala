package polyregion

import polyregion.jvm.runtime as rt
import polyregion.jvm.twin.TwinException
import polyregion.scalalang.*

object ReceiverSuite {
  case class Counter(var n: Int) {
    def inc(): Unit       = n += 1
    def add(k: Int): Unit = n += k
    def twice(): Unit     = { inc(); inc() }
    def get: Int          = n
  }
  class Plain(var total: Double) {
    def bump(by: Double): Unit = total += by
  }
  class Outer(val inner: Counter, var calls: Int) {
    def touch(): Unit = { calls += 1; inner.inc() }
  }
}

class ReceiverSuite extends munit.FunSuite {

  import ReceiverSuite.*

  private val devices = rt.Context.platforms().flatMap(_.devices)
  private val host    = devices.find(d => d.platformKind == 1 && d.moduleFormat == 2)
  private val nvidia  = devices.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0)

  private inline def launch(inline ctx: rt.Context)(inline f: Unit): Unit = {
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0(ctx, outcome = _)(f)
    outcome.fold(throw _, identity)
  }

  private def withContext(label: String, device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit = device match {
    case None => assume(false, s"no $label device in the catalogue")
    case Some(d) =>
      val ctx = rt.Context.create(d)
      try f(ctx)
      finally ctx.close()
  }

  class Worker(var count: Int, val step: Int) {
    private var secret: Int = 0
    def secretValue: Int    = secret

    def runPublic(ctx: rt.Context): Unit = launch(ctx) {
      count = count + step
    }

    def runPrivate(ctx: rt.Context): Unit = launch(ctx) {
      secret = secret + step
    }

    def runViaMethod(ctx: rt.Context): Unit = launch(ctx) {
      bump()
      bump()
    }

    def bump(): Unit = count += step
  }

  private inline def cases(label: String, device: Option[rt.DeviceInfo]): Unit = {

    test(s"[$label] a method on a captured case class mutates the host instance") {
      withContext(label, device) { ctx =>
        val c = Counter(10)
        launch(ctx) {
          c.inc()
        }
        assertEquals(c.n, 11)
      }
    }

    test(s"[$label] a method with an argument on a captured case class mutates the host instance") {
      withContext(label, device) { ctx =>
        val c = Counter(1)
        launch(ctx) {
          c.add(41)
        }
        assertEquals(c.n, 42)
      }
    }

    test(s"[$label] a method calling another mutating method on a captured case class") {
      withContext(label, device) { ctx =>
        val c = Counter(0)
        launch(ctx) {
          c.twice()
        }
        assertEquals(c.n, 2)
      }
    }

    test(s"[$label] a method on a captured plain class mutates the host instance") {
      withContext(label, device) { ctx =>
        val p = Plain(1.5)
        launch(ctx) {
          p.bump(2.25)
        }
        assertEquals(p.total, 3.75)
      }
    }

    test(s"[$label] a method on a nested receiver mutates the inner instance") {
      withContext(label, device) { ctx =>
        val inner = Counter(5)
        val o     = Outer(inner, 0)
        launch(ctx) {
          o.inner.inc()
        }
        assertEquals(inner.n, 6)
        assertEquals(o.calls, 0)
      }
    }

    test(s"[$label] a method on the outer receiver mutating its own and a nested receiver") {
      withContext(label, device) { ctx =>
        val inner = Counter(5)
        val o     = Outer(inner, 0)
        launch(ctx) {
          o.touch()
        }
        assertEquals(inner.n, 6)
        assertEquals(o.calls, 1)
      }
    }

    test(s"[$label] a method on a receiver inside a loop accumulates") {
      withContext(label, device) { ctx =>
        val c = Counter(0)
        launch(ctx) {
          var i = 0
          while (i < 10) {
            c.add(i)
            i += 1
          }
        }
        assertEquals(c.n, 45)
      }
    }

    test(s"[$label] a captured receiver read after mutation in the same region") {
      withContext(label, device) { ctx =>
        val c   = Counter(3)
        val out = Buffer.ofDim[Int](1)
        launch(ctx) {
          c.inc()
          c.inc()
          out(0) = c.get
        }
        assertEquals(out(0), 5)
        assertEquals(c.n, 5)
      }
    }

    test(s"[$label] two captured receivers of one class stay distinct") {
      withContext(label, device) { ctx =>
        val a = Counter(0)
        val b = Counter(100)
        launch(ctx) {
          a.inc()
          b.add(5)
        }
        assertEquals(a.n, 1)
        assertEquals(b.n, 105)
      }
    }

    test(s"[$label] a method on a counter built inside the region mutates it") {
      withContext(label, device) { ctx =>
        val out = Buffer.ofDim[Int](1)
        launch(ctx) {
          val c = Counter(40)
          c.inc()
          c.add(1)
          out(0) = c.n
        }
        assertEquals(out(0), 42)
      }
    }

    test(s"[$label] a mutable object captured on its own and nested in another capture is refused") {
      withContext(label, device) { ctx =>
        val inner = Counter(0)
        val o     = Outer(inner, 0)
        val e = intercept[TwinException] {
          launch(ctx) {
            inner.inc()
            o.inner.inc()
          }
        }
        assert(e.getMessage.contains("by value"), e.getMessage)
        assertEquals(inner.n, 0)
      }
    }

    test(s"[$label] a class method offloading mutates its public var") {
      withContext(label, device) { ctx =>
        val w = Worker(1, 4)
        w.runPublic(ctx)
        assertEquals(w.count, 5)
      }
    }

    test(s"[$label] a class method offloading mutates its private var") {
      withContext(label, device) { ctx =>
        val w = Worker(1, 4)
        w.runPrivate(ctx)
        assertEquals(w.secretValue, 4)
      }
    }

    test(s"[$label] a class method offloading calls its own mutating method") {
      withContext(label, device) { ctx =>
        val w = Worker(1, 4)
        w.runViaMethod(ctx)
        assertEquals(w.count, 9)
      }
    }
  }

  cases("host", host)
  cases("nvidia", nvidia)
}
