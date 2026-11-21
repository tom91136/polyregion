package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import scala.collection.mutable

object TwinSuite {
  class Lazy {
    val k: Int                   = 4
    var forced: Int              = 0
    lazy val heavy: Int          = { forced += 1; k * 100 }
    private lazy val hidden: Int = { forced += 1; k * 10 }
    def peek: Int                = hidden
  }
  class Local {
    val k: Int         = 5
    private val n: Int = k * 3
    def get: Int       = n
  }
  trait Labelled {
    val label: Int          = 30
    private val secret: Int = 4
    def reveal: Int         = secret
  }
  class Tagged extends Labelled {
    val k: Int = 2
  }
  class Outer {
    val base: Int = 100
    class Nested {
      val k: Int     = 7
      def total: Int = base + k
    }
    def nested: Nested = Nested()
  }
  class Expanded {
    private val a: Int = 1
    val b: Int         = 2
    def viaLocal: Int = {
      class Reader { def get: Int = a }
      Reader().get
    }
  }
}

class TwinSuite extends munit.FunSuite {

  import TwinSuite.*

  case class Point(x: Double, y: Double)
  case class Segment(from: Point, to: Point)
  class Body(var mass: Double, val pos: Point, var steps: Int)
  class Hidden(private val n: Int, val k: Int) {
    def get: Int = n
  }
  case class Pair(a: Buffer[Double], b: Buffer[Double])
  case class Cloud(points: Buffer[Double], scale: Double)
  class Sink(val out: Buffer[Double])
  class Cell(var v: Double)
  class Shy(private val xs: Buffer[Double]) {
    def first: Double = xs(0)
  }
  private val devices = rt.Context.platforms().flatMap(_.devices)
  private val host    = devices.find(d => d.platformKind == 1 && d.moduleFormat == 2)
  private val nvidia  = devices.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0)

  private inline def launch(inline targets: String, inline ctx: rt.Context)(inline f: Unit): Unit = {
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0(ctx, outcome = _, targets)(f)
    outcome.fold(throw _, identity)
  }

  private def withContext(label: String, device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit = device match {
    case None => assume(false, s"no $label device in the catalogue")
    case Some(d) =>
      val ctx = rt.Context.create(d)
      try f(ctx)
      finally ctx.close()
  }

  private inline def cases(inline targets: String, label: String, device: Option[rt.DeviceInfo]): Unit = {

    test(s"[$label] nested case classes are read inline by the kernel") {
      withContext(label, device) { ctx =>
        val s   = Segment(Point(1.0, 2.0), Point(4.0, 6.0))
        val out = Buffer.ofDim[Double](1)
        launch(targets, ctx) {
          val dx = s.to.x - s.from.x
          val dy = s.to.y - s.from.y
          out(0) = dx * dx + dy * dy
        }
        assertEquals(out(0), 25.0)
      }
    }

    test(s"[$label] mutable primitive fields are written back and nested objects stay the same instance") {
      withContext(label, device) { ctx =>
        val pos  = Point(2.0, 3.0)
        val body = Body(1.5, pos, 0)
        launch(targets, ctx) {
          body.mass = body.mass + body.pos.x
          body.steps = body.steps + 7
        }
        assertEquals(body.mass, 3.5)
        assertEquals(body.steps, 7)
        assert(body.pos eq pos)
      }
    }

    test(s"[$label] a buffer inside a case class is staged and written back") {
      withContext(label, device) { ctx =>
        val points = Buffer.ofDim[Double](8)
        (0 until 8).foreach(i => points(i) = i.toDouble)
        val c = Cloud(points, 2.0)
        launch(targets, ctx) {
          var i = 0
          while (i < 8) {
            c.points(i) = c.points(i) * c.scale
            i += 1
          }
        }
        assertEquals((0 until 8).map(points(_)).toList, (0 until 8).map(_ * 2.0).toList)
      }
    }

    test(s"[$label] a buffer reachable through two fields aliases") {
      withContext(label, device) { ctx =>
        val shared = Buffer.ofDim[Double](4)
        val p      = Pair(shared, shared)
        launch(targets, ctx) {
          p.a(0) = 5.0
          p.b(0) = p.b(0) + 1.0
        }
        assertEquals(shared(0), 6.0)
      }
    }

    test(s"[$label] a private immutable field is read by the kernel") {
      withContext(label, device) { ctx =>
        val h   = Hidden(40, 2)
        val out = Buffer.ofDim[Int](1)
        launch(targets, ctx) {
          out(0) = h.get + h.k
        }
        assertEquals(out(0), 42)
      }
    }

    test(s"[$label] a sequence element is rebuilt with its nested structs") {
      withContext(label, device) { ctx =>
        val segs = mutable.ListBuffer(Segment(Point(1.0, 2.0), Point(3.0, 4.0)))
        launch(targets, ctx) {
          segs(0) = Segment(Point(5.0, 6.0), Point(7.0, 8.0))
        }
        assertEquals(segs.toList, List(Segment(Point(5.0, 6.0), Point(7.0, 8.0))))
      }
    }

    test(s"[$label] an element of a captured sequence is the same object as its second reference") {
      withContext(label, device) { ctx =>
        val cells = mutable.ListBuffer(Cell(1.0), Cell(2.0))
        val first = cells(0)
        val out   = Buffer.ofDim[Double](1)
        launch(targets, ctx) {
          val before = cells(0).v
          first.v = before + 8.0
          out(0) = cells(0).v
        }
        assertEquals(out(0), 9.0)
      }
    }

    test(s"[$label] a buffer captured on its own and inside a case class is one buffer") {
      withContext(label, device) { ctx =>
        val points = Buffer.ofDim[Double](4)
        val c      = Cloud(points, 1.0)
        launch(targets, ctx) {
          points(0) = 3.0
          c.points(1) = points(0) + 1.0
        }
        assertEquals((points(0), points(1)), (3.0, 4.0))
      }
    }

    test(s"[$label] a case class captured before the buffer it shares is one buffer") {
      withContext(label, device) { ctx =>
        val points = Buffer.ofDim[Double](4)
        val c      = Cloud(points, 1.0)
        launch(targets, ctx) {
          c.points(1) = 5.0
          points(0) = c.points(1) + 1.0
        }
        assertEquals((points(0), points(1)), (6.0, 5.0))
      }
    }

    test(s"[$label] a private buffer member is staged through the fallback") {
      withContext(label, device) { ctx =>
        val xs = Buffer.ofDim[Double](2)
        xs(0) = 4.0
        val shy = Shy(xs)
        val out = Buffer.ofDim[Double](1)
        launch(targets, ctx) {
          out(0) = shy.first + 1.0
        }
        assertEquals(out(0), 5.0)
      }
    }

    test(s"[$label] a lazy val the kernel never reads is not marshalled") {
      withContext(label, device) { ctx =>
        val l   = Lazy()
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = l.k + 1
        }
        assertEquals(out(0), 5)
        assertEquals(l.forced, 0)
      }
    }

    test(s"[$label] a private lazy val the kernel never reads is not marshalled") {
      withContext(label, device) { ctx =>
        val l   = Lazy()
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = l.k + 2
        }
        assertEquals(out(0), 6)
        assertEquals(l.forced, 0)
      }
    }

    test(s"[$label] a forced lazy val is read as a local") {
      withContext(label, device) { ctx =>
        val l     = Lazy()
        val heavy = l.heavy
        val out   = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = heavy + l.k
        }
        assertEquals(out(0), 404)
        assertEquals(l.forced, 1)
      }
    }

    test(s"[$label] an object-private val is read by the kernel") {
      withContext(label, device) { ctx =>
        val l   = Local()
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = l.get + l.k
        }
        assertEquals(out(0), 20)
      }
    }

    test(s"[$label] a class mixing in a trait with vals is captured") {
      withContext(label, device) { ctx =>
        val t   = Tagged()
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = t.k + 1
        }
        assertEquals(out(0), 3)
      }
    }

    test(s"[$label] an inner class instance is captured without its outer pointer") {
      withContext(label, device) { ctx =>
        val n   = Outer().nested
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = n.k + 1
        }
        assertEquals(out(0), 8)
      }
    }

    test(s"[$label] a private val renamed for a nested class is read by the kernel") {
      withContext(label, device) { ctx =>
        val c   = Expanded()
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = c.b + 1
        }
        assertEquals(out(0), 3)
      }
    }

    test(s"[$label] a class holding only a buffer round trips") {
      withContext(label, device) { ctx =>
        val sink = Sink(Buffer.ofDim[Double](3))
        launch(targets, ctx) {
          sink.out(2) = 9.0
        }
        assertEquals(sink.out(2), 9.0)
      }
    }
  }

  test("a kernel reading a lazy val of a captured object is rejected naming the member") {
    val errors = compileErrors("""
      val l   = Lazy()
      val out = Buffer.ofDim[Int](1)
      compiletime.offload0(rt.Context.create(host.get), _ => (), "host@native")(out(0) = l.heavy)
    """)
    assert(errors.contains("lazy val `heavy`"), errors)
  }

  test("a method reading a lazy val of a captured object is rejected naming the member") {
    val errors = compileErrors("""
      val l   = Lazy()
      val out = Buffer.ofDim[Int](1)
      compiletime.offload0(rt.Context.create(host.get), _ => (), "host@native")(out(0) = l.peek)
    """)
    assert(errors.contains("lazy val `hidden`"), errors)
  }

  cases("host@native", "host", host)
  cases("cuda@sm_80", "nvidia", nvidia)
}
