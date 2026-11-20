package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import scala.collection.mutable

class TwinSuite extends munit.FunSuite {

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

  private inline def launch[C](inline ctx: rt.Context)(inline f: Unit): Unit = {
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0[C](ctx, outcome = _)(f)
    outcome.fold(throw _, identity)
  }

  private def withContext(label: String, device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit = device match {
    case None => assume(false, s"no $label device in the catalogue")
    case Some(d) =>
      val ctx = rt.Context.create(d)
      try f(ctx)
      finally ctx.close()
  }

  private inline def cases[C](label: String, device: Option[rt.DeviceInfo]): Unit = {

    test(s"[$label] nested case classes are read inline by the kernel") {
      withContext(label, device) { ctx =>
        val s   = Segment(Point(1.0, 2.0), Point(4.0, 6.0))
        val out = Buffer.ofDim[Double](1)
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
          out(0) = h.get + h.k
        }
        assertEquals(out(0), 42)
      }
    }

    test(s"[$label] a sequence element is rebuilt with its nested structs") {
      withContext(label, device) { ctx =>
        val segs = mutable.ListBuffer(Segment(Point(1.0, 2.0), Point(3.0, 4.0)))
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
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
        launch[C](ctx) {
          out(0) = shy.first + 1.0
        }
        assertEquals(out(0), 5.0)
      }
    }

    test(s"[$label] a class holding only a buffer round trips") {
      withContext(label, device) { ctx =>
        val sink = Sink(Buffer.ofDim[Double](3))
        launch[C](ctx) {
          sink.out(2) = 9.0
        }
        assertEquals(sink.out(2), 9.0)
      }
    }
  }

  cases[Config[Target.Host.type, Opt.O2]]("host", host)
  cases[Config[Target.NVPTX64["sm_80"], Opt.O2]]("nvidia", nvidia)
}
