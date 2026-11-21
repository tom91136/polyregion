package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

class ArrayMemberSuite extends munit.FunSuite {

  case class Ints(xs: Array[Int], n: Int)
  case class Doubles(xs: Array[Double], scale: Double)
  case class Longs(xs: Array[Long])
  case class Bools(xs: Array[Boolean])
  case class Twin(a: Array[Int], b: Array[Int])
  case class Rows(rows: Array[Array[Int]])
  case class Point(x: Int, y: Int)
  case class Points(ps: Array[Point])
  class Mutable(var xs: Array[Int])
  case class Outer(inner: Ints, tag: Int)
  case class Mixed(xs: Array[Int], buffer: Buffer[Int])

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

    test(s"[$label] an int array member is read by the kernel") {
      withContext(label, device) { ctx =>
        val h   = Ints(Array(10, 20, 30, 40), 5)
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = h.xs(2) + h.xs(3) + h.n
        }
        assertEquals(out(0), 75)
      }
    }

    test(s"[$label] an int array member is written back") {
      withContext(label, device) { ctx =>
        val h = Ints(Array(1, 2, 3, 4), 4)
        launch[C](ctx) {
          var i = 0
          while (i < h.n) {
            h.xs(i) = h.xs(i) * 10 + i
            i += 1
          }
        }
        assertEquals(h.xs.toList, List(10, 21, 32, 43))
      }
    }

    test(s"[$label] a double array member is read and written back") {
      withContext(label, device) { ctx =>
        val h = Doubles(Array(0.5, 1.5, 2.5), 4.0)
        launch[C](ctx) {
          var i = 0
          while (i < 3) {
            h.xs(i) = h.xs(i) * h.scale
            i += 1
          }
        }
        assertEquals(h.xs.toList, List(2.0, 6.0, 10.0))
      }
    }

    test(s"[$label] a long array member is read and written back") {
      withContext(label, device) { ctx =>
        val h = Longs(Array(1L << 40, 7L))
        launch[C](ctx) {
          h.xs(1) = h.xs(0) + h.xs(1)
        }
        assertEquals(h.xs.toList, List(1L << 40, (1L << 40) + 7L))
      }
    }

    test(s"[$label] a boolean array member is written back") {
      withContext(label, device) { ctx =>
        val h = Bools(Array(false, false, true))
        launch[C](ctx) {
          h.xs(0) = !h.xs(1)
          h.xs(2) = false
        }
        assertEquals(h.xs.toList, List(true, false, false))
      }
    }

    test(s"[$label] an array reachable through two members is staged once") {
      withContext(label, device) { ctx =>
        val shared = Array(1, 2, 3)
        val t      = Twin(shared, shared)
        launch[C](ctx) {
          t.a(0) = 100
          t.b(1) = t.a(0) + 1
        }
        assertEquals(shared.toList, List(100, 101, 3))
      }
    }

    test(s"[$label] an empty array member does not break the launch") {
      withContext(label, device) { ctx =>
        val h   = Ints(Array.empty[Int], 9)
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = h.n
        }
        assertEquals(out(0), 9)
      }
    }

    test(s"[$label] a null array member that the kernel does not touch is accepted") {
      withContext(label, device) { ctx =>
        val h   = Ints(null, 3)
        val out = Buffer.ofDim[Int](1)
        launch[C](ctx) {
          out(0) = h.n + 1
        }
        assertEquals(out(0), 4)
        assertEquals(h.xs, null)
      }
    }

    test(s"[$label] an array member of a nested case class is read and written back") {
      withContext(label, device) { ctx =>
        val o = Outer(Ints(Array(5, 6, 7), 3), 1)
        launch[C](ctx) {
          o.inner.xs(0) = o.inner.xs(2) + o.tag
        }
        assertEquals(o.inner.xs.toList, List(8, 6, 7))
      }
    }

    test(s"[$label] an array member beside a buffer member") {
      withContext(label, device) { ctx =>
        val m = Mixed(Array(1, 2, 3), Buffer.ofDim[Int](3))
        launch[C](ctx) {
          var i = 0
          while (i < 3) {
            m.buffer(i) = m.xs(i) * 2
            m.xs(i) = m.xs(i) + 1
            i += 1
          }
        }
        assertEquals(m.xs.toList, List(2, 3, 4))
        assertEquals((0 until 3).map(m.buffer(_)).toList, List(2, 4, 6))
      }
    }

    test(s"[$label] an array member of a class with a var field is written back") {
      withContext(label, device) { ctx =>
        val m = Mutable(Array(3, 4))
        launch[C](ctx) {
          m.xs(0) = m.xs(1)
        }
        assertEquals(m.xs.toList, List(4, 4))
      }
    }

    test(s"[$label] a nested array member is read and written back") {
      withContext(label, device) { ctx =>
        val r = Rows(Array(Array(1, 2), Array(3, 4)))
        launch[C](ctx) {
          r.rows(1)(0) = r.rows(0)(1) + 10
        }
        assertEquals(r.rows.map(_.toList).toList, List(List(1, 2), List(12, 4)))
      }
    }
  }

  test("an array member with struct elements is rejected at compile time naming the member") {
    val errors = compileErrors("""
      val h = Points(Array(Point(1, 2)))
      compiletime.offload0[Config[Target.Host.type, Opt.O2]](rt.Context.create(host.get), _ => ())(h.ps(0) = Point(3, 4))
    """)
    assert(errors.contains("`ps`"), errors)
  }

  cases[Config[Target.Host.type, Opt.O2]]("host", host)
  cases[Config[Target.NVPTX64["sm_80"], Opt.O2]]("nvidia", nvidia)
}
