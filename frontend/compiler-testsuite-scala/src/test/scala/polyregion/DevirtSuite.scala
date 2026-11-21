package polyregion

import polyregion.DevirtFixtures.{Op, Outsider}
import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import scala.collection.mutable

object DevirtSuite {
  sealed trait Shape {
    def area: Double
  }
  case class Circle(r: Double)          extends Shape { def area: Double = 3.0 * r * r }
  case class Rect(w: Double, h: Double) extends Shape { def area: Double = w * h       }
  case class Square(s: Double)          extends Shape { def area: Double = s * s       }

  sealed trait Counts {
    def total: Int
  }
  case class Pair(a: Int, b: Int)           extends Counts { def total: Int = a + b     }
  case class Triple(a: Int, b: Int, c: Int) extends Counts { def total: Int = a * b * c }

  sealed trait Animal {
    def legs: Int
  }
  sealed trait Pet            extends Animal
  case class Dog(age: Int)    extends Pet    { def legs: Int = 4 + age * 0       }
  case class Cat(lives: Int)  extends Pet    { def legs: Int = 4 + lives * 0 + 1 }
  case class Bird(wings: Int) extends Animal { def legs: Int = 2 + wings * 0     }

  class Inc(val n: Int) extends Op {
    def apply(x: Int): Int = x + n
  }
  class Scale(val k: Int) extends Op {
    def apply(x: Int): Int = x * k
  }
}

class DevirtSuite extends munit.FunSuite {

  import DevirtSuite.*

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

    test(s"[$label] sealed trait elements dispatch to their own class") {
      withContext(label, device) { ctx =>
        val shapes = mutable.ListBuffer[Shape](Circle(1.0), Rect(2.0, 3.0), Square(4.0), Circle(2.0), Rect(5.0, 1.0))
        val out    = Buffer.ofDim[Double](5)
        launch[C](ctx) {
          var i = 0
          while (i < 5) {
            out(i) = shapes(i).area
            i += 1
          }
        }
        assertEquals((0 until 5).map(out(_)).toList, List(3.0, 6.0, 16.0, 12.0, 5.0))
      }
    }

    test(s"[$label] trait elements of classes in the compilation unit dispatch") {
      withContext(label, device) { ctx =>
        val ops = mutable.ListBuffer[Op](Inc(1), Scale(3), Inc(10), Scale(2))
        val out = Buffer.ofDim[Int](4)
        launch[C](ctx) {
          var i = 0
          while (i < 4) {
            out(i) = ops(i)(i + 5)
            i += 1
          }
        }
        assertEquals((0 until 4).map(out(_)).toList, List(6, 18, 17, 16))
      }
    }

    test(s"[$label] candidates with 4 byte fields keep them in the shared slot") {
      withContext(label, device) { ctx =>
        val xs  = mutable.ListBuffer[Counts](Pair(3, 4), Triple(2, 5, 7), Pair(10, 20))
        val out = Buffer.ofDim[Int](3)
        launch[C](ctx) {
          var i = 0
          while (i < 3) {
            out(i) = xs(i).total
            i += 1
          }
        }
        assertEquals((0 until 3).map(out(_)).toList, List(7, 70, 30))
      }
    }

    test(s"[$label] a receiver typed by an intermediate trait dispatches over its own descendants") {
      withContext(label, device) { ctx =>
        val pets = mutable.ListBuffer[Pet](Dog(1), Cat(9), Dog(2))
        val out  = Buffer.ofDim[Int](3)
        launch[C](ctx) {
          var i = 0
          while (i < 3) {
            out(i) = pets(i).legs
            i += 1
          }
        }
        assertEquals((0 until 3).map(out(_)).toList, List(4, 5, 4))
      }
    }

    test(s"[$label] a class outside the candidate set raises on the device") {
      withContext(label, device) { ctx =>
        val ops = mutable.ListBuffer[Op](Inc(1), Outsider(7))
        val out = Buffer.ofDim[Int](2)
        val raised = intercept[Throwable] {
          launch[C](ctx) {
            var i = 0
            while (i < 2) {
              out(i) = ops(i)(3)
              i += 1
            }
          }
        }
        assert(raised.getMessage.contains("receiver has no implementation of"), raised.getMessage)
        assertEquals(out(1), 0)
      }
    }
  }

  cases[Config[Target.Host.type, Opt.O2]]("host", host)
  cases[Config[Target.NVPTX64["sm_80"], Opt.O2]]("nvidia", nvidia)
}
