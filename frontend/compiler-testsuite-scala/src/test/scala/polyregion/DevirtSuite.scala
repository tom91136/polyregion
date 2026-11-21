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

  sealed trait Sized {
    def size: Int
  }
  case class Plain(n: Int) extends Sized { def size: Int = n }
  case class Box[T](v: T)  extends Sized { def size: Int = 1 }

  sealed trait Held {
    def size: Int
  }
  case class Direct(n: Int) extends Held { def size: Int = n }
  class Holder[T] {
    class Inner extends Held { def size: Int = 1 }
  }

  abstract class Gauge {
    def read(x: Int): Int
  }
  class Doubler(val base: Int) extends Gauge { def read(x: Int): Int = base + x * 2   }
  class Adder(val base: Int)   extends Gauge { def read(x: Int): Int = base + x + 100 }

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

  private inline def cases(label: String, device: Option[rt.DeviceInfo]): Unit = {

    test(s"[$label] sealed trait elements dispatch to their own class") {
      withContext(label, device) { ctx =>
        val shapes = mutable.ListBuffer[Shape](Circle(1.0), Rect(2.0, 3.0), Square(4.0), Circle(2.0), Rect(5.0, 1.0))
        val out    = Buffer.ofDim[Double](5)
        launch(ctx) {
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
        launch(ctx) {
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
        launch(ctx) {
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
        launch(ctx) {
          var i = 0
          while (i < 3) {
            out(i) = pets(i).legs
            i += 1
          }
        }
        assertEquals((0 until 3).map(out(_)).toList, List(4, 5, 4))
      }
    }

    test(s"[$label] subclasses of an abstract class dispatch") {
      withContext(label, device) { ctx =>
        val gauges = mutable.ListBuffer[Gauge](new Doubler(1), new Adder(2), new Doubler(3))
        val out    = Buffer.ofDim[Int](3)
        launch(ctx) {
          var i = 0
          while (i < 3) {
            out(i) = gauges(i).read(10)
            i += 1
          }
        }
        assertEquals((0 until 3).map(out(_)).toList, List(21, 112, 23))
      }
    }

    test(s"[$label] a class outside the candidate set raises on the device") {
      withContext(label, device) { ctx =>
        val ops = mutable.ListBuffer[Op](Inc(1), Outsider(7))
        val out = Buffer.ofDim[Int](2)
        val raised = intercept[Throwable] {
          launch(ctx) {
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

  test("a generic candidate class is rejected at compile time") {
    val errors = compileErrors(
      """{
        val xs  = mutable.ListBuffer[Sized](Plain(1), Box(2.0))
        val out = Buffer.ofDim[Int](2)
        launch(null) {
          var i = 0
          while (i < 2) {
            out(i) = xs(i).size
            i += 1
          }
        }
      }"""
    )
    assert(errors.contains("Cannot devirtualise generic candidate classes"), errors)
    assert(errors.contains("DevirtSuite$.Box"), errors)
  }

  test("a class nested in a generic outer class is rejected as a candidate") {
    val errors = compileErrors(
      """{
        val xs  = mutable.ListBuffer[Held](Direct(1), Direct(2))
        val out = Buffer.ofDim[Int](2)
        launch(null) {
          var i = 0
          while (i < 2) {
            out(i) = xs(i).size
            i += 1
          }
        }
      }"""
    )
    assert(errors.contains("Cannot devirtualise generic candidate classes"), errors)
    assert(errors.contains("Holder"), errors)
  }

  cases("host", host)
  cases("nvidia", nvidia)
}
