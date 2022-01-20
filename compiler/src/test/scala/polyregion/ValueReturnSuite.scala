package polyregion

import polyregion.compileTime._
import scala.compiletime._

class ValueReturnSuite extends munit.FunSuite {

  final val Enable = true

  def assertValEquals[A](expected: A, actual: A): Unit = (expected.asMatchable, actual.asMatchable) match {
    case (e: Float, a: Float) => //
      assertEquals(java.lang.Float.floatToIntBits(e), java.lang.Float.floatToIntBits(a))
    case (e: Double, a: Double) => //
      assertEquals(java.lang.Double.doubleToLongBits(e), java.lang.Double.doubleToLongBits(a))
    case (e, a) => //
      assertEquals(e, a)
  }

  inline def assertOffload[A](inline f: => A) =
    assertValEquals(offload[A](f), f)

  inline def testValueReturn[A](inline r: A) = if (Enable) {
    test(s"${r.getClass}-const=$r")(assertOffload[A](r))
    val x: A = r
    test(s"${r.getClass}-ref1=$r")(assertOffload[A](x))
    val y: A = x
    test(s"${r.getClass}-ref2=$r")(assertOffload[A](y))
  }

  testValueReturn[Char](0)
  testValueReturn[Char](255)
  testValueReturn[Char](1)
  testValueReturn[Char](0)
  testValueReturn[Char](42)

  testValueReturn[Byte](-128)
  testValueReturn[Byte](127)
  testValueReturn[Byte](1)
  testValueReturn[Byte](-1)
  testValueReturn[Byte](0)
  testValueReturn[Byte](42)

  testValueReturn[Short](-32768)
  testValueReturn[Short](32767)
  testValueReturn[Short](1)
  testValueReturn[Short](-1)
  testValueReturn[Short](0)
  testValueReturn[Short](42)

  testValueReturn[Int](0x80000000)
  testValueReturn[Int](0x7fffffff)
  testValueReturn[Int](1)
  testValueReturn[Int](-1)
  testValueReturn[Int](0)
  testValueReturn[Int](42)

  testValueReturn[Long](0x8000000000000000L)
  testValueReturn[Long](0x7fffffffffffffffL)
  testValueReturn[Long](1)
  testValueReturn[Long](-1)
  testValueReturn[Long](0)
  testValueReturn[Long](42)

  testValueReturn[Float](1.4e-45f)
  testValueReturn[Float](1.17549435e-38f)
  testValueReturn[Float](3.4028235e+38) // XXX 3.4028235e+38f appears to not fit!
  testValueReturn[Float](0.0f / 0.0f)
  testValueReturn[Float](1)
  testValueReturn[Float](-1)
  testValueReturn[Float](0)
  testValueReturn[Float](42)
  testValueReturn[Float](3.14159265358979323846)

  testValueReturn[Double](4.9e-324d)
  testValueReturn[Double](2.2250738585072014e-308d)
  testValueReturn[Double](1.7976931348623157e+308d)
  testValueReturn[Double](0.0f / 0.0d)
  testValueReturn[Double](1)
  testValueReturn[Double](-1)
  testValueReturn[Double](0)
  testValueReturn[Double](42)
  testValueReturn[Double](3.14159265358979323846d)

  testValueReturn(true)
  testValueReturn(false)

  testValueReturn[Unit](())

  // test("while inc")(assertOffload {
  //   val lim = 10
  //   var i   = 0
  //   while (i < lim) i += 1
  //   i
  // })

  // test("copy capture") {
  //   val a = 1
  //   assertOffload(a)
  // }

  // test("while inc")(assertOffload {
  //   var i = 1
  //   while (i < 10) i += 1
  //   i
  // })
  // case class V(n : Int)
  //  val a : Int = 42
  //  def inv = 42

  // def go = {
  //   // Select
  //    val b : Int = 42
  //    val c : Int = 42
  //    val v = V(1)
  // offload{
  //   // var counter = 1
  //   // while (counter < 10) counter += 1
  //   val in4 = v.n
  //   val in3 = Int.MaxValue
  //   val inInv = inv
  //   val in2 = a
  //   val in1 = b
  //   val u = in3
  //   val u2 = in4
  //   ()
  // }
  // }

  // test("statements")(assertOffload { 1; 2 })

  // test("const cond") {
  //   assertEquals(offload(if (true) 42 else 69), 42)
  //   assertEquals(offload(if (false) 42 else 69), 69)
  // }

  // test("copy capture as cond test") {
  //   var a = true
  //   assertEquals(offload(if (a) 42 else 69), 42)
  //   a = false
  //   assertEquals(offload(if (a) 42 else 69), 69)
  // }

  // test("copy capture expr as cond test") {
  //   var a = 10
  //   assertEquals(offload(if (a == 10) 42 else 69), 42)
  //   a = 0
  //   assertEquals(offload(if (a == 10) 42 else 69), 69)
  // }

  // test("int math expr 1")(assertOffload(42 + 69))

}
