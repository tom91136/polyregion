package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag
import polyregion.NativeStruct

class InlineArraySuite extends BaseSuite {

  inline def testExpr(inline name: String)(inline r: Any) = if (Toggles.InlineArraySuite) {
    test(name)(assertOffload(r))
  }

  inline val FillN = 20

  inline def assertInlineFill[A <: AnyVal](inline n: Int, inline expected: A)(using C: ClassTag[A]) =
    if (Toggles.InlineArraySuite) {
      test(s"${C.runtimeClass}-fill-x$n=$expected") {
        val actual = doOffload {
          val xs = Array.ofDim[A](n)
          unrollInclusive(n - 1)(i => xs(i) = expected)
          xs
        }
        actual.foreach(x => assertValEquals(x, expected))
      }
    }

  assertInlineFill[Char](0, 0)
  assertInlineFill[Byte](0, 0)
  assertInlineFill[Short](0, 0)
  assertInlineFill[Int](0, 0)
  assertInlineFill[Long](0, 0)
  assertInlineFill[Float](0, 0)
  assertInlineFill[Double](0, 0)
  assertInlineFill[Boolean](0, false)

  assertInlineFill[Byte](FillN, -128)
  assertInlineFill[Byte](FillN, 127)
  assertInlineFill[Byte](FillN, 1)
  assertInlineFill[Byte](FillN, -1)
  assertInlineFill[Byte](FillN, 0)
  assertInlineFill[Byte](FillN, 42)

  assertInlineFill[Char](FillN, '\u0000')
  assertInlineFill[Char](FillN, '\uFFFF')
  assertInlineFill[Char](FillN, 1)
  assertInlineFill[Char](FillN, 0)
  assertInlineFill[Char](FillN, 42)

  assertInlineFill[Short](FillN, -32768)
  assertInlineFill[Short](FillN, 32767)
  assertInlineFill[Short](FillN, 1)
  assertInlineFill[Short](FillN, -1)
  assertInlineFill[Short](FillN, 0)
  assertInlineFill[Short](FillN, 42)

  assertInlineFill[Int](FillN, 0x80000000)
  assertInlineFill[Int](FillN, 0x7fffffff)
  assertInlineFill[Int](FillN, 1)
  assertInlineFill[Int](FillN, -1)
  assertInlineFill[Int](FillN, 0)
  assertInlineFill[Int](FillN, 42)

  assertInlineFill[Long](FillN, 0x8000000000000000L)
  assertInlineFill[Long](FillN, 0x7fffffffffffffffL)
  assertInlineFill[Long](FillN, 1)
  assertInlineFill[Long](FillN, -1)
  assertInlineFill[Long](FillN, 0)
  assertInlineFill[Long](FillN, 42)

  assertInlineFill[Float](FillN, 1.4e-45f)
  assertInlineFill[Float](FillN, 1.17549435e-38f)
  assertInlineFill[Float](FillN, 3.4028235e+38) // XXX 3.4028235e+38f appears to not fit!
  assertInlineFill[Float](FillN, 0.0f / 0.0f)
  assertInlineFill[Float](FillN, 1)
  assertInlineFill[Float](FillN, -1)
  assertInlineFill[Float](FillN, 0)
  assertInlineFill[Float](FillN, 42)
  assertInlineFill[Float](FillN, 3.14159265358979323846)

  assertInlineFill[Double](FillN, 4.9e-324d)
  assertInlineFill[Double](FillN, 2.2250738585072014e-308d)
  assertInlineFill[Double](FillN, 1.7976931348623157e+308d)
  assertInlineFill[Double](FillN, 0.0f / 0.0d)
  assertInlineFill[Double](FillN, 1)
  assertInlineFill[Double](FillN, -1)
  assertInlineFill[Double](FillN, 0)
  assertInlineFill[Double](FillN, 42)
  assertInlineFill[Double](FillN, 3.14159265358979323846d)

  assertInlineFill[Boolean](FillN, true)
  assertInlineFill[Boolean](FillN, false)

}