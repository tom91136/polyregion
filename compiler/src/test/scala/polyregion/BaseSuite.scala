package polyregion

import scala.annotation.tailrec

trait BaseSuite extends munit.FunSuite {

  import scala.compiletime._

  inline def doOffload[A](inline x: => A): A = if (Toggles.NoOffload) x else polyregion.compiletime.offload[A](x)

  def assertValEquals[A](actual: A, expected: A): Unit = (actual.asMatchable, expected.asMatchable) match {
    case (a: Float, e: Float) => //
      assertEquals(java.lang.Float.floatToIntBits(a), java.lang.Float.floatToIntBits(e))
    case (a: Double, e: Double) => //
      assertEquals(java.lang.Double.doubleToLongBits(a), java.lang.Double.doubleToLongBits(e))
    case (a, e) => //
      assertEquals(a, e)
  }

  inline def unrollInclusive(inline n: Int)(inline f: Int => Unit): Unit = {
    if (n >= 0) f(n)
    if (n > 0)
      unrollInclusive(n - 1)(f)
  }

  inline def unrollGen[A](inline n: Int, inline xs: Seq[A])(inline f: A => Unit): Unit = {
    if (n >= 0) f(xs(n))
    if (n > 0)
      unrollGen(n - 1, xs)(f)
  }

  inline def foreachInlined[A](inline xs: Seq[A])(inline f: A => Unit): Unit = {
    val n = xs.length
    unrollGen[A](2, xs)(f)
  }

  inline def assertOffload[A](inline f: => A) = {
    val expected =
      try
        f
      catch {
        case e: Throwable => throw new AssertionError(s"offload reference expression ${codeOf(f)} failed to execute", e)
      }
    assertValEquals(doOffload[A](f), expected)
  }

  inline def Bytes = Array[Byte](-128, 127, 1, -1, 0, 42)

  inline def Chars = Array[Char]('\u0000', '\uFFFF', 1, 0, 42)

  inline def Shorts = Array[Short](-32768, 32767, 1, -1, 0, 42)

  inline def Ints = Array[Int](0x80000000, 0x7fffffff, 1, -1, 0, 42)

  inline def Longs = Array[Long](0x8000000000000000L, 0x7fffffffffffffffL, 1, -1, 0, 42)

  inline def Floats = Array[Float](
    1.4e-45f,
    1.17549435e-38f,
    3.4028235e+38, // XXX 3.4028235e+38f appears to not fit!
    0.0f / 0.0f,
    1,
    -1,
    0,
    42,
    3.14159265358979323846
  )

  inline def Doubles = Array[Double](
    4.9e-324d,
    2.2250738585072014e-308d,
    1.7976931348623157e+308d,
    0.0f / 0.0d,
    1,
    -1,
    0,
    42,
    3.14159265358979323846d
  )

  inline def Booleans = Array[Boolean](true, false)


}
