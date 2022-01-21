package polyregion

trait BaseSuite extends munit.FunSuite {

  import scala.compiletime._

  def assertValEquals[A](actual: A, expected: A): Unit = (actual.asMatchable, expected.asMatchable) match {
    case (a: Float, e: Float) => //
      assertEquals(java.lang.Float.floatToIntBits(a), java.lang.Float.floatToIntBits(e))
    case (a: Double, e: Double) => //
      assertEquals(java.lang.Double.doubleToLongBits(a), java.lang.Double.doubleToLongBits(e))
    case (a, e) => //
      assertEquals(a, e)
  }

  inline def unrollInclusive[U](inline n: Int)(inline f: Int => Unit): Unit = {
    f(n)
    if (n > 0) unrollInclusive(n - 1)(f)
  }

  inline def assertOffload[A](inline f: => A) =
    assertValEquals(polyregion.compileTime.offload[A](f), f)

}
