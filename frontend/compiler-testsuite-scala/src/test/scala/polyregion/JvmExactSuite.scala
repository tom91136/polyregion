package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.compiletime.*
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.reflect.ClassTag

class JvmExactSuite extends BaseSuite {

  private inline def testExpr[A <: AnyVal](inline r: A)(using C: ClassTag[A]) = if (Toggles.JvmExactSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  {
    val iMin = Int.MinValue
    val lMin = Long.MinValue
    val m1l  = -1L

    val iMax = Int.MaxValue
    val lMax = Long.MaxValue
    val one  = 1
    val oneL = 1L
    val big  = 0x7fffffffffL

    testExpr[Int](java.lang.Math.addExact(iMax, -1))
    testExpr[Int](java.lang.Math.addExact(iMin, iMax))
    testExpr[Long](java.lang.Math.addExact(lMax, -1L))
    testExpr[Long](java.lang.Math.addExact(lMin, lMax))
    testExpr[Int](java.lang.Math.subtractExact(iMin, -1))
    testExpr[Int](java.lang.Math.subtractExact(-1, iMax))
    testExpr[Long](java.lang.Math.subtractExact(lMin, -1L))
    testExpr[Long](java.lang.Math.subtractExact(-1L, lMax))
    testExpr[Int](java.lang.Math.multiplyExact(46340, 46340))
    testExpr[Int](java.lang.Math.multiplyExact(iMin, one))
    testExpr[Int](java.lang.Math.multiplyExact(iMax, -1))
    testExpr[Long](java.lang.Math.multiplyExact(3037000499L, 3037000499L))
    testExpr[Long](java.lang.Math.multiplyExact(lMin, oneL))
    testExpr[Long](java.lang.Math.multiplyExact(lMax, -1L))
    testExpr[Long](java.lang.Math.multiplyExact(-4611686018427387904L, 2L))
    testExpr[Long](java.lang.Math.multiplyExact(big, 2))
    testExpr[Long](java.lang.Math.multiplyExact(lMin, 0))
    testExpr[Int](java.lang.Math.incrementExact(iMax - 1))
    testExpr[Long](java.lang.Math.incrementExact(lMax - 1))
    testExpr[Int](java.lang.Math.decrementExact(iMin + 1))
    testExpr[Long](java.lang.Math.decrementExact(lMin + 1))
    testExpr[Int](java.lang.Math.negateExact(iMax))
    testExpr[Long](java.lang.Math.negateExact(lMax))
    testExpr[Int](java.lang.Math.negateExact(-7))
    testExpr[Int](java.lang.Math.toIntExact(iMax.toLong))
    testExpr[Int](java.lang.Math.toIntExact(Int.MinValue.toLong))
    testExpr[Int](java.lang.Math.toIntExact(-5L))
    testExpr[Int](java.lang.Math.absExact(-7))
    testExpr[Long](java.lang.Math.absExact(-7L))
    testExpr[Int](java.lang.StrictMath.addExact(3, 4))
    testExpr[Long](java.lang.StrictMath.multiplyExact(3L, -4L))
    testExpr[Int](java.lang.StrictMath.toIntExact(9L))
    testExpr[Long](java.lang.Math.multiplyHigh(lMax, lMax))
    testExpr[Long](java.lang.Math.multiplyHigh(lMin, lMin))
    testExpr[Long](java.lang.Math.multiplyHigh(lMin, lMax))
    testExpr[Long](java.lang.Math.multiplyHigh(m1l, m1l))
    testExpr[Long](java.lang.Math.multiplyHigh(m1l, 5L))
    testExpr[Long](java.lang.Math.multiplyHigh(0x123456789abcdefL, -0x0fedcba987654321L))
    testExpr[Long](java.lang.StrictMath.multiplyHigh(lMax, 3L))
    testExpr[Long](java.lang.Math.multiplyFull(iMin, iMin))
    testExpr[Long](java.lang.Math.multiplyFull(iMax, -3))

  }

  private inline def blockingRaises[A](message: String)(inline f: => A): Unit = {
    val raised = intercept[ArithmeticException](blocking.Host.ops.task[A](f))
    assertEquals(raised.getMessage, message)
  }

  private inline def futureRaises[A](message: String)(inline f: => A): Unit = {
    val failed = future.Host.ops.task[A](f)
    assertEquals(intercept[ArithmeticException](Await.result(failed, 30.seconds)).getMessage, message)
  }

  test("exact arithmetic overflow raises ArithmeticException through the blocking adapter") {
    val iMax  = Int.MaxValue
    val iMin  = Int.MinValue
    val lMax  = Long.MaxValue
    val lMin  = Long.MinValue
    val one   = 1
    val oneL  = 1L
    val m1    = -1
    val m1l   = -1L
    val two31 = 1L << 31
    val ovI   = "integer overflow"
    val ovL   = "long overflow"
    blockingRaises[Int](ovI)(java.lang.Math.addExact(iMax, one))
    blockingRaises[Int](ovI)(java.lang.Math.addExact(iMin, m1))
    blockingRaises[Long](ovL)(java.lang.Math.addExact(lMax, oneL))
    blockingRaises[Long](ovL)(java.lang.Math.addExact(lMin, m1l))
    blockingRaises[Int](ovI)(java.lang.Math.subtractExact(iMin, one))
    blockingRaises[Int](ovI)(java.lang.Math.subtractExact(iMax, m1))
    blockingRaises[Long](ovL)(java.lang.Math.subtractExact(lMin, oneL))
    blockingRaises[Long](ovL)(java.lang.Math.subtractExact(lMax, m1l))
    blockingRaises[Int](ovI)(java.lang.Math.multiplyExact(iMin, m1))
    blockingRaises[Int](ovI)(java.lang.Math.multiplyExact(65536, 32768))
    blockingRaises[Long](ovL)(java.lang.Math.multiplyExact(lMin, m1l))
    blockingRaises[Long](ovL)(java.lang.Math.multiplyExact(lMax, 2L))
    blockingRaises[Long](ovL)(java.lang.Math.multiplyExact(lMin, m1))
    blockingRaises[Long](ovL)(java.lang.Math.multiplyExact(4294967296L, 2147483648L))
    blockingRaises[Int](ovI)(java.lang.Math.incrementExact(iMax))
    blockingRaises[Long](ovL)(java.lang.Math.incrementExact(lMax))
    blockingRaises[Int](ovI)(java.lang.Math.decrementExact(iMin))
    blockingRaises[Long](ovL)(java.lang.Math.decrementExact(lMin))
    blockingRaises[Int](ovI)(java.lang.Math.negateExact(iMin))
    blockingRaises[Long](ovL)(java.lang.Math.negateExact(lMin))
    blockingRaises[Int](ovI)(java.lang.Math.toIntExact(two31))
    blockingRaises[Int](ovI)(java.lang.Math.toIntExact(-two31 - 1))
    blockingRaises[Int](ovI)(java.lang.StrictMath.addExact(iMax, one))
    blockingRaises[Long](ovL)(java.lang.StrictMath.multiplyExact(lMax, 2L))
    blockingRaises[Unit](ovI) { java.lang.Math.addExact(iMax, one); () }
    blockingRaises[Int]("Overflow to represent absolute value of Integer.MIN_VALUE")(java.lang.Math.absExact(iMin))
    blockingRaises[Long]("Overflow to represent absolute value of Long.MIN_VALUE")(java.lang.Math.absExact(lMin))
  }

  test("exact arithmetic overflow fails the future adapter with ArithmeticException") {
    val iMax = Int.MaxValue
    val lMin = Long.MinValue
    val m1l  = -1L
    futureRaises[Int]("integer overflow")(java.lang.Math.addExact(iMax, 1))
    futureRaises[Long]("long overflow")(java.lang.Math.multiplyExact(lMin, m1l))
    futureRaises[Int]("integer overflow")(java.lang.Math.toIntExact(1L << 31))
  }

}
