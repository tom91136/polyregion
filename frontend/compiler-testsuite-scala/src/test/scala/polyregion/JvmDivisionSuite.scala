package polyregion

import polyregion.scalalang.*

import scala.concurrent.Await
import scala.concurrent.duration.*

class JvmDivisionSuite extends BaseSuite {

  private type Host = Config[Target.Host.type, Opt.O2]

  private inline def blockingValue[A](inline f: => A): A = blocking.Host.aot.task[Host, A](f)

  private inline def futureValue[A](inline f: => A): A = Await.result(future.Host.aot.task[Host, A](f), 30.seconds)

  private inline def blockingRaises[A](inline f: => A): Unit =
    assertEquals(intercept[ArithmeticException](blockingValue[A](f)).getMessage, "/ by zero")

  private inline def futureRaises[A](inline f: => A): Unit =
    assertEquals(intercept[ArithmeticException](futureValue[A](f)).getMessage, "/ by zero")

  test("signed division and remainder by -1 wrap like the JVM through the blocking adapter") {
    val iMin   = Int.MinValue
    val lMin   = Long.MinValue
    val minus  = -1
    val minusL = -1L
    assertEquals(blockingValue[Int](iMin / minus), iMin / minus)
    assertEquals(blockingValue[Int](iMin % minus), iMin % minus)
    assertEquals(blockingValue[Long](lMin / minusL), lMin / minusL)
    assertEquals(blockingValue[Long](lMin % minusL), lMin % minusL)
    assertEquals(blockingValue[Int](7 / minus), -7)
    assertEquals(blockingValue[Int](7 % minus), 0)
  }

  test("signed division and remainder by -1 wrap like the JVM through the future adapter") {
    val iMin   = Int.MinValue
    val lMin   = Long.MinValue
    val minus  = -1
    val minusL = -1L
    assertEquals(futureValue[Int](iMin / minus), iMin / minus)
    assertEquals(futureValue[Int](iMin % minus), iMin % minus)
    assertEquals(futureValue[Long](lMin / minusL), lMin / minusL)
    assertEquals(futureValue[Long](lMin % minusL), lMin % minusL)
  }

  test("signed division and remainder by zero raise ArithmeticException through the blocking adapter") {
    val iMin  = Int.MinValue
    val lMin  = Long.MinValue
    val zero  = 0
    val zeroL = 0L
    blockingRaises[Int](iMin / zero)
    blockingRaises[Int](iMin % zero)
    blockingRaises[Int](7 / zero)
    blockingRaises[Long](lMin / zeroL)
    blockingRaises[Long](lMin % zeroL)
    blockingRaises[Long](7L   % zeroL)
  }

  test("signed division and remainder by zero fail the future adapter with ArithmeticException") {
    val iMin  = Int.MinValue
    val lMin  = Long.MinValue
    val zero  = 0
    val zeroL = 0L
    futureRaises[Int](iMin / zero)
    futureRaises[Int](iMin % zero)
    futureRaises[Long](lMin / zeroL)
    futureRaises[Long](lMin % zeroL)
  }

  test("an unused division or remainder by zero still raises ArithmeticException through the blocking adapter") {
    val zero  = 0
    val zeroL = 0L
    blockingRaises[Unit] { 7 / zero; () }
    blockingRaises[Unit] { 7 % zero; () }
    blockingRaises[Unit] { 7L / zeroL; () }
    blockingRaises[Unit] { 7L % zeroL; () }
  }

  test("an unused division or remainder by zero fails the future adapter with ArithmeticException") {
    val zero  = 0
    val zeroL = 0L
    futureRaises[Unit] { 7 / zero; () }
    futureRaises[Unit] { 7 % zero; () }
    futureRaises[Unit] { 7L / zeroL; () }
    futureRaises[Unit] { 7L % zeroL; () }
  }

  test("a kernel keeps running after another launch raised ArithmeticException") {
    val zero = 0
    blockingRaises[Int](7 / zero)
    assertEquals(blockingValue[Int](zero + 7), 7)
  }

}
