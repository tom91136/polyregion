package polyregion

import polyregion.scalalang.{NativeStruct, Platforms}

import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.compiletime.*
import scala.reflect.ClassTag

trait BaseSuite extends munit.FunSuite {

//  inline def doOffload[A](inline x: => A): A = if (Toggles.NoOffload) x
//  else {
//    import polyregion.scala.*
//    import polyregion.scala.blocking.*
//    try
////      CUDA.devices(0).aot.task[Config[Target.NVPTX64.SM52.type, Opt.O2], A](x)
//      Host.aot.task[Config[Target.Host.type, Opt.O2], A](x)
//    catch {
//      case e: AssertionError => throw e
//      case e: Error =>
//        throw new AssertionError(e)
//    }
//  }

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

  enum AssertTarget {
    case JDK, Offload
  }

  inline def offload0(using inline target: AssertTarget)(inline f: => Any): Unit = target match {
    case AssertTarget.JDK => f
    case AssertTarget.Offload =>
      import polyregion.scalalang.*
      import polyregion.scalalang.blocking.Host
      val latch = new CountDownLatch(1)
      polyregion.scalalang.compiletime.offload0[Config[Target.Host.type, Opt.O2]](
        Host.underlying.createQueue(10000L),
        {
          case Left(e)   => throw e
          case Right(()) => latch.countDown()
        }
      ) { f; () }
      latch.await(5, TimeUnit.SECONDS)
  }

  inline def offload1[A](using inline target: AssertTarget)(inline f: => A): A = target match {
    case AssertTarget.JDK => f
    case AssertTarget.Offload =>
      import polyregion.scalalang.*
      import polyregion.scalalang.blocking.Host
      val r     = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
      val latch = new CountDownLatch(1)
      polyregion.scalalang.compiletime.offload0[Config[Target.Host.type, Opt.O2]](
        Host.underlying.createQueue(10000L),
        {
          case Left(e)   => throw e
          case Right(()) => latch.countDown()
        }
      ) { r(0) = f; () }
      latch.await(5, TimeUnit.SECONDS)
      r(0)
  }

  inline def assertOffloadValue[A](inline f: AssertTarget ?=> A): Unit = {
    // Run Offload BEFORE JDK so the JDK reference run doesn't pre-mutate captured state that the
    // Offload kernel would otherwise see in its by-value capture snapshot.
    val actual =
      if (!Toggles.NoOffload) Some(f(using AssertTarget.Offload))
      else None
    val expected =
      try f(using AssertTarget.JDK)
      catch {
        case e: Throwable => fail(s"offload reference expression failed to execute", e)
      }
    // Use bit-level self-equality so NaN doesn't trip the precheck (NaN != NaN by IEEE-754).
    assertValEquals(expected, expected)
    actual.foreach(a => assertValEquals(a, expected))
  }

  inline def assertOffloadEffect(inline f: AssertTarget ?=> Unit): Unit = {
    try f(using AssertTarget.JDK)
    catch {
      case e: Throwable => throw new AssertionError(s"offload reference expression failed to execute", e)
    }
    if (!Toggles.NoOffload) {
      f(using AssertTarget.Offload)
    }
  }

  inline def assertValEquals[A](inline actual: A, inline expected: A): Unit =
    inline (actual.asMatchable, expected.asMatchable) match {
      case (a: Float, e: Float) => //
        assertEquals(java.lang.Float.floatToIntBits(a), java.lang.Float.floatToIntBits(e))
      case (a: Double, e: Double) => //
        assertEquals(java.lang.Double.doubleToLongBits(a), java.lang.Double.doubleToLongBits(e))
      case (a, e) => //
        assertEquals(a, e)
    }

  // Compare two floating-point values with a ULP tolerance — useful for intrinsics whose
  // host (libm) and reference (JDK StrictMath) implementations are correctly rounded but
  // differ by a handful of last-place units (e.g. `hypot`).
  def assertValEqualsUlps(actual: Double, expected: Double, maxUlps: Long): Unit = {
    val same =
      if (java.lang.Double.isNaN(actual) && java.lang.Double.isNaN(expected)) true
      else if (java.lang.Double.isInfinite(actual) || java.lang.Double.isInfinite(expected))
        actual == expected
      else {
        val ab = java.lang.Double.doubleToLongBits(actual)
        val eb = java.lang.Double.doubleToLongBits(expected)
        // Map sign-magnitude to ordered representation so subtraction yields a meaningful ULP gap.
        def order(b: Long): Long = if (b < 0L) 0x8000000000000000L - b else b
        Math.abs(order(ab) - order(eb)) <= maxUlps
      }
    if (!same)
      fail(s"Expected $expected within $maxUlps ULPs, got $actual (Δ=${actual - expected})")
  }

  def assertValEqualsUlps(actual: Float, expected: Float, maxUlps: Int): Unit = {
    val same =
      if (java.lang.Float.isNaN(actual) && java.lang.Float.isNaN(expected)) true
      else if (java.lang.Float.isInfinite(actual) || java.lang.Float.isInfinite(expected))
        actual == expected
      else {
        val ab                 = java.lang.Float.floatToIntBits(actual)
        val eb                 = java.lang.Float.floatToIntBits(expected)
        def order(b: Int): Int = if (b < 0) 0x80000000 - b else b
        Math.abs(order(ab) - order(eb)) <= maxUlps
      }
    if (!same)
      fail(s"Expected $expected within $maxUlps ULPs, got $actual (Δ=${actual - expected})")
  }

  /** Like `assertOffloadValue`, but allows the offload result to differ from the JDK reference by up to `maxUlps`
    * last-place units. Use for intrinsics where host libm and JDK StrictMath are correctly rounded but disagree on the
    * last bit.
    */
  inline def assertOffloadValueUlps[A](inline f: AssertTarget ?=> A, maxUlps: Int): Unit = {
    val actual =
      if (!Toggles.NoOffload) Some(f(using AssertTarget.Offload))
      else None
    val expected =
      try f(using AssertTarget.JDK)
      catch {
        case e: Throwable => fail(s"offload reference expression failed to execute", e)
      }
    actual.foreach { a =>
      inline (a.asMatchable, expected.asMatchable) match {
        case (av: Float, ev: Float)   => assertValEqualsUlps(av, ev, maxUlps)
        case (av: Double, ev: Double) => assertValEqualsUlps(av, ev, maxUlps.toLong)
        case (av, ev)                 => assertValEquals(av, ev)
      }
    }
  }

}
