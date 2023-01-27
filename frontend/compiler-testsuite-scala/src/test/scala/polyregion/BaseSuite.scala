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

  inline def offload0(using inline target: AssertTarget)(inline f: => Any): Unit =   target match {
    case AssertTarget.JDK => f
    case AssertTarget.Offload =>
      import polyregion.scalalang.*
      import polyregion.scalalang.blocking.Host
      val latch = new CountDownLatch(1)
      polyregion.scalalang.compiletime.offload0[Config[Target.Host.type, Opt.O2]](
        Host.underlying.createQueue(),
        {
          case Left(e)   => throw e
          case Right(()) => latch.countDown()
        }
      ) { f; () }
      latch.await(5, TimeUnit.SECONDS)
  }

  inline def offload1[A](using inline target: AssertTarget)(inline f: => A): A =   target match {
    case AssertTarget.JDK => f
    case AssertTarget.Offload =>
      import polyregion.scalalang.*
      import polyregion.scalalang.blocking.Host
      val r     = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
      val latch = new CountDownLatch(1)
      polyregion.scalalang.compiletime.offload0[Config[Target.Host.type, Opt.O2]](
        Host.underlying.createQueue(),
        {
          case Left(e)   => throw e
          case Right(()) => latch.countDown()
        }
      ) { r(0) = f; () }
      latch.await(5, TimeUnit.SECONDS)
      r(0)
  }

  inline def assertOffloadValue[A](inline f: AssertTarget ?=> A): Unit = {
    val expected =
      try f(using AssertTarget.JDK)
      catch {
        case e: Throwable => fail(s"offload reference expression failed to execute", e)
      }
    assertEquals(expected, expected, "offload reference values are not self-equal")
    if (!Toggles.NoOffload) {
      val actual = f(using AssertTarget.Offload)
      assertValEquals(actual, expected)
    }
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

}
