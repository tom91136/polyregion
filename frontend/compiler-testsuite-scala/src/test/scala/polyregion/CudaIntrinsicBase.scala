package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.compiletime.*
import scala.reflect.ClassTag

trait CudaIntrinsicBase extends BaseSuite {

  protected lazy val hasCuda = CudaProbe.hasDevice

  protected inline def onCuda[A](inline f: => A): A = {
    val ctx   = polyregion.scalalang.blocking.CUDA.devices.head.context
    val r     = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    val latch = CountDownLatch(1)
    polyregion.scalalang.compiletime.offload0(
      ctx,
      {
        case Left(e)   => throw e
        case Right(()) => latch.countDown()
      }
    ) { r(0) = f; () }
    assert(latch.await(60, TimeUnit.SECONDS))
    r(0)
  }

  protected inline def offloadOutcome[A](inline f: => A): Either[Throwable, Unit] = {
    val ctx                                      = polyregion.scalalang.blocking.CUDA.devices.head.context
    val sink                                     = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    var outcome: Option[Either[Throwable, Unit]] = None
    val latch                                    = CountDownLatch(1)
    polyregion.scalalang.compiletime.offload0(
      ctx,
      { r =>
        outcome = Some(r)
        latch.countDown()
      }
    ) { sink(0) = f; () }
    assert(latch.await(60, TimeUnit.SECONDS))
    outcome.get
  }

  protected inline def testFails[A](name: String, message: String = "by zero")(inline f: => A) = if (
    Toggles.CudaIntrinsicSuite
  ) {
    test(name) {
      assume(hasCuda, "no CUDA device")
      intercept[ArithmeticException](f)
      offloadOutcome[A](f) match {
        case Left(e)  => assert(e.getMessage.contains(message), e.getMessage)
        case Right(_) => fail(s"offload did not report $message")
      }
    }
  }

  protected inline def testExpr[A <: AnyVal](inline r: A)(using C: ClassTag[A]) = if (Toggles.CudaIntrinsicSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}") {
      assume(hasCuda, "no CUDA device")
      val actual   = onCuda(r)
      val expected = r
      assertValEquals(actual, expected)
    }
  }

}
