package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration.*

object RecursionSuite {
  def down(n: Int): Int = if (n <= 0) 0 else 1 + down(n - 1)
}

class RecursionSuite extends munit.FunSuite {

  private lazy val hasCuda =
    try blocking.CUDA.devices.nonEmpty
    catch { case _: Throwable => false }

  private inline def downFrom(inline depth: Int): Int = {
    def down(n: Int): Int = if (n <= 0) 0 else 1 + down(n - 1)
    down(depth)
  }

  private inline def onCuda(inline depth: Int): Either[Throwable, Int] = {
    val ctx      = blocking.CUDA.devices.head.context
    val result   = mutable.ListBuffer(-1)
    val outcomes = LinkedBlockingQueue[Either[Throwable, Unit]]()
    compiletime.offload0(ctx, r => outcomes.put(r), "cuda@sm_89") {
      result(0) = RecursionSuite.down(depth); ()
    }
    Option(outcomes.poll(60, TimeUnit.SECONDS)).get.map(_ => result(0))
  }

  private def assertRecursionLimit(e: Throwable): Unit = e match {
    case raised: rt.PolyregionRuntimeException => assert(raised.getMessage.contains("RLIM"), raised.getMessage)
    case other                                 => fail(s"expected a PolyregionRuntimeException, got $other")
  }

  test("the callback core returns the value of a recursion within the limit") {
    val results = LinkedBlockingQueue[Either[Throwable, Int]]()
    callback.Host.ops.task[Int](downFrom(100))(r => results.put(r))
    assertEquals(Option(results.poll(30, TimeUnit.SECONDS)).get, Right(100))
    assert(results.isEmpty)
  }

  test("the callback core fails with the recursion limit") {
    val results = LinkedBlockingQueue[Either[Throwable, Int]]()
    callback.Host.ops.task[Int](downFrom(100000))(r => results.put(r))
    val outcome = Option(results.poll(30, TimeUnit.SECONDS)).get
    assert(results.isEmpty)
    assert(outcome.isLeft, outcome.toString)
    outcome.left.foreach(e => assertRecursionLimit(e))
  }

  test("the blocking adapter returns the value and throws the recursion limit") {
    assertEquals(blocking.Host.ops.task[Int](downFrom(100)), 100)
    assertRecursionLimit(intercept[rt.PolyregionRuntimeException](blocking.Host.ops.task[Int](downFrom(100000))))
  }

  test("the future adapter completes with the value and fails with the recursion limit") {
    assertEquals(Await.result(future.Host.ops.task[Int](downFrom(100)), 30.seconds), 100)
    val failed = future.Host.ops.task[Int](downFrom(100000))
    assertRecursionLimit(intercept[rt.PolyregionRuntimeException](Await.result(failed, 30.seconds)))
  }

  test("a context runs a recursion again after the limit was raised") {
    intercept[rt.PolyregionRuntimeException](blocking.Host.ops.task[Int](downFrom(100000)))
    assertEquals(blocking.Host.ops.task[Int](downFrom(1000)), 1000)
  }

  test("a recursion within the limit returns the value on cuda") {
    assume(hasCuda, "no CUDA device")
    assertEquals(onCuda(100), Right(100))
  }

  test("a recursion past the limit fails on cuda") {
    assume(hasCuda, "no CUDA device")
    val outcome = onCuda(100000)
    assert(outcome.isLeft, outcome.toString)
    outcome.left.foreach(e => assertRecursionLimit(e))
  }
}
