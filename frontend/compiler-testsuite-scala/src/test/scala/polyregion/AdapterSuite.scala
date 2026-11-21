package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.Await
import scala.concurrent.duration.*

class AdapterSuite extends munit.FunSuite {

  private def closedContext: rt.Context = {
    val host = rt.Context.platforms().flatMap(_.devices).find(d => d.platformKind == 1 && d.moduleFormat == 2).get
    val ctx  = rt.Context.create(host)
    ctx.close()
    ctx
  }

  test("the callback core completes once with the value") {
    val latch   = CountDownLatch(1)
    var results = List.empty[Either[Throwable, Int]]
    callback.Host.ops.task[Int](42) { r =>
      results ::= r; latch.countDown()
    }
    assert(latch.await(30, TimeUnit.SECONDS))
    assertEquals(results.map(_.toOption), List(Some(42)))
  }

  test("the callback core completes once with the failure") {
    val latch   = CountDownLatch(1)
    var results = List.empty[Either[Throwable, Int]]
    callback.Host.on(closedContext).ops.task[Int](42) { r =>
      results ::= r; latch.countDown()
    }
    assert(latch.await(30, TimeUnit.SECONDS))
    assertEquals(results.size, 1)
    assert(results.head.left.exists(_.getMessage.endsWith(" is closed")), results.toString)
  }

  test("the blocking adapter returns the value and throws the failure") {
    assertEquals(blocking.Host.ops.task[Int](42), 42)
    val e = intercept[rt.PolyregionRuntimeException](blocking.Host.on(closedContext).ops.task[Int](42))
    assert(e.getMessage.endsWith(" is closed"), e.getMessage)
  }

  test("the future adapter completes with the value and fails with the failure") {
    assertEquals(Await.result(future.Host.ops.task[Int](42), 30.seconds), 42)
    val failed = future.Host.on(closedContext).ops.task[Int](42)
    val e      = intercept[rt.PolyregionRuntimeException](Await.result(failed, 30.seconds))
    assert(e.getMessage.endsWith(" is closed"), e.getMessage)
  }
}
