package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

class OffloadFailureSuite extends munit.FunSuite {

  private def closedHostContext: rt.Context = {
    val host = rt.Context.platforms().flatMap(_.devices).find(d => d.platformKind == 1 && d.moduleFormat == 2).get
    val ctx  = rt.Context.create(host)
    ctx.close()
    ctx
  }

  test("a launch failure reaches the callback once") {
    var outcomes = List.empty[Either[Throwable, Unit]]
    var x        = 1
    compiletime.offload0[Config[Target.Host.type, Opt.O2]](closedHostContext, outcomes ::= _) { x += 1; () }
    outcomes match {
      case List(Left(e)) => assert(e.getMessage.endsWith(" is closed"), e.getMessage)
      case other         => fail(s"expected one failure, got $other")
    }
  }

  test("a callback that throws is not invoked again") {
    var calls = 0
    var x     = 1
    val host  = rt.Context.platforms().flatMap(_.devices).find(d => d.platformKind == 1 && d.moduleFormat == 2).get
    val ctx   = rt.Context.create(host)
    try
      intercept[IllegalStateException] {
        compiletime.offload0[Config[Target.Host.type, Opt.O2]](
          ctx,
          { _ =>
            calls += 1
            throw IllegalStateException("callback")
          }
        ) { x += 1; () }
      }
    finally ctx.close()
    assertEquals(calls, 1)
  }
}
