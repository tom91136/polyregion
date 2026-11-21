package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*

import scala.collection.mutable

trait RangeSuiteBase extends munit.FunSuite {

  private val devices = rt.Context.platforms().flatMap(_.devices)
  private val host    = devices.find(d => d.platformKind == 1 && d.moduleFormat == 2)
  private val nvidia  = devices.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0)

  private def withContext(label: String, device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit = device match {
    case None => assume(false, s"no $label device in the catalogue")
    case Some(d) =>
      val ctx = rt.Context.create(d)
      try f(ctx)
      finally ctx.close()
  }

  protected inline def run[C, A](ctx: rt.Context)(inline f: => A): A = {
    val holder                           = mutable.ListBuffer[A](null.asInstanceOf[A])
    var outcome: Either[Throwable, Unit] = null
    compiletime.offload0[C](ctx, outcome = _) { holder(0) = f; () }
    outcome.fold(throw _, _ => holder(0))
  }

  protected inline def same[A](name: String)(inline f: => A): Unit = {
    test(s"[host] $name")(
      withContext("host", host)(ctx => assertEquals(run[Config[Target.Host.type, Opt.O2], A](ctx)(f), f))
    )
    test(s"[nvidia] $name")(
      withContext("nvidia", nvidia)(ctx => assertEquals(run[Config[Target.NVPTX64["sm_80"], Opt.O2], A](ctx)(f), f))
    )
  }

  protected inline def sameOnHost[A](name: String)(inline f: => A): Unit =
    test(s"[host] $name")(
      withContext("host", host)(ctx => assertEquals(run[Config[Target.Host.type, Opt.O2], A](ctx)(f), f))
    )

  protected inline def raisesSame[A](name: String)(inline f: => A): Unit = {
    test(s"[host] $name") {
      withContext("host", host) { ctx =>
        val expected = intercept[Throwable](f)
        val actual   = intercept[Throwable](run[Config[Target.Host.type, Opt.O2], A](ctx)(f))
        assertEquals(actual.getClass, expected.getClass)
        assertEquals(actual.getMessage, expected.getMessage)
      }
    }
    test(s"[nvidia] $name") {
      withContext("nvidia", nvidia) { ctx =>
        val expected = intercept[Throwable](f)
        val actual   = intercept[Throwable](run[Config[Target.NVPTX64["sm_80"], Opt.O2], A](ctx)(f))
        assertEquals(actual.getClass, expected.getClass)
        assertEquals(actual.getMessage, expected.getMessage)
      }
    }
  }

  protected inline def raisesClass[A](name: String)(inline f: => A): Unit = {
    test(s"[host] $name") {
      withContext("host", host) { ctx =>
        val expected = intercept[Throwable](f)
        val actual   = intercept[Throwable](run[Config[Target.Host.type, Opt.O2], A](ctx)(f))
        assertEquals(actual.getClass, expected.getClass)
      }
    }
    test(s"[nvidia] $name") {
      withContext("nvidia", nvidia) { ctx =>
        val expected = intercept[Throwable](f)
        val actual   = intercept[Throwable](run[Config[Target.NVPTX64["sm_80"], Opt.O2], A](ctx)(f))
        assertEquals(actual.getClass, expected.getClass)
      }
    }
  }
}
