package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*
import polyregion.scalalang.blocking.Host

import scala.compiletime.testing.typeCheckErrors

trait SpectraLaunchBase extends munit.FunSuite {

  protected def device(accept: rt.DeviceInfo => Boolean): Option[rt.DeviceInfo] =
    rt.Context.platforms().flatMap(_.devices).find(accept)

  protected def withContext[A](device: Option[rt.DeviceInfo])(f: rt.Context => A): A = {
    assume(device.isDefined, "no matching device")
    val ctx = rt.Context.create(device.get)
    try f(ctx)
    finally ctx.close()
  }

  protected def host[A](f: rt.Context => A): A = withContext(device(d => d.platformKind == 1 && d.moduleFormat == 2))(f)

  protected def discreteDefault[A](f: rt.Context => A): A = {
    val ctx = rt.Context.current()
    assume(!ctx.sharedAddressSpace(), "the default context is not a discrete device")
    f(ctx)
  }

  protected def sortCase(ctx: rt.Context): Unit = {
    val xs = Array(5, 3, 9, 1, 7, 2, 8, 4, 6, 0)
    spectra.sort(ctx, xs, xs.length, (a: Int, b: Int) => a < b)
    assertEquals(xs.toList, (0 to 9).toList)
  }

  protected def transformCase(ctx: rt.Context): Unit = {
    val xs  = Array.tabulate(1000)(identity)
    val out = Array.ofDim[Int](xs.length)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x * 2 + 1)
    assertEquals(out.toList, xs.map(_ * 2 + 1).toList)
  }

  protected def reduceCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(1000)(identity)
    assertEquals(spectra.reduce(ctx, xs, xs.length, 0, (a: Int, b: Int) => a + b), xs.sum)
  }

  protected def captureCase(ctx: rt.Context): Unit = {
    val xs     = Array.tabulate(1000)(identity)
    val out    = Array.ofDim[Int](xs.length)
    val scale  = 3
    val offset = Array(7)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x * scale + offset(0))
    assertEquals(out.toList, xs.map(_ * 3 + 7).toList)
  }

  protected def sortDescendingCase(ctx: rt.Context): Unit = {
    val xs   = Array(5, 3, 9, 1, 7, 2, 8, 4, 6, 0)
    val sign = -1
    spectra.sort(ctx, xs, xs.length, (a: Int, b: Int) => a * sign < b * sign)
    assertEquals(xs.toList, (0 to 9).reverse.toList)
  }

  protected def sortByKeyArrayCase(ctx: rt.Context): Unit = {
    val keys  = Array(4, 8, 1, 9, 3, 0, 7, 2, 6, 5)
    val order = Array.tabulate(keys.length)(identity)
    spectra.sort(ctx, order, order.length, (a: Int, b: Int) => keys(a) < keys(b))
    assertEquals(order.toList, keys.indices.sortBy(keys).toList)
  }

  protected def pointerCaptureCase(ctx: rt.Context): Unit = {
    val xs     = Array.tabulate(1000)(identity)
    val out    = Array.ofDim[Int](xs.length)
    val offset = Array(7)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x + offset(0))
    assertEquals(out.toList, xs.map(_ + 7).toList)
  }

  protected def nestedPointerCase(ctx: rt.Context): Unit = {
    val xs    = Array.tabulate(1000)(identity)
    val out   = Array.ofDim[Int](xs.length)
    val scale = 3
    val low   = Array(5, 6)
    val high  = Array(100, 200)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x * scale + low(1) + high(0))
    assertEquals(out.toList, xs.map(_ * 3 + 6 + 100).toList)
  }

  private def skippingGaps(f: rt.Context => Unit): rt.Context => Unit = ctx =>
    try f(ctx)
    catch { case gap: rt.CapabilityGapException => assume(false, gap.getMessage) }

  private def launched(f: rt.Context => Unit): rt.Context => Unit = skippingGaps { ctx =>
    val before = ctx.stats().launches
    f(ctx)
    assert(ctx.stats().launches > before, "the call did not launch on the device")
  }

  protected def mirrorsOrRefuses(device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit =
    withContext(device)(ctx =>
      scala.util.Try(f(ctx)).failed.foreach {
        case gap: rt.CapabilityGapException => assume(false, gap.getMessage)
        case refused: rt.PolyregionRuntimeException =>
          assert(refused.getMessage.contains("created context"), refused.getMessage)
        case other => throw other
      }
    )

  private def launchedOnDefault(f: rt.Context => Unit): Unit = discreteDefault(launched(f))

  private def pointerBearingOnDefault(device: Option[rt.DeviceInfo])(f: rt.Context => Unit): Unit = {
    assume(device.isDefined, "no matching device")
    launchedOnDefault(f)
  }

  protected lazy val selectedBackend: Option[String] =
    sys.env.get("POLYRT_PLATFORM").map(_.takeWhile(c => c != '@' && c != '!')).filter(_.nonEmpty)

  protected def selectedTests(): Unit = {
    def onSelected(f: rt.Context => Unit): Unit = {
      assume(selectedBackend.exists(_ != "host"), "no device row is selected")
      launched(f)(rt.Context.current())
    }
    test("spectra.sort sorts on the selected device")(onSelected(sortCase(_)))
    test("spectra.transform maps on the selected device")(onSelected(transformCase(_)))
    test("spectra.reduce reduces on the selected device")(onSelected(reduceCase(_)))
    test("spectra.sort by a captured key array on the selected device")(onSelected(sortByKeyArrayCase(_)))
  }

  protected def deviceTests(kind: String, created: => Option[rt.DeviceInfo]): Unit = {
    test(s"a pointer-bearing capture on the default $kind context")(
      pointerBearingOnDefault(created)(pointerCaptureCase(_))
    )
    test(s"a nested pointer in the environment on the default $kind context")(
      pointerBearingOnDefault(created)(nestedPointerCase(_))
    )
    test(s"a pointer-bearing capture mirrors or is refused on a created $kind context")(
      mirrorsOrRefuses(created)(pointerCaptureCase(_))
    )
    test(s"a nested pointer in the environment mirrors or is refused on a created $kind context")(
      mirrorsOrRefuses(created)(nestedPointerCase(_))
    )
    test(s"spectra.transform captures a scalar and an array on the default $kind context")(
      pointerBearingOnDefault(created)(captureCase(_))
    )
    test(s"spectra.sort orders by a capturing comparator on $kind") {
      assume(created.isDefined, "no matching device")
      launchedOnDefault(sortDescendingCase(_))
    }
    test(s"spectra.sort sorts on $kind")(withContext(created)(launched(sortCase(_))))
    test(s"spectra.transform maps on $kind")(withContext(created)(launched(transformCase(_))))
    test(s"spectra.reduce reduces on $kind")(withContext(created)(launched(reduceCase(_))))
  }

  test("a pointer-bearing capture reads the host array in place on the host")(host(pointerCaptureCase(_)))
  test("a nested pointer in the environment reads the host arrays in place on the host")(host(nestedPointerCase(_)))

  test("spectra.sort sorts on the host")(host(sortCase(_)))
  test("spectra.transform captures a scalar and an array on the host")(host(captureCase(_)))
  test("spectra.sort orders by a capturing comparator on the host")(host(sortDescendingCase(_)))
  test("spectra.sort by a captured key array on the host")(host(sortByKeyArrayCase(_)))
  test("spectra.transform maps on the host")(host(transformCase(_)))
  test("spectra.reduce reduces on the host")(host(reduceCase(_)))

  test("a struct-array launch argument is rejected at expansion") {
    val errors = typeCheckErrors("""
      val points = Array(LaunchPoint(1, 2), LaunchPoint(3, 4))
      spectra.sort(Host.context, points, points.length, (a: LaunchPoint, b: LaunchPoint) => a.x < b.x)
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("launch argument of type scala.Array[polyregion.LaunchPoint] is not supported"), messages)
  }

  test("a spectra call inside an offload region is rejected naming both functions") {
    val errors = typeCheckErrors("""
      val xs = Array(1, 2, 3, 4)
      val ys = new Array[Int](4)
      compiletime.offload0(Host.context, _ => ()) {
        spectra.copy(xs, 4, ys)
        ()
      }
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("offload function"), messages)
    assert(messages.contains("calls host function `polyregion.spectra$.copy`"), messages)
  }

  test("a spectra call with a side-effecting argument inside an offload region is still rejected") {
    val errors = typeCheckErrors("""
      val xs = Array(1, 2, 3, 4)
      val ys = new Array[Int](4)
      var calls = 0
      compiletime.offload0(Host.context, _ => ()) {
        spectra.copy(xs, { calls += 1; 4 }, ys)
        ()
      }
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("calls host function `polyregion.spectra$.copy`"), messages)
  }
}
