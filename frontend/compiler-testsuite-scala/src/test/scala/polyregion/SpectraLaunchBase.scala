package polyregion

import polyregion.SpectraRaw.*
import polyregion.ast.PolyAST as p
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

  protected def reduceOrderCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(100000)(_ + 1)
    assertEquals(spectra.reduce(ctx, xs, xs.length, 0, (_: Int, b: Int) => b), xs.last)
  }

  protected def transformReduceOrderCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(100000)(_ + 1)
    assertEquals(spectra.transform_reduce(ctx, xs, xs.length, 0, (x: Int) => x * 2, (_: Int, b: Int) => b), xs.last * 2)
  }

  protected def reduceUnorderedCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(100000)(_ + 1)
    assertEquals(spectra.reduce_unordered(ctx, xs, xs.length, 0, (a: Int, b: Int) => a + b), xs.sum)
  }

  protected def inclusiveScanCase(ctx: rt.Context): Unit = {
    val xs  = Array.tabulate(100000)(i => i % 7)
    val out = Array.ofDim[Int](xs.length)
    spectra.inclusive_scan(ctx, xs, out, xs.length, (a: Int, b: Int) => a + b)
    assertEquals(out.toList, xs.scanLeft(0)(_ + _).tail.toList)
  }

  protected def transformReduceUnorderedCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(100000)(_ + 1)
    assertEquals(
      spectra.transform_reduce_unordered(ctx, xs, xs.length, 0, (x: Int) => x % 7, (a: Int, b: Int) => a + b),
      xs.map(_ % 7).sum
    )
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

  protected def zeroDivisionCase(ctx: rt.Context): Unit = {
    val xs  = Array(1, 2, 0, 4)
    val out = Array.ofDim[Int](xs.length)
    val e   = intercept[ArithmeticException](spectra.transform(ctx, xs, out, xs.length, (x: Int) => 10 / x))
    assertEquals(e.getMessage, "/ by zero")
  }

  protected def zeroRemainderCase(ctx: rt.Context): Unit = {
    val xs  = Array(1, 2, 0, 4)
    val out = Array.ofDim[Int](xs.length)
    val e   = intercept[ArithmeticException](spectra.transform(ctx, xs, out, xs.length, (x: Int) => 10 % x))
    assertEquals(e.getMessage, "/ by zero")
  }

  protected def minDivisionCase(ctx: rt.Context): Unit = {
    val xs    = Array(Int.MinValue, 7, -7, Int.MinValue)
    val out   = Array.ofDim[Int](xs.length)
    val minus = -1
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x / minus)
    assertEquals(out.toList, xs.map(_ / minus).toList)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x % minus)
    assertEquals(out.toList, xs.map(_ % minus).toList)
  }

  private def zeroReadOp: p.Function = {
    val (x, xRef)         = select(i32, "x")
    val (fresh, freshRef) = select(ptrI32, "fresh")
    val (last, lastRef)   = select(i32, "last")
    val (v, vRef)         = select(i32, "v")
    callable(
      "rawZeroRead",
      x,
      i32,
      List(
        p.Stmt.Var(fresh, Some(p.Expr.Alloc(i32, xRef, g, p.Region.Opaque))),
        p.Stmt.Var(last, Some(p.Expr.IntrOp(p.Intr.Sub(xRef, p.Term.IntS32Const(1), i32)))),
        p.Stmt.Var(v, Some(p.Expr.Index(freshRef, lastRef, i32))),
        p.Stmt.Return(p.Expr.Alias(vRef))
      )
    )
  }

  private def escapingOp: p.Function = {
    val (x, xRef)       = select(i32, "x")
    val (arr, arrRef)   = select(ptrI32, "arr")
    val (addr, addrRef) = select(p.Type.IntS64, "addr")
    val (r, rRef)       = select(i32, "r")
    callable(
      "rawEscape",
      x,
      i32,
      List(
        p.Stmt.Var(arr, Some(p.Expr.Alloc(i32, xRef, g, p.Region.Opaque))),
        p.Stmt.Var(addr, Some(p.Expr.Cast(arrRef, p.Type.IntS64))),
        p.Stmt.Var(r, Some(p.Expr.Cast(addrRef, i32))),
        p.Stmt.Return(p.Expr.Alias(rRef))
      )
    )
  }

  protected def zeroedAllocationCase(ctx: rt.Context): Unit = {
    val out = Array(-1)
    dirtyHeap(64)
    rawLaunch(ctx, zeroReadOp, Array(16), out)
    assertEquals(out.toList, List(0))
  }

  protected def refusedEscapeCase(): Unit = {
    val e = intercept[Exception](compile(escapingOp, hostOnly = false))
    assert(e.getMessage.contains("may not let an allocation escape"), e.getMessage)
    assert(e.getMessage.contains("converted to an integer"), e.getMessage)
  }

  protected def jvmSemanticsTests(kind: String, created: => Option[rt.DeviceInfo]): Unit = {
    test(s"a division by zero in a spectra.transform lambda raises ArithmeticException on $kind")(
      withContext(created)(zeroDivisionCase(_))
    )
    test(s"a remainder by zero in a spectra.transform lambda raises ArithmeticException on $kind")(
      withContext(created)(zeroRemainderCase(_))
    )
    test(s"MIN / -1 and MIN % -1 in a spectra.transform lambda wrap on $kind")(withContext(created)(minDivisionCase(_)))
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
    test("spectra.reduce keeps encounter order on the selected device")(onSelected(reduceOrderCase(_)))
    test("spectra.transform_reduce keeps encounter order on the selected device")(
      onSelected(transformReduceOrderCase(_))
    )
    test("spectra.reduce_unordered reduces on the selected device")(onSelected(reduceUnorderedCase(_)))
    test("spectra.transform_reduce_unordered reduces on the selected device")(
      onSelected(transformReduceUnorderedCase(_))
    )
    test("spectra.sort by a captured key array on the selected device")(onSelected(sortByKeyArrayCase(_)))
  }

  protected def deviceTests(kind: String, created: => Option[rt.DeviceInfo]): Unit = {
    test(s"a pointer-bearing capture on the default $kind context")(
      pointerBearingOnDefault(created)(pointerCaptureCase(_))
    )
    test(s"a nested pointer in the environment on the default $kind context")(
      pointerBearingOnDefault(created)(nestedPointerCase(_))
    )
    test(s"a pointer-bearing capture mirrors on a created $kind context")(
      withContext(created)(launched(pointerCaptureCase(_)))
    )
    test(s"a nested pointer in the environment mirrors on a created $kind context")(
      withContext(created)(launched(nestedPointerCase(_)))
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
    test(s"spectra.reduce keeps encounter order on $kind")(withContext(created)(launched(reduceOrderCase(_))))
    test(s"spectra.transform_reduce keeps encounter order on $kind")(
      withContext(created)(launched(transformReduceOrderCase(_)))
    )
    test(s"spectra.inclusive_scan scans on $kind")(withContext(created)(launched(inclusiveScanCase(_))))
    test(s"spectra.reduce_unordered reduces on $kind")(withContext(created)(launched(reduceUnorderedCase(_))))
    test(s"spectra.transform_reduce_unordered reduces on $kind")(
      withContext(created)(launched(transformReduceUnorderedCase(_)))
    )
  }

  test("an allocation escaping a spectra.transform callable is refused")(refusedEscapeCase())
  test("a dynamically sized allocation in a spectra.transform callable is zero filled on the host")(
    host(zeroedAllocationCase(_))
  )

  jvmSemanticsTests("the host", device(d => d.platformKind == 1 && d.moduleFormat == 2))

  test("a pointer-bearing capture reads the host array in place on the host")(host(pointerCaptureCase(_)))
  test("a nested pointer in the environment reads the host arrays in place on the host")(host(nestedPointerCase(_)))

  test("spectra.sort sorts on the host")(host(sortCase(_)))
  test("spectra.transform captures a scalar and an array on the host")(host(captureCase(_)))
  test("spectra.sort orders by a capturing comparator on the host")(host(sortDescendingCase(_)))
  test("spectra.sort by a captured key array on the host")(host(sortByKeyArrayCase(_)))
  test("spectra.transform maps on the host")(host(transformCase(_)))
  test("spectra.reduce reduces on the host")(host(reduceCase(_)))
  test("spectra.reduce keeps encounter order on the host")(host(reduceOrderCase(_)))
  test("spectra.transform_reduce keeps encounter order on the host")(host(transformReduceOrderCase(_)))
  test("spectra.reduce_unordered reduces on the host")(host(reduceUnorderedCase(_)))
  test("spectra.transform_reduce_unordered reduces on the host")(host(transformReduceUnorderedCase(_)))

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
