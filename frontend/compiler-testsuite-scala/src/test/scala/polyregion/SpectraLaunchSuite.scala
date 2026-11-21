package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.*
import polyregion.scalalang.blocking.Host

import scala.compiletime.testing.typeCheckErrors

class SpectraLaunchSuite extends munit.FunSuite {

  private def device(accept: rt.DeviceInfo => Boolean): Option[rt.DeviceInfo] =
    rt.Context.platforms().flatMap(_.devices).find(accept)

  private def withContext[A](device: Option[rt.DeviceInfo])(f: rt.Context => A): A = {
    assume(device.isDefined, "no matching device")
    val ctx = rt.Context.create(device.get)
    try f(ctx)
    finally ctx.close()
  }

  private def host[A](f: rt.Context => A): A = withContext(device(d => d.platformKind == 1 && d.moduleFormat == 2))(f)
  private def cuda[A](f: rt.Context => A): A =
    withContext(device(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0))(f)

  private def cudaDefault[A](f: rt.Context => A): A = {
    val ctx = rt.Context.current()
    assume(!ctx.sharedAddressSpace(), "the default context is not a discrete device")
    f(ctx)
  }

  private def sortCase(ctx: rt.Context): Unit = {
    val xs = Array(5, 3, 9, 1, 7, 2, 8, 4, 6, 0)
    spectra.sort(ctx, xs, xs.length, (a: Int, b: Int) => a < b)
    assertEquals(xs.toList, (0 to 9).toList)
  }

  private def transformCase(ctx: rt.Context): Unit = {
    val xs  = Array.tabulate(1000)(identity)
    val out = new Array[Int](xs.length)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x * 2 + 1)
    assertEquals(out.toList, xs.map(_ * 2 + 1).toList)
  }

  private def reduceCase(ctx: rt.Context): Unit = {
    val xs = Array.tabulate(1000)(identity)
    assertEquals(spectra.reduce(ctx, xs, xs.length, 0, (a: Int, b: Int) => a + b), xs.sum)
  }

  private def captureCase(ctx: rt.Context): Unit = {
    val xs     = Array.tabulate(1000)(identity)
    val out    = new Array[Int](xs.length)
    val scale  = 3
    val offset = Array(7)
    spectra.transform(ctx, xs, out, xs.length, (x: Int) => x * scale + offset(0))
    assertEquals(out.toList, xs.map(_ * 3 + 7).toList)
  }

  private def sortDescendingCase(ctx: rt.Context): Unit = {
    val xs   = Array(5, 3, 9, 1, 7, 2, 8, 4, 6, 0)
    val sign = -1
    spectra.sort(ctx, xs, xs.length, (a: Int, b: Int) => a * sign < b * sign)
    assertEquals(xs.toList, (0 to 9).reverse.toList)
  }

  test("spectra.sort sorts on the host")(host(sortCase))
  test("spectra.transform captures a scalar and an array on the host")(host(captureCase))
  test("spectra.sort orders by a capturing comparator on the host")(host(sortDescendingCase))
  test("spectra.transform captures a scalar and an array on CUDA")(cudaDefault(captureCase))
  test("spectra.sort orders by a capturing comparator on CUDA")(cudaDefault(sortDescendingCase))
  test("spectra.transform maps on the host")(host(transformCase))
  test("spectra.reduce reduces on the host")(host(reduceCase))
  test("spectra.sort sorts on CUDA")(cuda(sortCase))
  test("spectra.transform maps on CUDA")(cuda(transformCase))
  test("spectra.reduce reduces on CUDA")(cuda(reduceCase))

  test("a spectra call inside an offload region is rejected naming both functions") {
    val errors = typeCheckErrors("""
      val xs = Array(1, 2, 3, 4)
      val ys = new Array[Int](4)
      compiletime.offload0(Host.context, _ => (), "host@native") {
        spectra.copy(xs, 4, ys)
        ()
      }
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("offload function"), messages)
    assert(messages.contains("calls host function `polyregion.spectra$.copy`"), messages)
  }
}
