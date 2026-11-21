package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.Platforms

import java.nio.file.{Files, Path}
import java.nio.{ByteBuffer, ByteOrder}

object RefusalProbe {

  def main(args: Array[String]): Unit = {
    rt.Context.loadHostProgram("refuse@host", Files.readAllBytes(Path.of(sys.env("REFUSAL_REFUSE"))))
    rt.Context.loadHostProgram("usable@host", Files.readAllBytes(Path.of(sys.env("REFUSAL_USABLE"))))
    val device = rt.Context.platforms().flatMap(_.devices).find(_.platformKind == 2).get
    val ctx    = rt.Context.create(device)
    try {
      val host = ByteBuffer.allocateDirect(8)
      val data =
        ByteBuffer.allocate(8).order(ByteOrder.nativeOrder).putLong(Platforms.pointerOfDirectBuffer(host)).array
      val held = ctx.stats().remoteAllocations
      try {
        ctx.callHost(Array.empty, "refuse@host", "refuse", Array(rt.Type.PTR.value), data)
        println("no exception")
      } catch {
        case e: rt.PolyregionRuntimeException => println(s"raised: ${e.getMessage}")
      }
      (1 to 2).foreach { _ =>
        try ctx.callHost(Array.empty, "refuse@host", "refuse", Array(rt.Type.PTR.value), data)
        catch { case _: rt.PolyregionRuntimeException => }
      }
      println(s"owned: remoteAllocations=${ctx.stats().remoteAllocations - held}")
      val before = ctx.stats().remoteMallocs
      ctx.callHost(Array.empty, "usable@host", "usable", Array.empty, Array.empty)
      println(s"usable: remoteMallocs=${ctx.stats().remoteMallocs - before}")
    } finally ctx.close()
  }
}
