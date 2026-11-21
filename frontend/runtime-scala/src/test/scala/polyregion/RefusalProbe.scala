package polyregion

import polyregion.jvm.runtime as rt

import java.nio.file.{Files, Path}
import java.nio.{ByteBuffer, ByteOrder}

object RefusalProbe {

  def main(args: Array[String]): Unit = {
    rt.Context.loadHostProgram("refuse@host", Files.readAllBytes(Path.of(sys.env("REFUSAL_REFUSE"))))
    rt.Context.loadHostProgram("usable@host", Files.readAllBytes(Path.of(sys.env("REFUSAL_USABLE"))))
    val device = rt.Context.platforms().flatMap(_.devices).find(_.platformKind == 2).get
    val ctx    = rt.Context.create(device)
    try {
      val data = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder).putLong(0x1000L).array
      try {
        ctx.callHost(Array.empty, "refuse@host", "refuse", Array(rt.Type.PTR.value), data)
        println("no exception")
      } catch {
        case e: rt.PolyregionRuntimeException => println(s"raised: ${e.getMessage}")
      }
      val before = ctx.stats().remoteMallocs
      ctx.callHost(Array.empty, "usable@host", "usable", Array.empty, Array.empty)
      println(s"usable: remoteMallocs=${ctx.stats().remoteMallocs - before}")
    } finally ctx.close()
  }
}
