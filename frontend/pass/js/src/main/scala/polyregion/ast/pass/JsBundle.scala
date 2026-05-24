package polyregion.ast.pass

import polyregion.ast.PolyAST.PolyPassAbi

import scala.scalajs.js
import scala.scalajs.js.annotation.JSExportTopLevel
import scala.scalajs.js.typedarray.Uint8Array

object JsBundle {

  private val plugin: PluginEntry = DefaultPlugin

  @JSExportTopLevel(PolyPassAbi.AbiVersion.Name)
  def abiVersion(): Int = PolyPassAbi.Version

  @JSExportTopLevel(PolyPassAbi.PassCount.Name)
  def passCount(): Int = plugin.passNames.length

  @JSExportTopLevel(PolyPassAbi.PassName.Name)
  def passName(i: Int): String =
    if (i < 0 || i >= plugin.passNames.length) null else plugin.passNames(i)

  @JSExportTopLevel(PolyPassAbi.PassDescr.Name)
  def passDescr(i: Int): String =
    if (i < 0 || i >= plugin.passNames.length) null
    else plugin.passDescr(plugin.passNames(i)).orNull

  @JSExportTopLevel(PolyPassAbi.RunPasses.Name)
  def runPasses(steps: js.Array[String], bytes: Uint8Array): Uint8Array = {
    val inBytes  = JsBytes.toArray(bytes)
    val outBytes = plugin.runStepsMsgpack(steps.toVector, inBytes)
    JsBytes.fromArray(outBytes)
  }

  private object JsBytes {
    def toArray(bytes: Uint8Array): Array[Byte] = {
      val out = new Array[Byte](bytes.length)
      var i   = 0
      while (i < bytes.length) {
        out(i) = bytes(i).toByte
        i += 1
      }
      out
    }

    def fromArray(bytes: Array[Byte]): Uint8Array = {
      val out = new Uint8Array(bytes.length)
      var i   = 0
      while (i < bytes.length) {
        out(i) = (bytes(i).toInt & 0xff).toShort
        i += 1
      }
      out
    }
  }
}
