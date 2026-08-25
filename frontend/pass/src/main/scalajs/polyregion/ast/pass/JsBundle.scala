package polyregion.ast.pass

import polyregion.ast.{MsgPack, PackageLinker, PackageSymResolver}
import polyregion.ast.PolyAST.{PolyPackageAbi, PolyPassAbi}
import polyregion.ast.PolyAST as p
import polyregion.ast.generated.PolyPackageWireSchema

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

  private def packageOperation[A: MsgPack.Codec, B: MsgPack.Codec](
      label: String,
      bytes: Uint8Array
  )(operation: A => Either[List[String], B]): Uint8Array =
    MsgPack.decode[MsgPack.Versioned[A]](JsBytes.toArray(bytes)) match {
      case Left(error) =>
        throw js.JavaScriptException(s"PolyPackage $label: cannot decode request: ${error.getMessage}")
      case Right(envelope) if envelope.hash != PolyPackageWireSchema.Hash =>
        throw js.JavaScriptException(
          s"PolyPackage $label: package-service wire hash differs: expected ${PolyPackageWireSchema.Hash}, got ${envelope.hash}"
        )
      case Right(envelope) =>
        operation(envelope.t) match {
          case Left(errors) => throw js.JavaScriptException(s"PolyPackage $label: ${errors.mkString("\n")}")
          case Right(result) =>
            JsBytes.fromArray(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, result)))
        }
    }

  @JSExportTopLevel(PolyPackageAbi.AbiVersion.Name)
  def packageAbiVersion(): Int = PolyPackageAbi.Version

  @JSExportTopLevel(PolyPackageAbi.LinkPackage.Name)
  def linkPackage(bytes: Uint8Array): Uint8Array =
    packageOperation[p.Package.LinkRequest, p.Package]("link package", bytes)(PackageLinker.link)

  @JSExportTopLevel(PolyPackageAbi.ResolveSym.Name)
  def resolvePackageSym(bytes: Uint8Array): Uint8Array =
    packageOperation[p.Package.SymRequest, p.Package.SymResolvedProgram]("resolve Sym", bytes)(
      PackageSymResolver.resolveSym
    )

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
      val out = Uint8Array(bytes.length)
      var i   = 0
      while (i < bytes.length) {
        out(i) = (bytes(i).toInt & 0xff).toShort
        i += 1
      }
      out
    }
  }
}
