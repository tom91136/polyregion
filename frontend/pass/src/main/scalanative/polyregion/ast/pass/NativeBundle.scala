package polyregion.ast.pass

import polyregion.ast.{MsgPack, PackageLinker, PackageSymResolver}
import polyregion.ast.PolyAST.{PolyPackageAbi, PolyPassAbi}
import polyregion.ast.PolyAST as p
import polyregion.ast.generated.PolyPackageWireSchema

import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*
import scala.scalanative.libc.stdlib

object NativeBundle {

  private val plugin: PluginEntry = DefaultPlugin

  private var errorZone: Zone              = null
  private var errorCString: CString        = null
  private var packageErrorZone: Zone       = null
  private var packageErrorCString: CString = null
  private val nameZone                     = Zone.open()
  private val nameStrings: Array[CString] = {
    val arr = new Array[CString](plugin.passNames.length)
    var i   = 0
    while (i < plugin.passNames.length) {
      arr(i) = toCString(plugin.passNames(i))(using nameZone)
      i += 1
    }
    arr
  }
  private val descrCache: scala.collection.mutable.HashMap[String, CString] = scala.collection.mutable.HashMap.empty

  private def setError(msg: String): Unit = {
    if (errorZone != null) errorZone.close()
    if (msg.isEmpty) { errorZone = null; errorCString = null }
    else {
      errorZone = Zone.open()
      errorCString = toCString(msg)(using errorZone)
    }
  }

  private def setPackageError(msg: String): Unit = {
    if (packageErrorZone != null) packageErrorZone.close()
    if (msg.isEmpty) { packageErrorZone = null; packageErrorCString = null }
    else {
      packageErrorZone = Zone.open()
      packageErrorCString = toCString(msg)(using packageErrorZone)
    }
  }

  @exported(PolyPassAbi.AbiVersion.Name)
  def abiVersion(): CUnsignedInt = PolyPassAbi.Version.toUInt

  @exported(PolyPassAbi.PassCount.Name)
  def passCount(): CSize = nameStrings.length.toCSize

  @exported(PolyPassAbi.PassName.Name)
  def passName(i: CSize): CString = {
    val idx = i.toInt
    if (idx < 0 || idx >= nameStrings.length) null else nameStrings(idx)
  }

  @exported(PolyPassAbi.PassDescr.Name)
  def passDescr(i: CSize): CString = {
    val idx = i.toInt
    if (idx < 0 || idx >= nameStrings.length) null
    else
      plugin.passDescr(plugin.passNames(idx)) match {
        case None    => null
        case Some(d) => descrCache.getOrElseUpdate(d, toCString(d)(using nameZone))
      }
  }

  @exported(PolyPassAbi.LastError.Name)
  def lastErrorPtr(): CString = errorCString

  @exported(PolyPassAbi.RunPasses.Name)
  def runPasses(
      steps: Ptr[CString],
      inPtr: Ptr[Byte],
      inLen: CSize,
      outPtr: Ptr[Ptr[Byte]],
      outLen: Ptr[CSize]
  ): CInt =
    try {
      val collected = scala.collection.mutable.ArrayBuffer.empty[String]
      var k         = 0
      while ({ val s = !(steps + k); s != null }) {
        collected += fromCString(!(steps + k))
        k += 1
      }

      val inLenInt = inLen.toInt
      val inBytes  = new Array[Byte](inLenInt)
      var i        = 0
      while (i < inLenInt) {
        inBytes(i) = !(inPtr + i)
        i += 1
      }

      val outBytes = plugin.runStepsMsgpack(collected.toVector, inBytes)

      val outBuf = stdlib.malloc(outBytes.length.toCSize).asInstanceOf[Ptr[Byte]]
      if (outBuf == null) {
        setError(s"PolyPass: malloc(${outBytes.length}) returned null")
        return PolyPassAbi.Status.AllocFailed
      }
      var j = 0
      while (j < outBytes.length) {
        !(outBuf + j) = outBytes(j)
        j += 1
      }
      !outPtr = outBuf
      !outLen = outBytes.length.toCSize
      setError("")
      PolyPassAbi.Status.Ok
    } catch {
      case t: Throwable =>
        val sw = java.io.StringWriter()
        t.printStackTrace(java.io.PrintWriter(sw))
        setError(s"PolyPass: ${t.getClass.getName}: ${Option(t.getMessage).getOrElse("<no message>")}\n${sw.toString}")
        PolyPassAbi.Status.PipelineError
    }

  @exported(PolyPassAbi.Free.Name)
  def freeBuffer(p: Ptr[Byte]): Unit =
    if (p != null) stdlib.free(p)

  private def readBytes(inPtr: Ptr[Byte], inLen: CSize): Array[Byte] = {
    val bytes = new Array[Byte](inLen.toInt)
    var i     = 0
    while (i < bytes.length) {
      bytes(i) = !(inPtr + i)
      i += 1
    }
    bytes
  }

  private def writeBytes(bytes: Array[Byte], outPtr: Ptr[Ptr[Byte]], outLen: Ptr[CSize]): CInt = {
    val buffer = stdlib.malloc(bytes.length.toCSize).asInstanceOf[Ptr[Byte]]
    if (buffer == null) {
      setPackageError(s"PolyPackage: malloc(${bytes.length}) returned null")
      PolyPackageAbi.Status.AllocFailed
    } else {
      var i = 0
      while (i < bytes.length) {
        !(buffer + i) = bytes(i)
        i += 1
      }
      !outPtr = buffer
      !outLen = bytes.length.toCSize
      setPackageError("")
      PolyPackageAbi.Status.Ok
    }
  }

  private def packageOperation[A: MsgPack.Codec, B: MsgPack.Codec](
      label: String,
      inPtr: Ptr[Byte],
      inLen: CSize,
      outPtr: Ptr[Ptr[Byte]],
      outLen: Ptr[CSize]
  )(operation: A => Either[List[String], B]): CInt =
    try
      MsgPack.decode[MsgPack.Versioned[A]](readBytes(inPtr, inLen)) match {
        case Left(error) =>
          setPackageError(s"PolyPackage $label: cannot decode request: ${error.getMessage}")
          PolyPackageAbi.Status.Invalid
        case Right(envelope) if envelope.hash != PolyPackageWireSchema.Hash =>
          setPackageError(
            s"PolyPackage $label: package-service wire hash differs: expected ${PolyPackageWireSchema.Hash}, got ${envelope.hash}"
          )
          PolyPackageAbi.Status.AbiMismatch
        case Right(envelope) =>
          val request = envelope.t
          operation(request) match {
            case Left(errors) =>
              setPackageError(s"PolyPackage $label: ${errors.mkString("\n")}")
              PolyPackageAbi.Status.Invalid
            case Right(result) =>
              writeBytes(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, result)), outPtr, outLen)
          }
      }
    catch {
      case error: Throwable =>
        val sw = java.io.StringWriter()
        error.printStackTrace(java.io.PrintWriter(sw))
        setPackageError(
          s"PolyPackage $label: ${error.getClass.getName}: ${Option(error.getMessage).getOrElse("<no message>")}\n${sw.toString}"
        )
        PolyPackageAbi.Status.Invalid
    }

  @exported(PolyPackageAbi.AbiVersion.Name)
  def packageAbiVersion(): CUnsignedInt = PolyPackageAbi.Version.toUInt

  @exported(PolyPackageAbi.LinkPackage.Name)
  def linkPackage(
      inPtr: Ptr[Byte],
      inLen: CSize,
      outPtr: Ptr[Ptr[Byte]],
      outLen: Ptr[CSize]
  ): CInt =
    packageOperation[p.Package.LinkRequest, p.Package]("link package", inPtr, inLen, outPtr, outLen)(
      PackageLinker.link
    )

  @exported(PolyPackageAbi.ResolveSym.Name)
  def resolvePackageSym(
      inPtr: Ptr[Byte],
      inLen: CSize,
      outPtr: Ptr[Ptr[Byte]],
      outLen: Ptr[CSize]
  ): CInt =
    packageOperation[p.Package.SymRequest, p.Package.SymResolvedProgram]("resolve Sym", inPtr, inLen, outPtr, outLen)(
      PackageSymResolver.resolveSym
    )

  @exported(PolyPackageAbi.LastError.Name)
  def packageLastErrorPtr(): CString = packageErrorCString

  @exported(PolyPackageAbi.Free.Name)
  def freePackageBuffer(p: Ptr[Byte]): Unit =
    if (p != null) stdlib.free(p)
}
