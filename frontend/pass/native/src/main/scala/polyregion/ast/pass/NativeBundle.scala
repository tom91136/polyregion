package polyregion.ast.pass

import polyregion.ast.PolyAST.PolyPassAbi

import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*
import scala.scalanative.libc.stdlib

object NativeBundle {

  private val plugin: PluginEntry = DefaultPlugin

  private var errorZone: Zone       = null
  private var errorCString: CString = null
  private val nameZone              = Zone.open()
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
}
