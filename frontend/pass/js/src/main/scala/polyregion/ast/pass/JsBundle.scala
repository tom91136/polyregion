package polyregion.ast.pass

import polyregion.ast.{MsgPack, PolyAST as p}
import polyregion.ast.Log

import scala.scalajs.js
import scala.scalajs.js.annotation.JSExportTopLevel
import scala.scalajs.js.typedarray.Uint8Array

object JsBundle {

  private object NoopLog extends Log {
    def info(message: String, details: String*): Unit = ()
    def subLog(name: String): Log                     = this
  }

  private val JsClock: PassClock = {
    val epochProbe = js.Dynamic.global.__polyregionEpochMillis
    val nanoProbe  = js.Dynamic.global.__polyregionNowNanos
    val epochFn: () => Long =
      if (js.typeOf(epochProbe) == "function") () => epochProbe().asInstanceOf[Double].toLong
      else () => js.Date.now().toLong
    val nanoFn: () => Long =
      if (js.typeOf(nanoProbe) == "function") () => nanoProbe().asInstanceOf[Double].toLong
      else () => (js.Date.now() * 1000000.0).toLong
    new PassClock {
      def epochMillis(): Long = epochFn()
      def nanoTime(): Long    = nanoFn()
    }
  }

  @JSExportTopLevel("runPipeline")
  def runPipeline(spec: String, bytes: Uint8Array): Uint8Array = {
    val rootEpoch = JsClock.epochMillis()
    val rootNanos = JsClock.nanoTime()
    val (prog, deserialiseEvent) = timed("polyast_msgpack_deserialise_js", s"bytes=${bytes.length}") {
      MsgPack.decodeInput[p.Program](JsBytes.input(bytes)).fold(throw _, identity)
    }
    val pipeline = PassPipelineParser.parse(spec).fold(e => throw new IllegalArgumentException(e), identity)
    val passes   = PassRegistry.build(pipeline).fold(e => throw new IllegalArgumentException(e), identity)
    val passRun  = new PassRunner(JsClock).runPipeline(passes, prog, NoopLog)

    val rootEvent =
      p.CompileEvent(
        rootEpoch,
        math.max(0L, JsClock.nanoTime() - rootNanos),
        "polypass",
        spec,
        deserialiseEvent :: passRun.event.items
      )
    encode(p.PassRunResult(passRun.program, rootEvent)).toUint8Array
  }

  private def timed[A](name: String, data: String = "")(f: => A): (A, p.CompileEvent) = {
    val startEpoch = JsClock.epochMillis()
    val startNanos = JsClock.nanoTime()
    val out        = f
    out -> p.CompileEvent(startEpoch, math.max(0L, JsClock.nanoTime() - startNanos), name, data, Nil)
  }

  private def encode(result: p.PassRunResult): JsBytes.Uint8Output = {
    val outBytes = JsBytes.output()
    MsgPack.encodeTo(result, outBytes)
    outBytes
  }

  private object JsBytes {
    def input(bytes: Uint8Array): MsgPack.ByteInput  = new Uint8Input(bytes)
    def output(initialSize: Int = 4096): Uint8Output = new Uint8Output(initialSize)

    private final class Uint8Input(bytes: Uint8Array) extends MsgPack.ByteInput {
      def length: Int = bytes.length

      def unsignedByteAt(index: Int): Int =
        bytes(index).toInt & 0xff

      override def copyToArray(srcPos: Int, dest: Array[Byte], destPos: Int, length: Int): Unit = {
        var i = 0
        while (i < length) {
          dest(destPos + i) = bytes(srcPos + i).toByte
          i += 1
        }
      }
    }

    final class Uint8Output(initialSize: Int) extends MsgPack.ByteOutput {
      private var bytes  = new Uint8Array(math.max(16, initialSize))
      private var cursor = 0

      def size: Int = cursor

      def toByteArray: Array[Byte] = {
        val out = new Array[Byte](cursor)
        var i   = 0
        while (i < cursor) {
          out(i) = bytes(i).toByte
          i += 1
        }
        out
      }

      def toUint8Array: Uint8Array = bytes.subarray(0, cursor)

      private def ensure(extra: Int): Unit = {
        val needed = cursor + extra
        if (needed > bytes.length) {
          var next = bytes.length
          while (next < needed) next = next << 1
          val resized = new Uint8Array(next)
          resized.set(bytes.subarray(0, cursor))
          bytes = resized
        }
      }

      def writeByte(x: Int): Unit = {
        ensure(1)
        bytes(cursor) = (x & 0xff).toShort
        cursor += 1
      }

      override def writeBytes(xs: Array[Byte], srcPos: Int, length: Int): Unit = {
        ensure(length)
        var i = 0
        while (i < length) {
          bytes(cursor + i) = (xs(srcPos + i).toInt & 0xff).toShort
          i += 1
        }
        cursor += length
      }

      override def writeShortBE(x: Int): Unit = {
        ensure(2)
        bytes(cursor) = ((x >>> 8) & 0xff).toShort
        bytes(cursor + 1) = (x & 0xff).toShort
        cursor += 2
      }

      override def writeIntBE(x: Int): Unit = {
        ensure(4)
        bytes(cursor) = ((x >>> 24) & 0xff).toShort
        bytes(cursor + 1) = ((x >>> 16) & 0xff).toShort
        bytes(cursor + 2) = ((x >>> 8) & 0xff).toShort
        bytes(cursor + 3) = (x & 0xff).toShort
        cursor += 4
      }

      override def writeLongBE(x: Long): Unit = {
        ensure(8)
        bytes(cursor) = ((x >>> 56) & 0xff).toShort
        bytes(cursor + 1) = ((x >>> 48) & 0xff).toShort
        bytes(cursor + 2) = ((x >>> 40) & 0xff).toShort
        bytes(cursor + 3) = ((x >>> 32) & 0xff).toShort
        bytes(cursor + 4) = ((x >>> 24) & 0xff).toShort
        bytes(cursor + 5) = ((x >>> 16) & 0xff).toShort
        bytes(cursor + 6) = ((x >>> 8) & 0xff).toShort
        bytes(cursor + 7) = (x & 0xff).toShort
        cursor += 8
      }
    }
  }
}
