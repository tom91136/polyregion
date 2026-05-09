package polyregion.ast.pass

import polyregion.ast.{MsgPack, PolyAST as p}
import polyregion.ast.Log

import scala.scalajs.js.annotation.JSExportTopLevel
import scala.scalajs.js.typedarray.Uint8Array

object JsBundle {

  private sealed trait Entry
  private final case class Line(message: String, details: Vector[String]) extends Entry
  private final case class SubTree(log: TreeLog)                          extends Entry
  private final class TreeLog(val name: String) extends Log {
    val entries: scala.collection.mutable.ArrayBuffer[Entry] = scala.collection.mutable.ArrayBuffer.empty
    def info(message: String, details: String*): Unit        = entries += Line(message, details.toVector)
    def subLog(name: String): Log = {
      val sub = new TreeLog(name)
      entries += SubTree(sub)
      sub
    }
  }

  private def renderTree(log: TreeLog): Vector[String] = {
    def go(log: TreeLog, indent: String, isLast: Boolean): Vector[String] = {
      val header      = s"${indent}┏━ ${log.name} "
      val childIndent = indent + "┃ "
      val body = log.entries.toVector.flatMap {
        case Line(msg, details) =>
          val first = s"${childIndent}▓ $msg ▓"
          val detailLines = details.flatMap { d =>
            d.linesIterator.toList match {
              case x :: xs => s"${childIndent}╰ $x" :: xs.map(x => s"${childIndent}  $x")
              case Nil     => Nil
            }
          }
          first +: detailLines
        case SubTree(sub) => go(sub, childIndent, isLast = false)
      }
      val footer = s"${indent}┗━${"━" * (log.name.length + 2)}"
      (header +: body) :+ footer
    }
    go(log, "", isLast = true)
  }

  private val Opt: ProgramPass = (program, log) =>
    scala.Function.chain(
      List(
        printPass(IntrinsifyPass),
        // printPass(DynamicDispatchPass),
        printPass(SpecialisationPass),
        //    FnInlinePass,
        ConstantFoldPass,
        VarReducePass,
        UnitExprElisionPass,
        DeadArgEliminationPass
      ).map(p => p(_, log))
    )(program)

  private val passes: Map[String, ProgramPass | BoundaryPass[?]] = Map(
    "ConstantFold"          -> ConstantFoldPass,
    "DeadArgElimination"    -> DeadArgEliminationPass,
    "DeadStructElimination" -> DeadStructEliminationPass,
    "FnInline"              -> FnInlinePass,
    "Intrinsify"            -> IntrinsifyPass,
    "MonoStruct"            -> MonoStructPass,
    "Specialisation"        -> SpecialisationPass,
    "UnitExprElision"       -> UnitExprElisionPass,
    "VarReduce"             -> VarReducePass,
    "Opt"                   -> Opt
  )

  @JSExportTopLevel("runPass")
  def runPass(name: String, bytes: Uint8Array): Uint8Array = {
    val prog = MsgPack.decodeInput[p.Program](JsBytes.input(bytes)).fold(throw _, identity)
    val log  = new TreeLog(name)
    val out: p.Program = passes.get(name) match {
      case Some(pp: ProgramPass)     => pp(prog, log)
      case Some(bp: BoundaryPass[?]) => bp(prog, log)._2
      case None                      => throw new IllegalArgumentException(s"unknown pass: $name")
    }
    println(renderTree(log).mkString("\n"))
    val outBytes = JsBytes.output()
    MsgPack.encodeTo(out, outBytes)
    outBytes.toUint8Array
  }

  private object JsBytes {
    def input(bytes: Uint8Array): MsgPack.ByteInput = new Uint8Input(bytes)
    def output(initialSize: Int = 256): Uint8Output = new Uint8Output(initialSize)

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
