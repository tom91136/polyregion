package polyregion.ast.pass

import polyregion.ast.{MsgPack, PolyAST as p}
import polyregion.ast.Log

import scala.scalajs.js
import scala.scalajs.js.JSConverters.*
import scala.scalajs.js.annotation.JSExportTopLevel
import scala.scalajs.js.typedarray.{ArrayBuffer, Int8Array, Uint8Array}

object JsBundle {

  private sealed trait Entry
  private final case class Line(message: String, details: Vector[String])    extends Entry
  private final case class SubTree(log: TreeLog)                             extends Entry
  private final class TreeLog(val name: String) extends Log {
    val entries: scala.collection.mutable.ArrayBuffer[Entry] = scala.collection.mutable.ArrayBuffer.empty
    def info(message: String, details: String*): Unit                    = entries += Line(message, details.toVector)
    def subLog(name: String): Log = {
      val sub = new TreeLog(name)
      entries += SubTree(sub)
      sub
    }
  }

  private def renderTree(log: TreeLog): Vector[String] = {
    def go(log: TreeLog, indent: String, isLast: Boolean): Vector[String] = {
      val header = s"${indent}┏━ ${log.name} "
      val childIndent = indent + "┃ "
      val body = log.entries.toVector.flatMap {
        case Line(msg, details) =>
          val first = s"${childIndent}▓ $msg ▓"
          val detailLines = details.flatMap { d =>
            d.linesIterator.toList match {
              case x :: xs => (s"${childIndent}╰ $x") :: xs.map(x => s"${childIndent}  $x")
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
  def runPass(name: String, bytes: js.Any): ArrayBuffer = {
    val in   = readBytes(bytes)
    val prog = MsgPack.decode[p.Program](in).fold(throw _, identity)
    val log  = new TreeLog(name)
    val out: p.Program = passes.get(name) match {
      case Some(pp: ProgramPass)     => pp(prog, log)
      case Some(bp: BoundaryPass[?]) => bp(prog, log)._2
      case None                      => throw new IllegalArgumentException(s"unknown pass: $name")
    }
    println(renderTree(log).mkString("\n"))
    val outBytes = MsgPack.encode(out)
    val buf      = new ArrayBuffer(outBytes.length)
    new Int8Array(buf).set(outBytes.toJSArray)
    buf
  }

  private def readBytes(value: js.Any): Array[Byte] = value match {
    case ab: ArrayBuffer => new Int8Array(ab).toArray
    case u8: Uint8Array =>
      val out = new Array[Byte](u8.length)
      var i   = 0
      while (i < u8.length) { out(i) = u8(i).toByte; i += 1 }
      out
    case other =>
      throw new IllegalArgumentException(s"runPass: expected ArrayBuffer/Uint8Array, got ${js.typeOf(other)}")
  }
}
