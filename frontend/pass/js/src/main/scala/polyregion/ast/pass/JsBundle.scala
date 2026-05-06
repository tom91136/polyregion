package polyregion.ast.pass

import polyregion.ast.{MsgPack, PolyAST as p}
import polyregion.ast.Log

import scala.scalajs.js
import scala.scalajs.js.JSConverters.*
import scala.scalajs.js.annotation.JSExportTopLevel
import scala.scalajs.js.typedarray.{ArrayBuffer, Int8Array, Uint8Array}

object JsBundle {

  private final class ConsoleLog(prefix: String) extends Log {
    def info(message: String, details: String*): Unit = println(s"[$prefix] $message ${details.mkString(", ")}")
    def subLog(name: String): Log                     = new ConsoleLog(s"$prefix/$name")
  }

  private val Opt: ProgramPass = (program, log) =>
    scala.Function.chain(
      List(
        printPass(IntrinsifyPass),
        // printPass(DynamicDispatchPass),
        printPass(SpecialisationPass),
       //    FnInlinePass,
        VarReducePass,
        UnitExprElisionPass,
        DeadArgEliminationPass
      ).map(p => p(_, log))
    )(program)

  private val passes: Map[String, ProgramPass | BoundaryPass[?]] = Map(
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
    val log  = new ConsoleLog(name)
    val out: p.Program = passes.get(name) match {
      case Some(pp: ProgramPass)     => pp(prog, log)
      case Some(bp: BoundaryPass[?]) => bp(prog, log)._2
      case None                      => throw new IllegalArgumentException(s"unknown pass: $name")
    }
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
