package polyregion.compiler

import scala.quoted.Expr
import polyregion.internal.*
import polyregion.ast.PolyAst
import java.nio.file.Paths

object Compiler {

  def compile(using q: Quoted)(x: Expr[Any]): Result[(List[(q.Ref, PolyAst.Type)], PolyAst.Program)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n${x.asTerm.toString.indent(4)}")

    for {

      (ctx, typedExternalRefs) <- RefOutliner.outline(term)

      capturedNames = typedExternalRefs
        .collect {
          case (r, q.Reference(name: String, tpe)) if tpe != PolyAst.Type.Unit =>
            (r, PolyAst.Named(name, tpe))
        }
        .distinctBy(_._1.symbol.pos)

      _ = println(
        s" -> all refs (typed):         \n${typedExternalRefs.map((tree, ref) => s"$ref => $tree").mkString("\n").indent(4)}"
      )
      _ = println(s" -> captured refs:    \n${capturedNames.map(_._2.repr).mkString("\n").indent(4)}")
      _ = println(s" -> body(long):\n${x.asTerm.show(using q.Printer.TreeAnsiCode).indent(4)}")



    } yield ???
  }

}
