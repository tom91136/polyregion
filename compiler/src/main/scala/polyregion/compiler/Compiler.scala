package polyregion.compiler

import scala.quoted.Expr
import polyregion.internal.*
import polyregion.ast.{PolyAst => p}
import java.nio.file.Paths
import cats.syntax.all.*

object Compiler {

  def compile(using q: Quoted)(x: Expr[Any]): Result[(List[(q.Ref, p.Type)], p.Program)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n${x.asTerm.toString.indent(4)}")

    import TreeMapper.*
    import Retyper.*
    import RefOutliner.*
    import TreeIntrinsifier.*

    for {

      (c, typedExternalRefs) <- outline(term)

      capturedNames = typedExternalRefs
        .collect {
          case (r, q.Reference(name: String, tpe)) if tpe != p.Type.Unit =>
            (r, p.Named(name, tpe))
        }
        .distinctBy(_._1.symbol.pos)

      _ = println(
        s" -> all refs (typed):         \n${typedExternalRefs.map((tree, ref) => s"$ref => $tree").mkString("\n").indent(4)}"
      )
      _ = println(s" -> captured refs:    \n${capturedNames.map(_._2.repr).mkString("\n").indent(4)}")
      _ = println(s" -> body(long):\n${x.asTerm.show(using q.Printer.TreeAnsiCode).indent(4)}")

      (ref, c) <- c
        .inject(typedExternalRefs.map((ref, r) => ref.symbol -> r).toMap)
        .mapTerm(term)
        .resolve

      (_, fnTpe, c) <- c.typer(term.tpe).resolve

      clsDefs <- c.clss.toList.traverse(s => lowerProductType(s.typeSymbol)).resolve

      _ = println(s"ClsDefs=${clsDefs}")

      returnTerm = ref
      _ <-
        if (fnTpe != returnTerm.tpe) {
          s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
        } else ().success
      fnReturn = p.Stmt.Return(p.Expr.Alias(returnTerm))

      fnStmts = intrinsify(c.stmts :+ fnReturn)

      _ = println(s" -> PolyAst:\n${fnStmts.map(_.repr).mkString("\n")}")

      args     = capturedNames.map(_._2)
      captures = capturedNames.map((r, n) => r -> n.tpe)

    } yield (
      captures.toList,
      p.Program(p.Function(closureName, args.toList, fnTpe, fnStmts), Nil, clsDefs)
    )
  }

}
