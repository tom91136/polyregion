package polyregion.compiler

import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.internal.*

import java.nio.file.Paths
import scala.quoted.Expr

object Compiler {

  def compile(using q: Quoted)(x: Expr[Any]): Result[(List[(q.Ref, p.Type)], p.Program)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n${x.asTerm.toString.indent(4)}")

    import RefOutliner.*
    import Retyper.*
    import TreeIntrinsifier.*
    import TreeMapper.*
    import TreeUnitExprEliminator.*

    for {

      (typedExternalRefs, c) <- outline(term)

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

      clsDefs = c.clss.values.toList

      _ = println(s"ClsDefs = ${clsDefs}")
      _ = println(s"DefDefs = ${c.defs.map(_.show).mkString("\n")}")

      auxFns <- c.defs.toList.traverse { f =>
        for {
          (_, fnTpe, c) <- c.typer(f.returnTpt.tpe)

          args = f.paramss
            .flatMap(_.params)
            .collect { case d: q.ValDef => d }
          // TODO handle default value (ValDef.rhs)
          // FIXME extra stmts here
          (namedArgs, c) <- args.foldMapM(a => c.typer(a.tpt.tpe).map((_, t, c) => (p.Named(a.name, t) :: Nil, c)))
          (ref, c) <- c.mapTerm(f.rhs.get)
        } yield p.Function(p.Sym(f.symbol.fullName), namedArgs, fnTpe, c.stmts)
      }.resolve

      _ = println(s"AUX:\n==>${auxFns.map(_.repr).mkString("\n==>")}")

      returnTerm = ref
      _ <-
        if (fnTpe != returnTerm.tpe) {
          s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
        } else ().success

      fnReturn = p.Stmt.Return(p.Expr.Alias(returnTerm))

      passes = intrinsify >>> eliminateUnitExpr

      fnStmts = passes(c.stmts :+ fnReturn)

      _ = println(s" -> PolyAst:\n${fnStmts.map(_.repr).mkString("\n")}")

      args     = capturedNames.map(_._2)
      captures = capturedNames.map((r, n) => r -> n.tpe)

    } yield (
      captures.toList,
      p.Program(p.Function(p.Sym(closureName :: Nil), args.toList, fnTpe, fnStmts), Nil, clsDefs)
    )
  }

}
