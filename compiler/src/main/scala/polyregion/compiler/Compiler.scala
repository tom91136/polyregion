package polyregion.compiler

import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import java.nio.file.Paths
import scala.quoted.Expr
import polyregion.compiler.pass.FnAllocElisionPass
import polyregion.compiler.pass.FnInlinePass

object Compiler {

  import RefOutliner.*
  import TreeMapper.*
  import Retyper.*

  import pass.IntrinsifyPass.*
  import pass.FnInlinePass.*
  import pass.UnitExprElisionPass.*

  def compileFn(using q: Quoted)(f: q.DefDef): Result[(q.FnContext, p.Function)] = (for {
    (fnRtnValue, fnTpe, c) <- q.FnContext().typer(f.returnTpt.tpe)
    args = f.paramss
      .flatMap(_.params)
      .collect { case d: q.ValDef => d }
    // TODO handle default value (ValDef.rhs)
    (namedArgs, c) <- args.foldMapM(a => c.typer(a.tpt.tpe).map((_, t, c) => (p.Named(a.name, t) :: Nil, c)))
    (lastTerm, c)  <- fnRtnValue.fold(c.mapTerm(f.rhs.get))((_, c).success.deferred)
  } yield c -> p.Function(
    p.Sym(f.symbol.fullName),
    namedArgs,
    fnTpe,
    c.stmts :+ p.Stmt.Return(p.Expr.Alias(lastTerm))
  )).resolve

  def compileExpr(using q: Quoted)(x: Expr[Any]): Result[(List[(q.Ref, p.Type)], p.Program)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n${x.asTerm.toString.indent(4)}")

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

      (returnTerm, c) <- c
        .inject(typedExternalRefs.map((ref, r) => ref.symbol -> r).toMap)
        .mapTerm(term)
        .resolve

      (_, fnTpe, c) <- c.typer(term.tpe).resolve

      args      = capturedNames.map(_._2)
      fnReturn  = p.Stmt.Return(p.Expr.Alias(returnTerm))
      closureFn = p.Function(p.Sym(closureName :: Nil), args.toList, fnTpe, c.stmts :+ fnReturn)

      _ <-
        if (fnTpe != returnTerm.tpe) {
          s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
        } else ().success

      dependentFns <- c.defs.toList.traverse { f =>
        compileFn(f)
      }
      (ctxs, fns) = dependentFns.unzip

      allFns = closureFn :: fns

      elided = FnInlinePass.inlineAll ( (allFns))
      // elided = allFns

      passes = intrinsify >>> eliminateUnitExpr

      optimised = elided.map(f => f.copy(body = passes(f.body)))

      _ = println(s"PolyAST:\n==>${optimised.map(_.repr).mkString("\n==>")}")

      clsDefs = c.clss.values.toList

      _ = println(s"ClsDefs = ${clsDefs}")
      // _ = println(s"DefDefs = ${c.defs.map(_.show).mkString("\n")}")

      captures = capturedNames.map((r, n) => r -> n.tpe)

    } yield (
      captures.toList,
      p.Program(optimised.head, optimised.tail, clsDefs)
    )
  }

}
