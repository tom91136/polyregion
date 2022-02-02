package polyregion.compiler

import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import java.nio.file.Paths
import scala.quoted.Expr

import polyregion.compiler.pass.*

object Compiler {

  import RefOutliner.*
  import TreeMapper.*
  import Retyper.*

  private val LocalOptPasses = //
    IntrinsifyPass.intrinsify >>>
      UnitExprElisionPass.eliminateUnitExpr

  private val GlobalOptPasses = //
    FnInlinePass.inlineAll >>>
      FnAllocElisionPass.transform

  def compileFn(using q: Quoted)(f: q.DefDef): Result[(q.FnDependencies, p.Function)] = (for {
    (fnRtnValue, fnTpe, c) <- q.FnContext().typer(f.returnTpt.tpe)
    _ = println(s" -> Compile dependent method: ${f.show}")
    args = f.paramss
      .flatMap(_.params)
      .collect { case d: q.ValDef => d }
    // TODO handle default value (ValDef.rhs)
    (namedArgs, c) <- args.foldMapM(a => c.typer(a.tpt.tpe).map((_, t, c) => (p.Named(a.name, t) :: Nil, c)))
    (lastTerm, c)  <- fnRtnValue.fold(c.mapTerm(f.rhs.get))((_, c).success.deferred)
  } yield c.deps -> p.Function(
    p.Sym(f.symbol.fullName),
    namedArgs,
    fnTpe,
    LocalOptPasses(c.stmts :+ p.Stmt.Return(p.Expr.Alias(lastTerm)))
  )).resolve

  def compileClosure(using q: Quoted)(x: Expr[Any]): Result[(q.FnDependencies, List[(q.Ref, p.Type)], p.Function)] = {

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

      _ <-
        if (fnTpe != returnTerm.tpe) {
          s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
        } else ().success

      (captures, names) = capturedNames.toList.map((r, n) => (r -> n.tpe, n)).unzip
      closureFn = p.Function(
        p.Sym(closureName :: Nil),
        names,
        fnTpe,
        LocalOptPasses(c.stmts :+ p.Stmt.Return(p.Expr.Alias(returnTerm)))
      )
    } yield (c.deps, captures, closureFn)
  }

  def compileExpr(using q: Quoted)(x: Expr[Any]): Result[
    (
        List[p.Type],
        List[(q.Ref, p.Type)],
        p.Program
    )
  ] =
    for {

      (deps, closureArgs, closureFn) <- compileClosure(x)

      (deps, fns) <- (deps, List.empty[p.Function]).iterateWhileM { case (deps, fs) =>
        deps.defs.toList.traverse(compileFn(_)).map(xs => xs.unzip.bimap(_.combineAll, _ ++ fs))
      }(_._1.defs.nonEmpty)

      allFns = closureFn :: fns

      optimised = GlobalOptPasses(allFns)

      _       = println(s"PolyAST:\n==>${optimised.map(_.repr).mkString("\n==>")}")
      clsDefs = deps.clss.values.toList
      _       = println(s"ClsDefs = ${clsDefs}")

      allocArgs = optimised.head.args.lastIndexOfSlice(closureFn.args) match {
        case -1 => ???
        case n  => optimised.head.args.take(n).map(_.tpe)
      }

    } yield (
      allocArgs,
      closureArgs,
      p.Program(optimised.head, optimised.tail, clsDefs)
    )

}
