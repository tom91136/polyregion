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

  private def runLocalOptPass(using Quoted) = //
    IntrinsifyPass.intrinsify >>>
      UnitExprElisionPass.eliminateUnitExpr

  private val GlobalOptPasses = //
    FnInlinePass.inlineAll >>>
      FnPtrReturnToOutParamPass.transform

  def compileFn(using q: Quoted)(f: q.DefDef): Result[(q.FnDependencies, p.Function)] = (for {
    (fnRtnValue, fnTpe, c) <- q.FnContext().typer(f.returnTpt.tpe)
    fnTpe <- fnTpe match {
      case tpe: p.Type => tpe.success.deferred
      case bad         => s"bad function return type $bad".fail.deferred
    }

    _ = println(s" -> Compile dependent method: ${f.show}")
    args = f.paramss
      .flatMap(_.params)
      .collect { case d: q.ValDef => d }
    // TODO handle default value (ValDef.rhs)
    (namedArgs, c) <- args.foldMapM(arg =>
      c.typer(arg.tpt.tpe).subflatMap {
        case (_, t: p.Type, c) => (p.Named(arg.name, t) :: Nil, c).success
        case (_, bad, c)       => s"erased arg type encountered $bad".fail
      }
    )

    body <- fnRtnValue.fold(f.rhs.traverse(c.mapTerm(_)))(x => Some((x, c)).success.deferred)
    mkFn   = (xs: List[p.Stmt]) => p.Function(p.Sym(f.symbol.fullName), namedArgs, fnTpe, xs)
    runOpt = (c: q.FnContext) => runLocalOptPass(c)

  } yield body match {
    case None =>
      (runOpt(c).deps, mkFn(Nil))
    case Some((value, preOptCtx)) =>
      val term = value match {
        case t: p.Term => t
        case _         => ???
      }

      val c = runOpt(preOptCtx.mapStmts(_ :+ p.Stmt.Return(p.Expr.Alias(term))))
      (c.deps, mkFn((c.stmts)))
  }).resolve

  def compileClosure(using q: Quoted)(x: Expr[Any]): Result[(q.FnDependencies, List[(q.Ref, p.Type)], p.Function)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n")
    pprint.pprintln(x.asTerm, indent = 2, showFieldNames = true)
    println(s" -> body(long):\n${x.asTerm.show(using q.Printer.TreeAnsiCode).indent(4)}")

    for {
      (typedExternalRefs, c) <- outline(term)
      // TODO if we have a reference with an erased closure type, we need to find the 
      // implementation and suspend it to FnContext otherwise we won't find the suspension in mapper

      // we can discard incoming references here safely iff we also never use them in the resulting p.Function
      capturedNames = typedExternalRefs
        .collect {
          case (r, q.Reference(name: String, tpe: p.Type)) if tpe != p.Type.Unit => (r, p.Named(name, tpe))
        }
        .distinctBy(_._1.symbol.pos)

      _ = println(
        s" -> all refs (typed):         \n${typedExternalRefs.map((tree, ref) => s"$ref => $tree").mkString("\n").indent(4)}"
      )
      _ = println(s" -> captured refs:    \n${capturedNames.map(_._2.repr).mkString("\n").indent(4)}")

      (returnTerm, c) <- c
        .inject(typedExternalRefs.map((ref, r) => ref.symbol -> r).toMap)
        .mapTerm(term)
        .resolve

      (_, fnTpe, preOptCtx) <- c.typer(term.tpe).resolve

      fnTpe <- fnTpe match {
        case tpe: p.Type => tpe.success
        case bad         => s"bad function return type $bad".fail
      }

      returnTerm <- returnTerm match {
        case term: p.Term => term.success
        case bad          => s"bad function return value $bad".fail
      }

      _ <-
        if (fnTpe != returnTerm.tpe) {
          s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
        } else ().success

      (captures, names) = capturedNames.toList.map((r, n) => (r -> n.tpe, n)).unzip

      c = runLocalOptPass(preOptCtx)

      closureFn = p.Function(
        p.Sym(closureName :: Nil),
        names,
        fnTpe,
        (c.stmts :+ p.Stmt.Return(p.Expr.Alias(returnTerm)))
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

      _ = println("=======\nmain closure compiled\n=======")
      _ = println(s"${closureFn.repr}")

      (deps, fns) <- (deps, List.empty[p.Function]).iterateWhileM { case (deps, fs) =>
        deps.defs.values.toList.traverse(compileFn(_)).map(xs => xs.unzip.bimap(_.combineAll, _ ++ fs))
      }(_._1.defs.nonEmpty)

      allFns = closureFn :: fns

      optimised = GlobalOptPasses(allFns)

      _       = println(s"PolyAST:\n==>${optimised.map(_.repr).mkString("\n==>")}")
      clsDefs = deps.clss.values.toList
      _       = println(s"ClsDefs = ${clsDefs}")

      outReturnParams = optimised.head.args.lastIndexOfSlice(closureFn.args) match {
        case -1 => ???
        case n  => optimised.head.args.take(n).map(_.tpe)
      }

    } yield (
      outReturnParams,
      closureArgs,
      p.Program(optimised.head, optimised.tail, clsDefs)
    )

}
