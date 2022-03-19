package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.pass.{FnInlinePass, UnitExprElisionPass}
import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.StdLib

import java.nio.file.Paths
import scala.quoted.Expr

object Compiler {

  import RefOutliner.*
  import Retyper.*
  import TreeMapper.*

  private def runLocalOptPass(using Quoted) = //
    IntrinsifyPass.intrinsify // >>> MirrorPass.mirror(StdLib.Functions)

  private val GlobalOptPasses = //
    FnInlinePass.inlineAll >>> UnitExprElisionPass.eliminateUnitExpr // >>> FnPtrReturnToOutParamPass.transform

  // kind is there for cases where we have `object A extends B; abstract class B {def m = ???}`, B.m has receiver but A.m doesn't
  def deriveSignature(using q: Quoted)(f: q.DefDef, concreteKind: q.ClassKind): Result[p.Signature] = {
    def simpleTyper(t: q.TypeRepr) = Retyper.typer0(t).flatMap {
      case (_, t: p.Type) => t.success
      case (_, bad)       => s"erased arg type encountered $bad".fail
    }
    for {
      fnRtnTpe <- simpleTyper(f.returnTpt.tpe)
      fnArgsTpes <- f.paramss
        .flatMap(_.params)
        .collect { case d: q.ValDef => d.tpt.tpe }
        .traverse(simpleTyper(_))
      owningClass <- Retyper.clsSymTyper0(f.symbol.owner)

      receiver <- (concreteKind, owningClass) match {
        case (q.ClassKind.Object, q.ErasedClsTpe(_, _, _)) => None.success
        case (q.ClassKind.Object, s: p.Type.Struct)        => None.success
        case (q.ClassKind.Class, s: p.Type.Struct)         => Some(s).success
        case (kind, x)                                     => s"Illegal receiver (concrete kind=${kind}): $x".fail
      }
    } yield p.Signature(p.Sym(receiver.fold(f.symbol.fullName)(_ => f.symbol.name)), receiver, fnArgsTpes, fnRtnTpe)
  }

  def compileFnAndDependencies(using q: Quoted)(f: q.DefDef): Result[(List[p.Function], List[p.StructDef])] =
    (f :: Nil, List.empty[p.Function], List.empty[p.StructDef])
      .iterateWhileM { case (remaining, fnAcc, sdefAcc) =>
        remaining
          .traverse(compileFn(_))
          .map { xs =>
            val (depss, ys) = xs.unzip
            val deps        = depss.combineAll
            (deps.defs.values.toList, ys ::: fnAcc, deps.clss.values.toList ::: sdefAcc)
          }
      }(_._1.nonEmpty)
      .map((_, fns, sdefs) => (fns, sdefs))

  def compileFn(using q: Quoted)(f: q.DefDef): Result[(q.FnDependencies, p.Function)] = {
    println(s" -> Compile dependent method: ${f.show}")
    println(s" -> body(long):\n${f.show(using q.Printer.TreeAnsiCode).indent_(4)}")

    (for {

      (fnRtnValue, fnRtnTpe, c) <- q.FnContext().typer(f.returnTpt.tpe)

      // fn <- c.typer(f.symbol.owner)

      fnRtnTpe <- fnRtnTpe match {
        case tpe: p.Type => tpe.success.deferred
        case bad         => s"bad function return type $bad".fail.deferred
      }

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

      (owningClass, c) <- c.clsSymTyper(f.symbol.owner).deferred
      _ = println(s"[compiler] found: ${f} = ${owningClass}")

      receiver <- owningClass match {
        case s: p.Type.Struct        => Some(p.Named("this", s)).success.deferred
        case q.ErasedClsTpe(_, _, _) => None.success.deferred
        case x                       => s"Illegal receiver: ${x}".fail.deferred
      }

      body <- fnRtnValue.fold(f.rhs.traverse(c.mapTerm(_)))(x => Some((x, c)).success.deferred)

      mkFn = (xs: List[p.Stmt]) =>
        p.Function(p.Sym(receiver.fold(f.symbol.fullName)(_ => f.symbol.name)), receiver, namedArgs, fnRtnTpe, xs)
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

        val deptFn = mkFn((c.stmts))

        println(s"=====Dependent method ${deptFn.name} ====")
        println(deptFn.repr)

        (c.deps, deptFn)
    }).resolve
  }

  def compileClosure(using q: Quoted)(x: Expr[Any]): Result[(q.FnDependencies, List[(q.Ref, p.Type)], p.Function)] = {

    val term        = x.asTerm
    val pos         = term.pos
    val closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    println(s"========${closureName}=========")
    println(Paths.get(".").toAbsolutePath)
    println(s" -> name:               ${closureName}")
    println(s" -> body(Quotes):\n")
    pprint.pprintln(x.asTerm, indent = 2, showFieldNames = true)
    println(s" -> body(long):\n${x.asTerm.show(using q.Printer.TreeAnsiCode).indent_(4)}")

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
        s" -> all refs (typed):         \n${typedExternalRefs.map((tree, ref) => s"$ref => $tree").mkString("\n").indent_(4)}"
      )
      _ = println(s" -> captured refs:    \n${capturedNames.map(_._2.repr).mkString("\n").indent_(4)}")

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
        None,
        names,
        fnTpe,
        (c.stmts :+ p.Stmt.Return(p.Expr.Alias(returnTerm)))
      )
    } yield (c.deps, captures, closureFn)
  }

  def compileExpr(using q: Quoted)(x: Expr[Any]): Result[
    (
        // List[p.Type],
        List[(q.Ref, p.Type)],
        p.Program
    )
  ] =
    for {

      (deps, closureArgs, closureFn) <- compileClosure(x)

      _ = println("=======\nmain closure compiled\n=======")
      _ = println(s"${closureFn.repr}")

      _ = println(s" -> dependent methods = ${deps.defs.size}")
      _ = println(s" -> dependent structs = ${deps.clss.size}")




      // TODO rewrite me, this looks like garbage
      (deps, fns) <- (deps, List.empty[p.Function]).iterateWhileM { case (deps, fs) =>
        deps.defs.toList
          .traverse { case (sig: p.Signature, defdef) =>
            compileFn(defdef)
          }
          .map(xs => xs.unzip.bimap(_.combineAll, _ ++ fs))
          .map((d, f) => (d |+| deps.copy(defs = Map()), f))
      }(_._1.defs.nonEmpty)

      _ = println(s" -> dependent methods compiled")
      _ = println(s" -> dependent structs = ${deps.clss.size}")

      allFns = closureFn :: fns ::: StdLib.Functions

      optimised = GlobalOptPasses(allFns)

      _       = println(s"PolyAST:\n==>${optimised.map(_.repr).mkString("\n==>")}")
      clsDefs = deps.clss.values.toList
      _       = println(s"ClsDefs = ${clsDefs}")

      // outReturnParams = optimised.head.args.lastIndexOfSlice(closureFn.args) match {
      //   case -1 => ???
      //   case n  => optimised.head.args.take(n).map(_.tpe)
      // }

    } yield (
      // outReturnParams,
      closureArgs,
      p.Program(optimised.head, optimised.tail, clsDefs)
    )

}
