package polyregion.scala

import cats.syntax.all.*
import cats.syntax.validated
import fansi.Str
import polyregion.ast.pass.{FnInlinePass, UnitExprElisionPass, VerifyPass}
import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.StdLib

import java.nio.file.Paths
import scala.collection.immutable.VectorMap
import scala.quoted.Expr

object Compiler {

  import Remapper.*
  import Retyper.*

  private def runLocalOptPass(using Quoted) = IntrinsifyPass.intrinsify

  private def runProgramOptPasses(program: p.Program)(log: Log): Result[(p.Program, Log)] = {
    val (p0, l0) = FnInlinePass.run(program)(log)
    val (p1, l1) = UnitExprElisionPass.run(p0)(l0)
    (p1, l1).success
  }

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

  def compileAllDependencies(using q: Quoted)(
      deps: q.FnDependencies,
      fnLut: Map[p.Signature, p.Function] = Map.empty,
      clsLut: Map[p.Sym, p.StructDef] = Map.empty
  )(log: Log): Result[(List[p.Function], q.FnDependencies, Log)] = (deps.defs, List.empty[p.Function], deps, log)
    .iterateWhileM { case (remaining, fnAcc, depsAcc, log) =>
      remaining.toList
        .foldLeftM((fnAcc, depsAcc, log)) { case ((xs, deps, log), (defSig, defDef)) =>
          fnLut.get(defSig) match {
            case Some(x) => (x :: xs, deps, log.info_(s"Dependent method replaced: ${defSig}")).success
            case None    => compileFn(defDef)(log).map { case ((x, deps1), log) => (x :: xs, deps |+| deps1, log) }
          }
        }
        .map((xs, deps, log) =>
          (deps.defs, xs, deps.copy(defs = Map.empty, clss = deps.clss.map((k, v) => k -> clsLut.getOrElse(k, v))), log)
        )
    }(_._1.nonEmpty)
    .map(_.drop(1))

  def compileFn(using q: Quoted)(f: q.DefDef)(log: Log): Result[((p.Function, q.FnDependencies), Log)] =
    log.mark(s"Compile DefDef: ${f.name}") { log =>
      for {

        log <- log.info(s"Body", f.show(using q.Printer.TreeAnsiCode))
        _ = println(s"Compile DefDef: ${f.name}")

        rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${f.symbol.maybeOwner}) ${f.show}")

        // first we run the typer on the return type to see if we can just return a term based on the type
        (fnRtnValue, fnRtnTpe, fnRtnDeps) <- Retyper.typer1(f.returnTpt.tpe)
        // we then validate the return types
        fnRtnTpe <- fnRtnTpe match {
          case tpe: p.Type => tpe.success
          case bad         => s"Bad function return type $bad".fail
        }
        // we also run the typer on all the def's arguments
        // TODO handle default value of args (ValDef.rhs)
        (fnArgs, fnArgDeps) <- f.paramss.flatMap(_.params).foldMapM {
          case arg: q.ValDef => // all args should come in the form of a ValDef
            Retyper.typer1(arg.tpt.tpe).flatMap {
              case (_, t: p.Type, c) => ((arg, p.Named(arg.name, t)) :: Nil, c).success
              case (_, bad, c)       => s"Erased arg type encountered $bad".fail
            }
          case _ => ???
        }
        // and work out whether this def is part a class instance or free-standing
        // class defs will have a `this` receiver arg with the appropriate type
        (owningClass, fnReceiverDeps) <- Retyper.clsSymTyper1(f.symbol.owner)
        log                           <- log.info(s"Method owner: $owningClass")
        receiver <- owningClass match {
          case s: p.Type.Struct        => Some(p.Named("this", s)).success
          case q.ErasedClsTpe(_, _, _) => None.success
          case x                       => s"Illegal receiver: ${x}".fail
        }

        log <- log.info("DefDef arguments", fnArgs.map((a, n) => s"${a}(symbol=${a.symbol}) ~> $n")*)

        // finally, we compile the def body like a closure or just return the term if we have one
        ((rhsStmts, rhsDeps, rhsCaptures), log) <- fnRtnValue match {
          case Some(t) => ((p.Stmt.Return(p.Expr.Alias(t)) :: Nil, q.FnDependencies(), Nil), log).success
          case None =>
            for {
              // when run the outliner first and then replace any reference to function args or `this`
              // whatever is left will be module references which needs to be added to FnDep's var table
              ((captures, captureDeps), log) <- RefOutliner.outline(rhs)(log)
              (capturedNames, captureScope) <- captures
                .foldMapM[Result, (List[(p.Named, q.Ref)], List[(q.Ref, p.Term)])] {
                  case (root, ref, value, tpe: p.Type)
                      if root == ref && root.symbol.owner == f.symbol.owner && receiver.nonEmpty =>
                    (Nil, Nil).success // this is a reference to `this`
                  case (root, ref, value, tpe: p.Type) =>
                    // this is a generic reference possibly from the function argument
                    val isArg = fnArgs.exists { (arg, n) =>
                      arg.symbol == ref.underlying.symbol || arg.symbol == root.underlying.symbol
                    }
                    if (isArg) (Nil, Nil).success
                    else
                      value match {
                        case Some(x) => (Nil, (ref -> x) :: Nil).success
                        case None =>
                          val name = ref match {
                            case s @ q.Select(_, name) => s"_ref_${name}_${s.pos.startLine}_"
                            case i @ q.Ident(_)        => i.name
                          }
                          val named = p.Named(name, tpe)
                          ((named -> ref) :: Nil, (ref -> p.Term.Select(Nil, named)) :: Nil).success
                      }

                  case (root, ref, value, bad) => ???
                }

              // log <- log.info("Captured vars before removal", fnDeps.vars.toList.map(_.toString)*)
              // log <- log.info("Captured vars after removal", fnDepsWithoutExtraCaptures.vars.toList.map(_.toString)*)

              // we pass an empty var table because
              ((rhsStmts, rhsTpe, rhsDeps), log) <- compileTerm(rhs, captureScope.toMap)(log)
              _ = println(s"DDD=${f.name} = ${rhsDeps.defs.values.map(x => x.show).toList}")
              log <- log.info("External captures", capturedNames.map((n, r) => s"$r(symbol=${r.symbol}) ~> ${n.repr}")*)

            } yield (
              (
                rhsStmts,
                captureDeps |+| rhsDeps |+| q.FnDependencies( /* vars = capturedNames.toMap */ ),
                capturedNames
              ),
              log
            )
        }

        fnDeps <- (fnRtnDeps |+| fnArgDeps |+| fnReceiverDeps |+| rhsDeps).success

        compiledFn = p.Function(
          name = p.Sym(receiver.fold(f.symbol.fullName)(_ => f.symbol.name)),
          receiver = receiver,
          args = fnArgs.map(_._2),
          captures = rhsCaptures.map(_._1),
          rtn = fnRtnTpe,
          body = rhsStmts
        )

        log <- log.info("Result", compiledFn.repr)

      } yield ((compiledFn, fnDeps), log)
    }

  def compileTerm(using q: Quoted)(term: q.Term, scope: Map[q.Ref, p.Term])(
      log: Log
  ): Result[((List[p.Stmt], p.Type, q.FnDependencies), Log)] =
    log.mark(s"Compile term: ${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}") { log =>
      for {

        log <- log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
        log <- log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))

        // first, outline what variables this term captures
        // ((typedExternalRefs, c), log) <- RefOutliner.outline(term)(log)
        // Map[q.Ref, p.Named | p.Term]

        // TODO if we have a reference with an erased closure type, we need to find the
        // implementation and suspend it to FnContext otherwise we won't find the suspension in mapper

        // we can discard incoming references here safely iff we also never use them in the resulting p.Function
        // capturedNames = typedExternalRefs
        //   .collect {
        //     case (root, ref, q.Reference(name: String, tpe: p.Type)) if tpe != p.Type.Unit =>
        //       // p.Named | p.Term

        //       (p.Named(name, tpe), (root, ref))
        //     case (root, ref, q.Reference(name: p.Term, tpe: p.Type)) => ??? // TODO handle default value
        //   }
        // // .distinctBy(_._2.symbol.pos)

        // xx = typedExternalRefs
        //   .map[(q.Ref, p.Named | p.Term)] { (root, ref, value, tpe) =>
        //     (value, tpe) match {
        //       case (Some(x), _) => ref -> x
        //       case (None, t: p.Type) =>
        //         val name = ref match {
        //           case s @ q.Select(_, name) => s"_ref_${name}_${s.pos.startLine}_"
        //           case i @ q.Ident(_)        => i.name
        //         }
        //         ref -> p.Named(name, t)
        //       case (None, bad) => ???
        //     }
        //   }
        //   .toMap

        (termValue, c) <- q
          .FnContext()
          .inject(scope)
          .mapTerm(term)

        (_, termTpe, c) <- c.typer(term.tpe)

        // validate term type and values
        termTpe <- termTpe match {
          case tpe: p.Type => tpe.success
          case bad         => s"Bad function return type $bad".fail
        }
        termValue <- termValue match {
          case term: p.Term => term.success
          case bad          => s"Bad function return value $bad".fail
        }
        _ <-
          if (termTpe != termValue.tpe) {
            s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
          } else ().success

        // (captures, names) = capturedNames.toList.map((n, r) => (r -> n.tpe, n)).unzip

        statements = c.stmts :+ p.Stmt.Return(p.Expr.Alias(termValue))

        (optimisedStmts, optimisedFnDeps) = runLocalOptPass(statements, c.deps)

      } yield ((optimisedStmts, termTpe, optimisedFnDeps), log)
    }

  def compileExpr(using q: Quoted)(expr: Expr[Any]): Result[(List[(p.Named, q.Ref)], p.Program, Log)] = for {
    log <- Log("Expr compiler")
    term = expr.asTerm
    // generate a name for the expr first
    exprName = s"${term.pos.sourceFile.name}:${term.pos.startLine}-${term.pos.endLine}"
    log <- log.info(s"Expr name: `${exprName}`")

    // outline here

    ((captures, captureDeps), log) <- RefOutliner.outline(term)(log)
    (capturedNames, captureScope) = captures.foldMap[(List[(p.Named, q.Ref)], List[(q.Ref, p.Term)])] {
      (root, ref, value, tpe) =>
        (value, tpe) match {
          case (Some(x), _) => (Nil, (ref -> x) :: Nil)
          case (None, t: p.Type) =>
            val name = ref match {
              case s @ q.Select(_, _) => s"_capture_${s.show}_${s.pos.startLine}_"
              case i @ q.Ident(_)     => s"_capture_${i.name}_${i.pos.startLine}_"
            }
            val named = p.Named(name, t)
            ((named -> ref) :: Nil, (ref -> p.Term.Select(Nil, named)) :: Nil)
          case (None, bad) => ???
        }
    }

    ((exprStmts, exprTpe, exprDeps), log) <- compileTerm(term, captureScope.toMap)(log)
    log                                   <- log.info("Expr Stmts", exprStmts.map(_.repr).mkString("\n"))
    log      <- log.info("External captures", capturedNames.map((n, r) => s"$r(symbol=${r.symbol}) ~> ${n.repr}")*)
    exprDeps <- (exprDeps |+| captureDeps |+| q.FnDependencies( /* vars = xxx.toMap*/ )).success

    // we got a compiled term, compile all dependencies as well
    log <- log.info(s"Expr dependent methods", exprDeps.defs.map(_.toString).toList*)
    log <- log.info(s"Expr dependent structs", exprDeps.clss.map(_.toString).toList*)
    // log                 <- log.info(s"Expr dependent vars   ", exprDeps.vars.map(_.toString).toList*)
    (depFns, deps, log) <- compileAllDependencies(exprDeps, StdLib.Functions, StdLib.StructDefs)(log)
    log                 <- log.info(s"Expr+Deps Dependent methods", deps.defs.map(_.toString).toList*)
    log                 <- log.info(s"Expr+Deps Dependent structs", deps.clss.map(_.toString).toList*)

    // sort the captures so that the order is stable for codegen
    // captures = deps.vars.toList.sortBy(_._1.symbol)
    captures = capturedNames.toList.distinctBy(_._2).sortBy((name, ref) => ref.symbol.pos.map(_.start) -> name.symbol)
    // log                 <- log.info(s"Expr+Deps Dependent vars   ", captures.map(_.toString)*)

    exprFn = p.Function(
      name = p.Sym(exprName),
      receiver = None,
      args = Nil,
      captures = captures.map(_._1),
      rtn = exprTpe,
      body = exprStmts
    )
    unoptimisedProgram = p.Program(exprFn, depFns, deps.clss.values.toList)
    log <- log.info(s"Program compiled, dependencies: ${unoptimisedProgram.functions.size}")
    log <- log.info(s"Program compiled, structures", unoptimisedProgram.defs.map(_.repr)*)

    //  _ = println(log.render().mkString("\n"))
    // verify before optimisation
    (unoptimisedVerification, log) <- VerifyPass.run(unoptimisedProgram)(log).success
    log <- log.info(
      s"Verification (unopt)",
      unoptimisedVerification.map((f, xs) => s"${f.signatureRepr}\nE=${xs.map("\t->" + _).mkString("\n")}")*
    )
    _ <-
      if (unoptimisedVerification.exists(_._2.nonEmpty))
        s"Validation failed (unopt, error=${unoptimisedVerification.map(_._2.size).sum})".fail
      else ().success

    // _ = println(log.render().mkString("\n"))

    // run the global optimiser
    (optimisedProgram, log) <- runProgramOptPasses(unoptimisedProgram)(log)

    // verify again after optimisation
    (optimisedVerification, log) <- VerifyPass.run(optimisedProgram)(log).success
    log <- log.info(
      s"Verification(opt)",
      optimisedVerification.map((f, xs) => s"${f.signatureRepr}\n${xs.map("\t" + _).mkString("\n")}")*
    )
    _ <-
      if (optimisedVerification.exists(_._2.nonEmpty))
        s"Validation failed (opt, error=${optimisedVerification.map(_._2.size).sum})".fail
      else ().success

    log <- log.info(s"Program optimised, dependencies", optimisedProgram.functions.map(_.repr)*)
    log <- log.info(s"Program optimised, entry", optimisedProgram.entry.repr)
    // log                 <- log.info(s"Program optimised, entry", captures.map(_.toString)*)


    capturesLUT = captures.toMap
    _ = println(log.render().mkString("\n"))

  } yield (optimisedProgram.entry.captures.map{ n =>  n -> capturesLUT(n) }, optimisedProgram, log)

}
