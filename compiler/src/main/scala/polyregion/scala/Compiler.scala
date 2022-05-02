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
import polyregion.ast.pass.MonoStructPass

object Compiler {

  import Remapper.*
  import Retyper.*

  private def runLocalOptPass(using Quoted) = IntrinsifyPass.intrinsify

  private def runProgramOptPasses(program: p.Program)(log: Log): Result[(p.Program, Log)] = {
    val (p0, l0) = FnInlinePass.run(program)(log)
    val (p1, l1) = UnitExprElisionPass.run(p0)(l0)
    val (p2, l2) = MonoStructPass.run(p1)(l1)
    (p2, l2).success
  }

  // kind is there for cases where we have `object A extends B; abstract class B {def m = ???}`, B.m has receiver but A.m doesn't
  def deriveSignature(using q: Quoted)(f: q.DefDef, concreteKind: q.ClassKind): Result[p.Signature] = {
    def simpleTyper(t: q.TypeRepr) = Retyper.typer0(t).map((_, t) => t)
    for {
      fnRtnTpe <- simpleTyper(f.returnTpt.tpe)
      fnArgsTpes <- f.paramss
        .flatMap(_.params)
        .collect { case d: q.ValDef => d.tpt.tpe }
        .traverse(simpleTyper(_))
      owningClass <- Retyper.clsSymTyper0(f.symbol.owner)
      // We're processing an unapplied defdef, so all names become type vars
      fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => p.Type.Var(name) }
      receiver <- (concreteKind, owningClass) match {
        // case (q.ClassKind.Object, q.ErasedClsTpe(_, _, _, _)) => None.success
        case (q.ClassKind.Object, s: p.Type.Struct) => None.success
        case (q.ClassKind.Class, s: p.Type.Struct)  => Some(s).success
        case (kind, x)                              => s"Illegal receiver (concrete kind=${kind}): $x".fail
      }
      argTpes = receiver.map(s => s.tpeVars.map(p.Type.Var(_))).getOrElse(Nil) ::: fnArgsTpes
    } yield p.Signature(p.Sym(f.symbol.fullName), fnTypeVars, receiver, argTpes, fnRtnTpe)
  }

  def compileAllDependencies(using q: Quoted)(
      deps: q.Dependencies,
      fnLut: Map[p.Signature, p.Function] = Map.empty,
      clsLut: Map[p.Sym, p.StructDef] = Map.empty
  )(log: Log): Result[(List[p.Function], q.Dependencies, Log)] = (deps.functions, List.empty[p.Function], deps, log)
    .iterateWhileM { case (remaining, fnAcc, depsAcc, log) =>
      remaining.toList
        .foldLeftM((fnAcc, depsAcc, log)) { case ((xs, deps, log), (defDef, invoke)) =>
          // specialise for invoke
          // fnLut.get(defSig) match {
          //   case Some(x) => (x :: xs, deps, log.info_(s"Dependent method replaced: ${defSig}")).success
          //   case None    => compileFn(defDef)(log).map { case ((x, deps1), log) => (x :: xs, deps |+| deps1, log) }
          // }

          compileFn(defDef)(log).map { case ((x, deps1), log) => (x :: xs, deps |+| deps1, log) }
        }
        .map((xs, deps, log) =>
          (
            deps.functions,
            xs,
            deps.copy(functions = Map.empty /*clss = deps.classes.map((k, v) => k -> clsLut.getOrElse(k, v))*/ ),
            log
          )
        )
    }(_._1.nonEmpty)
    .map(_.drop(1))

  private def deriveModuleStructCaptures(using q: Quoted)(d: q.Dependencies): List[p.Named] =
    d.classes
      .collect {
        case (c, ts) if c.symbol.flags.is(q.Flags.Module) && ts.size == 1 => ts.head
      }
      .map(t => p.Named(t.name.fqn.mkString("_"), t))
      .toList

  def compileFn(using q: Quoted)(f: q.DefDef)(log: Log): Result[((p.Function, q.Dependencies), Log)] =
    log.mark(s"Compile DefDef: ${f.name}") { log =>
      for {

        log <- log.info(s"Body", f.show(using q.Printer.TreeAnsiCode))

        rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${f.symbol.maybeOwner}) ${f.show}")

        // First we run the typer on the return type to see if we can just return a term based on the type.
        (fnRtnTerm, fnRtnTpe) <- Retyper.typer0(f.returnTpt.tpe)

        // We also run the typer on all the def's arguments.
        // TODO handle default value of args (ValDef.rhs)

        // q.TermParamClause()

        fnArgs <- f.termParamss.flatMap(_.params).foldMapM { arg => // all args should come in the form of a ValDef
          Retyper.typer0(arg.tpt.tpe).map((_, t) => ((arg, p.Named(arg.name, t)) :: Nil))
        }

        fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }

        // And then work out whether this def is part of a class/object instance or free-standing (e.g. local methods),
        // class defs will have a `this` receiver arg with the appropriate type.
        owningClass <- Retyper.clsSymTyper0(f.symbol.owner)
        log         <- log.info(s"Method owner: $owningClass")
        (receiver, receiverTpeVars) <- owningClass match {
          case t @ p.Type.Struct(_, tpeVars, _) => (Some(p.Named("this", t)), tpeVars).success
          case x                                => s"Illegal receiver: ${x}".fail
        }

        log <- log.info("DefDef arguments", fnArgs.map((a, n) => s"${a}(symbol=${a.symbol}) ~> $n")*)

        // Finally, we compile the def body like a closure or just return the term if we have one.
        ((rhsStmts, rhsDeps), log) <- fnRtnTerm match {
          case Some(t) => ((p.Stmt.Return(p.Expr.Alias(t)) :: Nil, q.Dependencies()), log).success
          case None =>
            for {
              // when run the outliner first and then replace any reference to function args or `this`
              // whatever is left will be module references which needs to be added to FnDep's var table
              // (captures, log) <- RefOutliner.outline(rhs)(log)
              // (capturedNames, captureScope) <- captures
              //   .foldMapM[Result, (List[(p.Named, q.Ref)], List[(q.Ref, p.Term)])] {
              //     case (root, ref, value, tpe)
              //         if root == ref && root.symbol.owner == f.symbol.owner && receiver.nonEmpty =>
              //       (Nil, Nil).success // this is a reference to `this`
              //     case (root, ref, value, tpe) =>
              //       // this is a generic reference possibly from the function argument
              //       val isArg = fnArgs.exists { (arg, n) =>
              //         arg.symbol == ref.underlying.symbol || arg.symbol == root.underlying.symbol
              //       }
              //       if (isArg) (Nil, Nil).success
              //       else
              //         value match {
              //           case Some(x) => (Nil, (ref -> x) :: Nil).success
              //           case None =>
              //             val name = ref match {
              //               case s @ q.Select(_, name) => s"_ref_${name}_${s.pos.startLine}_"
              //               case i @ q.Ident(_)        => i.name
              //             }
              //             val named = p.Named(name, tpe)
              //             ((named -> ref) :: Nil, (ref -> p.Term.Select(Nil, named)) :: Nil).success
              //         }
              //   }

              // log <- log.info("Captured vars before removal", fnDeps.vars.toList.map(_.toString)*)
              // log <- log.info("Captured vars after removal", fnDepsWithoutExtraCaptures.vars.toList.map(_.toString)*)

              // we pass an empty var table because
              ((rhsStmts, rhsTpe, rhsDeps), log) <- compileTerm(term = rhs, root = f.symbol, scope = Map.empty)(log)

              // log <- log.info("External captures", capturedNames.map((n, r) => s"$r(symbol=${r.symbol}) ~> ${n.repr}")*)

            } yield ((rhsStmts, rhsDeps), log)
        }

        compiledFn = p.Function(
          name = p.Sym(f.symbol.fullName),
          tpeVars = receiverTpeVars ::: fnTypeVars,
          receiver = receiver,
          args = fnArgs.map(_._2),
          captures = deriveModuleStructCaptures(rhsDeps),
          rtn = fnRtnTpe,
          body = rhsStmts
        )

        log <- log.info("Result", compiledFn.repr)
        _ = println(log.render().mkString("\n"))

      } yield ((compiledFn, rhsDeps), log)
    }

  def compileTerm(using q: Quoted)(term: q.Term, root: q.Symbol, scope: Map[q.Ref, p.Term])(log: Log) //
      : Result[((List[p.Stmt], p.Type, q.Dependencies), Log)] =
    log.mark(s"Compile term: ${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}") { log =>
      for {
        log            <- log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
        log            <- log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))
        (termValue, c) <- q.RemapContext(root = root, refs = scope).mapTerm(term)
        (_, termTpe)   <- Retyper.typer0(term.tpe)
        _ <-
          if (termTpe != termValue.tpe) {
            s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
          } else ().success
        statements                        = c.stmts :+ p.Stmt.Return(p.Expr.Alias(termValue))
        _                                 = println(s"Deps1 = ${c.deps.classes.values} ${c.deps.functions.values}")
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

    (captures, log) <- RefOutliner.outline(term)(log)
    (capturedNames, captureScope) = captures.foldMap[(List[(p.Named, q.Ref)], List[(q.Ref, p.Term)])] {
      (root, ref, value, tpe) =>
        (value, tpe) match {
          case (Some(x), _) => (Nil, (ref -> x) :: Nil)
          case (None, t) =>
            val name = ref match {
              case s @ q.Select(_, _) => s"_capture_${s.show}_${s.pos.startLine}_"
              case i @ q.Ident(_)     => s"_capture_${i.name}_${i.pos.startLine}_"
            }
            val named = p.Named(name, t)
            ((named -> ref) :: Nil, (ref -> p.Term.Select(Nil, named)) :: Nil)
        }
    }

    ((exprStmts, exprTpe, exprDeps), log) <- compileTerm(
      term = term,
      root = q.Symbol.noSymbol, // XXX not `spliceOwner` for now to avoid `this` captures
      scope = captureScope.toMap
    )(log)

    log <- log.info("Expr Stmts", exprStmts.map(_.repr).mkString("\n"))
    log <- log.info("External captures", capturedNames.map((n, r) => s"$r(symbol=${r.symbol}) ~> ${n.repr}")*)

    // we got a compiled term, compile all dependencies as well
    log <- log.info(s"Expr dependent methods", exprDeps.functions.values.map(_.toString).toList*)
    log <- log.info(s"Expr dependent structs", exprDeps.classes.values.map(_.toString).toList*)

    _ = println(log.render().mkString("\n"))

    // log                 <- log.info(s"Expr dependent vars   ", exprDeps.vars.map(_.toString).toList*)
    (depFns, deps, log) <- compileAllDependencies(exprDeps, StdLib.Functions, StdLib.StructDefs)(log)
    log                 <- log.info(s"Expr+Deps Dependent methods", deps.functions.values.map(_.toString).toList*)
    log                 <- log.info(s"Expr+Deps Dependent structs", deps.classes.values.map(_.toString).toList*)

    // sort the captures so that the order is stable for codegen
    // captures = deps.vars.toList.sortBy(_._1.symbol)
    // captures = capturedNames.toList.distinctBy(_._2).sortBy((name, ref) => ref.symbol.pos.map(_.start) -> name.symbol)
    // log                 <- log.info(s"Expr+Deps Dependent vars   ", captures.map(_.toString)*)

    exprFn = p.Function(
      name = p.Sym(exprName),
      tpeVars = Nil,
      receiver = None,
      args = Nil,
      captures = capturedNames.map(_._1) ++ deriveModuleStructCaptures(deps),
      rtn = exprTpe,
      body = exprStmts
    )

    sdefs <- deps.classes.toList.traverse { (clsDef, aps) =>
      structDef0(clsDef.symbol).map(_ -> aps)
    }

    captureNameToModuleRefTable = deps.classes.collect {
      case (c, ts) if c.symbol.flags.is(q.Flags.Module) && ts.size == 1 =>
        ts.head.name.fqn.mkString("_") -> selectObject(c.symbol)
    }.toMap

    unoptimisedProgram = p.Program(exprFn, depFns, sdefs.map(_._1))
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
    _ = println(log.render().mkString("\n"))

    capturedNameTable = capturedNames.map((name, ref) => name.symbol -> (name.tpe, ref)).toMap
    captured <- optimisedProgram.entry.captures.traverse { n =>
      // Struct type of symbols may have been modified through specialisation so we just validate whether it's still a struct for now
      capturedNameTable.get(n.symbol) match {
        case None                                       => (n -> captureNameToModuleRefTable(n.symbol)).success
        case Some((tpe, ref)) if tpe.kind == n.tpe.kind => (n -> ref).success
        case Some((tpe, ref)) =>
          s"Unexpected type conversion, capture was ${tpe.repr} for ${ref} but function expected ${n.repr}".fail

      }
    }

  } yield (captured, optimisedProgram, log)

}
