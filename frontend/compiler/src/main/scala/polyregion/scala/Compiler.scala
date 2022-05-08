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

  def deriveSignature(using q: Quoted)(f: q.DefDef): Result[p.Signature] = for {
    (_, fnRtnTpe) <- typer0(f.returnTpt.tpe)
    fnArgsTpes <- f.paramss
      .flatMap(_.params)
      .collect { case d: q.ValDef => d.tpt.tpe }
      .traverse(typer0(_).map(_._2))
    owningClass <- clsSymTyper0(f.symbol.owner)
    fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }
    (receiver, receiverTpeVars) <- owningClass match {
      case t @ p.Type.Struct(_, tpeVars, _) => (Some(t), tpeVars).success
      case x                                => s"Illegal receiver: ${x}".fail
    }
  } yield p.Signature(p.Sym(f.symbol.fullName), receiverTpeVars ::: fnTypeVars, receiver, fnArgsTpes, fnRtnTpe)

  def compileAllDependencies(using q: Quoted)(deps: q.Dependencies)(
      fnLut: Map[p.Signature, p.Function] = Map.empty
  ): Result[(List[p.Function], q.Dependencies, Log)] =
    for {
      log <- Log(s"Compile dependent functions")
      log <- deps.functions.toList.foldLeftM(log) { case (log, (fn, ivks)) =>
        log.info(q.DefDef(fn.symbol, rhsFn = _ => None).show, ivks.map(_.repr).toList.sorted*)
      }
      (_, fns, deps, logs) <- (deps.functions, List.empty[p.Function], deps, List.empty[Log]).iterateWhileM {
        (remaining, fnAcc, depsAcc, logs) =>
          remaining.toList
            .foldLeftM((fnAcc, depsAcc, logs)) { case ((xs, depss, logs), (defDef, invoke)) =>
              // XXX Generic specialisation on invoke is done in a pass later so we can ignore it for now
              deriveSignature(defDef).flatMap { s =>
                fnLut.get(s) match {
                  case Some(x) => Log(s"Replaced: ${s}").map(l => (x :: xs, deps, l :: logs))
                  case None =>
                    compileFn(defDef).map { case ((x, deps), log0) => (x :: xs, deps |+| depss, log :: logs) }
                }
              }
            }
            .map((xs, deps, log) =>
              (
                deps.functions,
                xs,
                deps.copy(functions = Map.empty),
                log
              )
            )
      }(_._1.nonEmpty)
    } yield (fns, deps, log)

  def deriveAllStructs(using q: Quoted)(deps: q.Dependencies)(
      clsLut: Map[p.Sym, p.StructDef] = Map.empty
  ): Result[(List[p.StructDef], Log)] = for {
    log <- Log(s"Compile dependent structs")
    classSdefs <- deps.classes.toList.traverse { (clsDef, aps) =>
      structDef0(clsDef.symbol).map(_ -> aps)
    }
    log <- log.info("Classes", classSdefs.map(_._1.repr)*)
    moduleSdef <- deps.modules.toList.distinct.traverse { (symbol, tpe) =>
      log.info(symbol.fullName) *>
        structDef0(symbol).map(_ -> Set(tpe))
    }
    log <- log.info("Modules", moduleSdef.map(_._1.repr)*)

    sdefs = (classSdefs ::: moduleSdef).map(_._1)
    log <- log.info(
      "Replacements",
      sdefs.map(sd => s"${sd.repr} => ${clsLut.get(sd.name).map(_.repr).getOrElse("(keep)")}")*
    )
  } yield (sdefs.map(sd => clsLut.getOrElse(sd.name, sd)), log)

  private def deriveModuleStructCaptures(using q: Quoted)(d: q.Dependencies): List[p.Named] =
    d.modules.map(_._2).toList.map(t => p.Named(t.name.fqn.mkString("_"), t))

  def compileFn(using q: Quoted)(f: q.DefDef): Result[((p.Function, q.Dependencies), Log)] = for {
    log <- Log(s"Compile DefDef: ${f.name}")
    log <- log.info(s"Body", f.show(using q.Printer.TreeAnsiCode))
    rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${f.symbol.maybeOwner}) ${f.show}")

    // First we run the typer on the return type to see if we can just return a term based on the type.
    (fnRtnTerm, fnRtnTpe) <- Retyper.typer0(f.returnTpt.tpe)

    // We also run the typer on all the def's arguments.
    // TODO handle default value of args (ValDef.rhs)
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
          ((rhsStmts, rhsTpe, rhsDeps), rhsLog) <- compileTerm(term = rhs, root = f.symbol, scope = Map.empty)

          // log <- log.info("External captures", capturedNames.map((n, r) => s"$r(symbol=${r.symbol}) ~> ${n.repr}")*)

        } yield ((rhsStmts, rhsDeps), log + rhsLog)
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

  def compileTerm(using q: Quoted)(
      term: q.Term,
      root: q.Symbol,
      scope: Map[q.Symbol, p.Term]
  ): Result[((List[p.Stmt], p.Type, q.Dependencies), Log)] = for {
    log            <- Log(s"Compile term: ${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}")
    log            <- log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
    log            <- log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))
    (termValue, c) <- q.RemapContext(root = root, refs = scope).mapTerm(term)
    (_, termTpe)   <- Retyper.typer0(term.tpe)
    _ <-
      if (termTpe != termValue.tpe) {
        s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
      } else ().success
    statements = c.stmts :+ p.Stmt.Return(p.Expr.Alias(termValue))
    _          = println(s"Deps1 = ${c.deps.classes.values} ${c.deps.functions.values}")

    (optStmts, optDeps) = runLocalOptPass(statements, c.deps)
  } yield ((optStmts, termTpe, optDeps), log)

  def compileExpr(using q: Quoted)(expr: Expr[Any]): Result[(List[(p.Named, q.Term)], p.Program, Log)] = for {
    log <- Log("Expr compiler")
    term = expr.asTerm
    // generate a name for the expr first
    exprName = s"${term.pos.sourceFile.name}:${term.pos.startLine}-${term.pos.endLine}"
    log <- log.info(s"Expr name: `${exprName}`")

    // outline here

    (captures, log) <- RefOutliner.outline(term)(log)
    (capturedNames, captureScope) = captures.foldMap[(List[(p.Named, q.Ref)], List[(q.Symbol, p.Term)])] {
      (root, ref, value, tpe) =>
        (value, tpe) match {
          case (Some(x), _) => (Nil, (ref.symbol -> x) :: Nil)
          case (None, t) =>
            val name = ref match {
              case s @ q.Select(_, _) => s"_capture_${s.show}_${s.pos.startLine}_"
              case i @ q.Ident(_)     => s"_capture_${i.name}_${i.pos.startLine}_"
            }
            val named = p.Named(name, t)
            ((named -> ref) :: Nil, (ref.symbol -> p.Term.Select(Nil, named)) :: Nil)
        }
    }

    ((exprStmts, exprTpe, exprDeps), termLog) <- compileTerm(
      term = term,
      root = q.Symbol.noSymbol, // XXX not `spliceOwner` for now to avoid `this` captures
      scope = captureScope.toMap
    )

    log <- log.info("Expr Stmts", exprStmts.map(_.repr).mkString("\n"))
    log <- log.info(
      "External captures",
      capturedNames.map((n, r) => s"$r(symbol=${r.symbol}, ${r.symbol.pos}) ~> ${n.repr}")*
    )

    // we got a compiled term, compile all dependencies as well
    log <- log.info(s"Expr dependent methods", exprDeps.functions.values.map(_.toString).toList*)
    log <- log.info(s"Expr dependent structs", exprDeps.classes.values.map(_.toString).toList*)

    _ = println(log.render().mkString("\n"))

    // log                 <- log.info(s"Expr dependent vars   ", exprDeps.vars.map(_.toString).toList*)
    (depFns, deps, depLog) <- compileAllDependencies(exprDeps)(StdLib.Functions)
    log                    <- log ~+ depLog

    log <- log.info(s"Expr+Deps Dependent methods", deps.functions.values.map(_.toString).toList*)
    log <- log.info(s"Expr+Deps Dependent structs", deps.classes.values.map(_.toString).toList*)

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

    (sdefs, sdefLog) <- deriveAllStructs(deps)(StdLib.StructDefs)
    log              <- log ~+ sdefLog

    captureNameToModuleRefTable = deps.modules
      .map { (symbol, struct) =>
        // It appears that `q.Ref.apply` prepends a `q.This(Some("this"))` at the root of the select if
        // the object is local to the scope, not really sure why it does that but we need to remove it.
        def removeThisFromRef(t: q.Symbol): q.Ref = q.Ref(t.companionModule) match {
          case select @ q.Select(root @ q.This(_), n) => removeThisFromRef(root.symbol).select(select.symbol)
          case x                                      => x
        }
        struct.name.fqn.mkString("_") -> removeThisFromRef(symbol)
      }

    unopt = p.Program(exprFn, depFns, sdefs)
    log <- log.info(
      s"Program compiled (unpot), structures = ${unopt.defs.size}, functions = ${unopt.functions.size}"
    )
    unoptLog <- Log("Unopt")
    unoptLog <- unoptLog.info(s"Structures = ${unopt.defs.size}", unopt.defs.map(_.repr)*)
    unoptLog <- unoptLog.info(s"Functions  = ${unopt.functions.size}", unopt.functions.map(_.signatureRepr)*)
    unoptLog <- unoptLog.info(s"Entry", unopt.entry.repr)

    // verify before optimisation
    (unoptVerification, unoptLog) <- VerifyPass.run(unopt)(unoptLog).success
    unoptLog <- unoptLog.info(
      s"Verifier",
      unoptVerification.map((f, xs) => s"${f.signatureRepr}\nError = ${xs.map("\t->" + _).mkString("\n")}")*
    )
    _ <-
      if (unoptVerification.exists(_._2.nonEmpty))
        s"Validation failed (error=${unoptVerification.map(_._2.size).sum})".fail
      else ().success
    log <- log ~+ unoptLog

    // run the global optimiser
    (opt, log) <- runProgramOptPasses(unopt)(log)

    // verify again after optimisation
    optLog                    <- Log("Opt")
    (optVerification, optLog) <- VerifyPass.run(opt)(optLog).success

    optLog <- optLog.info(s"Structures = ${opt.defs.size}", opt.defs.map(_.repr)*)
    optLog <- optLog.info(s"Functions  = ${opt.functions.size}", opt.functions.map(_.signatureRepr)*)
    optLog <- optLog.info(s"Entry", opt.entry.repr)

    optLog <- optLog.info(
      s"Verifier",
      optVerification.map((f, xs) => s"${f.signatureRepr}\nError = ${xs.map("\t->" + _).mkString("\n")}")*
    )
    _ <-
      if (optVerification.exists(_._2.nonEmpty))
        s"Validation failed (error=${optVerification.map(_._2.size).sum})".fail
      else ().success
    log <- log ~+ optLog

    capturedNameTable = capturedNames.map((name, ref) => name.symbol -> (name.tpe, ref)).toMap
    captured <- opt.entry.captures.traverse { n =>
      // Struct type of symbols may have been modified through specialisation so we just validate whether it's still a struct for now
      capturedNameTable.get(n.symbol) match {
        case None                                       => (n -> captureNameToModuleRefTable(n.symbol)).success
        case Some((tpe, ref)) if tpe.kind == n.tpe.kind => (n -> ref).success
        case Some((tpe, ref)) =>
          s"Unexpected type conversion, capture was ${tpe.repr} for ${ref} but function expected ${n.repr}".fail
      }
    }
    log <- log.info(s"Final captures", captured.map((name, ref) => s"${name.repr} = ${ref.show}")*)
    _ = println(log.render().mkString("\n"))

  } yield (captured, opt, log)

}
