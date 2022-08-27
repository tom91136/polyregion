package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.pass.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.StdLib

import java.nio.file.Paths
import scala.quoted.Expr

object Compiler {

  import Remapper.*
  import Retyper.*

  private def runLocalOptPass(using Quoted) = IntrinsifyPass.intrinsify

  private val ProgramPasses: List[ProgramPass] = List(
    FnInlinePass,           //
    UnitExprElisionPass,    //
    DeadArgEliminationPass, //
    MonoStructPass          //
  )

  private def runProgramOptPasses(program: p.Program)(log: Log): Result[(p.Program, Log)] =
    Function.chain(ProgramPasses.map(_.tupled))((program, log)).success

  // This derives the signature based on the *owning* symbol only!
  // This means, for cases where the method's direct root can be different, the actual owner(i.e. the context of the definition site) is used.
  def deriveSignature(using q: Quoted)(f: q.DefDef): Result[p.Signature] = for {
    (_ -> fnRtnTpe, _)      <- typer0(f.returnTpt.tpe) // Ignore the class witness as we're deriving the signature.
    (fnArgsTpeWithTerms, _) <- typer0N(f.paramss.flatMap(_.params).collect { case d: q.ValDef => d.tpt.tpe })
    owningClass             <- clsSymTyper0(f.symbol.owner)
    fnArgsTpes = fnArgsTpeWithTerms.map(_._2)
    fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }
    (receiver, receiverTpeVars) <- owningClass match {
      case t @ p.Type.Struct(_, tpeVars, _) => (Some(t), tpeVars).success
      case x                                => s"Illegal receiver: $x".fail
    }
  } yield p.Signature(p.Sym(f.symbol.fullName), receiverTpeVars ::: fnTypeVars, receiver, fnArgsTpes, fnRtnTpe)

  def compileAllDependencies(using q: Quoted)(
      deps: q.Dependencies
  )(fnLut: Map[p.Signature, p.Function] = Map.empty): Result[(List[p.Function], q.Dependencies, Log)] = for {
    log <- Log(s"Compile ${deps.functions.size} dependent function(s)")
    log <- deps.functions.toList.foldLeftM(log) { case (log, (fn, ivks)) =>
      log.info(q.DefDef(fn.symbol, rhsFn = _ => None).show, ivks.map(_.repr).toList.sorted*)
    }
    (_ /*input*/, fns, deps, logs) <- (
      deps.functions,
      List.empty[p.Function],
      deps.copy(functions = Map.empty),
      List.empty[Log]
    ).iterateWhileM { (remaining, fnAcc, depsAcc, logs) =>
      remaining.toList
        .foldLeftM((fnAcc, depsAcc, logs)) { case ((xs, depss, logs), (defDef, invoke)) =>
          // XXX Generic specialisation on invoke is done in a pass later so we can ignore it for now
          deriveSignature(defDef).flatMap { s =>
            fnLut.get(s) match {
              case Some(x) =>
                println(s"Replace: replace ${s.repr}")
                Log(s"${s.repr}")
                  .map(
                    _.info_("Callsites", invoke.map(_.repr).toList: _*)
                      .info_("Replacing with impl:", x.repr)
                  )
                  .map(l => (x :: xs, depss, l :: logs))
              case None =>



                println(s"Replace: cannot replace ${s.repr}\n${fnLut.keySet.toList.map(x => s"\t${x.repr}").sorted.mkString("\n")} \n ${fnLut.keySet.map(_.copy(tpeVars = Nil) ).contains(s.copy(tpeVars = Nil))}")
                for {
                  l                 <- Log(s"Compile (no replacement): ${s.repr}")
                  ((x, deps), log0) <- compileFn(defDef)
                } yield (x :: xs, deps |+| depss, (l + log0) :: logs)
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

  } yield (fns, deps, log + Log("Replacements", logs.toVector))

  def deriveAllStructs(using q: Quoted)(deps: q.Dependencies)(
      clsLut: Map[p.Sym, p.StructDef] = Map.empty
  ): Result[(List[p.StructDef], Log)] = for {
    log <- Log(s"Compile dependent structs")
    (replacedClasses, classes) <- deps.classes.toList.partitionEitherM { case (clsDef, aps) =>
      clsLut.get(structName0(clsDef.symbol)) match {
        case Some(x) => Left(x).success
        case None    => structDef0(clsDef.symbol).map(Right(_))
      }
    }
    (replacedModules, modules) <- deps.modules.toList.partitionEitherM { case (symbol, tpe) =>
      clsLut.get(structName0(symbol)) match {
        case Some(x) => Left(x).success
        case None    => structDef0(symbol).map(Right(_))
      }
    }
    log <- log.info("Replaced classes ", replacedClasses.map(_.repr)*)
    log <- log.info("Replaced modules ", replacedModules.map(_.repr)*)
    log <- log.info("Reflected classes", classes.map(_.repr)*)
    log <- log.info("Reflected modules", modules.map(_.repr)*)
  } yield (replacedModules ::: modules ::: replacedClasses ::: classes, log)

  private def deriveModuleStructCaptures(using q: Quoted)(d: q.Dependencies): List[p.Named] =
    d.modules.values.toList.map(t => p.Named(t.name.fqn.mkString("_"), t))

  def compileFn(using q: Quoted)(f: q.DefDef, intrinsify: Boolean = true): Result[((p.Function, q.Dependencies), Log)] =
    for {
      log <- Log(s"Compile DefDef: ${f.name}")
      log <- log.info(s"Body", f.show(using q.Printer.TreeAnsiCode))
      rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${f.symbol.maybeOwner}) ${f.show}")

      // First we run the typer on the return type to see if we can just return a term based on the type.
      (fnRtnTerm -> fnRtnTpe, fnRtnWit) <- Retyper.typer0(f.returnTpt.tpe)

      // We also run the typer on all the def's arguments, all of which should come in the form of a ValDef.
      // TODO handle default value of args (ValDef.rhs)
      (fnArgs, fnArgsWit) <- f.termParamss.flatMap(_.params).foldMapM { arg =>
        Retyper.typer0(arg.tpt.tpe).map { case (_ -> t, wit) => ((arg, p.Named(arg.name, t)) :: Nil, wit) }
      }

      fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }

      // And then work out whether this def is part of a class/object instance or free-standing (e.g. local methods),
      // class defs will have a `this` receiver arg with the appropriate type.
      owningClass <- Retyper.clsSymTyper0(f.symbol.owner)
      log         <- log.info(s"Method owner: $owningClass")
      (receiver, receiverTpeVars) <- owningClass match {
        case t @ p.Type.Struct(_, tpeVars, _) => (Some(p.Named("this", t)), tpeVars).success
        case x                                => s"Illegal receiver: $x".fail
      }

      log <- log.info("DefDef arguments", fnArgs.map((a, n) => s"$a(symbol=${a.symbol}) ~> $n")*)

      // Finally, we compile the def body like a closure or just return the term if we have one.
      ((rhsStmts, rhsDeps, rhsThisCls), log) <- fnRtnTerm match {
        case Some(t) =>
          ((p.Stmt.Return(p.Expr.Alias(t)) :: Nil, q.Dependencies(), None), log).success
        case None =>
          compileTerm(
            term = rhs,
            root = f.symbol,
            scope = Map.empty, // We pass an empty scope table as we are compiling an independent fn.
            intrinsify = intrinsify
          ).map { case ((stmts, _, deps, thisCls), rhsLog) => ((stmts, deps, thisCls), log + rhsLog) }
      }

      // The rhs terms *may* contain a `this` reference, we check that it matches the receiver we computed previously
      // if such a reference exists at all.
      _ <-
        if (rhsThisCls.exists { case (_, tpe) => tpe != owningClass })
          s"This type mismatch ($rhsThisCls != $owningClass)".fail
        else ().success

      // Make sure we record any class witnessed from the params and return type together with rhs.
      deps = rhsDeps.copy(classes = rhsDeps.classes |+| fnRtnWit |+| fnArgsWit)

      compiledFn = p.Function(
        name = p.Sym(f.symbol.fullName),
        tpeVars = receiverTpeVars ::: fnTypeVars,
        receiver = receiver,
        args = fnArgs.map(_._2),
        captures = deriveModuleStructCaptures(deps),
        rtn = fnRtnTpe,
        body = rhsStmts
      )

      log <- log.info("Result", compiledFn.repr)
      _ = println(log.render().mkString("\n"))

    } yield ((compiledFn, deps), log)

  def compileTerm(using q: Quoted)(
      term: q.Term,
      root: q.Symbol,
      scope: Map[q.Symbol, p.Term],
      intrinsify: Boolean = true
  ): Result[((List[p.Stmt], p.Type, q.Dependencies, Option[(q.ClassDef, p.Type.Struct)]), Log)] = for {
    log                 <- Log(s"Compile term: ${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}")
    log                 <- log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
    log                 <- log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))
    (termValue, c)      <- q.RemapContext(root = root, refs = scope).mapTerm(term)
    (_ -> termTpe, wit) <- Retyper.typer0(term.tpe)
    _ <-
      if (termTpe != termValue.tpe) {
        s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
      } else ().success
    statements = c.stmts :+ p.Stmt.Return(p.Expr.Alias(termValue))
    _          = println(s"Deps1 = ${c.deps.classes.values} ${c.deps.functions.values}")

    (optStmts, optDeps) =
      if (intrinsify) runLocalOptPass(statements, c.deps)
      else (statements, c.deps)
  } yield ((optStmts, termTpe, optDeps, c.thisCls), log)

  def compileExpr(using q: Quoted)(expr: Expr[Any]): Result[(List[(p.Named, q.Term)], p.Program, Log)] = for {
    log <- Log("Expr compiler")
    term = expr.asTerm
    // generate a name for the expr first
    exprName = s"${term.pos.sourceFile.name}:${term.pos.startLine}-${term.pos.endLine}"
    log <- log.info(s"Expr name: `$exprName`")

    // outline here
    (captures, wit, log) <- RefOutliner.outline(term)(log)
    (capturedNames, captureScope) = captures.foldMap[(List[(p.Named, q.Ref)], List[(q.Symbol, p.Term)])] {
      case (root, ref, value -> tpe) =>
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

    ((exprStmts, exprTpe, exprDeps, exprThisCls), termLog) <- compileTerm(
      term = term,
      root = q.Symbol.spliceOwner, // XXX not `spliceOwner` for now to avoid `this` captures
      scope = captureScope.toMap
    )

    log <- log ~+ termLog
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

    log <- log.info("This instance", exprThisCls.map((cls, tpe) => s"${cls.symbol} ~> ${tpe.repr}").toList*)

    // sort the captures so that the order is stable for codegen
    // captures = deps.vars.toList.sortBy(_._1.symbol)
    // captures = capturedNames.toList.distinctBy(_._2).sortBy((name, ref) => ref.symbol.pos.map(_.start) -> name.symbol)
    // log                 <- log.info(s"Expr+Deps Dependent vars   ", captures.map(_.toString)*)

    exprFn = p.Function(
      name = p.Sym(exprName),
      tpeVars = Nil,
      receiver = None,
      args = Nil,
      captures = capturedNames.map(_._1) ++
        deriveModuleStructCaptures(deps) ++
        exprThisCls.map((_, tpe) => p.Named("this", tpe)),
      rtn = exprTpe,
      body = exprStmts
    )

    (sdefs, sdefLog) <- deriveAllStructs(deps)(StdLib.StructDefs)
    log              <- log ~+ sdefLog

    captureNameToModuleRefTable = deps.modules.map { (symbol, struct) =>
      // It appears that `q.Ref.apply` prepends a `q.This(Some("this"))` at the root of the
      // select if the object is local to the scope regardless of the actual ownership type
      // (i.e. nested modules are not path-dependent, where as modules nested in classes are not),
      // not really sure why it does that but we need to remove it otherwise it's essentially a
      // syntax error at splice site.
      def removeThisFromRef(t: q.Symbol): q.Ref = q.Ref(t.companionModule) match {
        case select @ q.Select(root @ q.This(_), _) if t.companionModule.maybeOwner.flags.is(q.Flags.Module) =>
          removeThisFromRef(root.symbol).select(select.symbol)
        case x => x
      }
      struct.name.fqn.mkString("_") -> removeThisFromRef(symbol)
    } ++ exprThisCls.map((clsDef, _) => "this" -> q.This(clsDef.symbol)).toMap

    unopt = p.Program(exprFn, depFns, sdefs)
    log <- log.info(
      s"Program compiled (unpot), structures = ${unopt.defs.size}, functions = ${unopt.functions.size}"
    )

    unoptLog <- Log("Unopt")
    unoptLog <- unoptLog.info(s"Structures = ${unopt.defs.size}", unopt.defs.map(_.repr)*)
    unoptLog <- unoptLog.info(s"Functions  = ${unopt.functions.size}", unopt.functions.map(_.signatureRepr)*)
    unoptLog <- unoptLog.info(s"Entry", unopt.entry.repr)

    _ = println(log.render().mkString("\n"))

    // verify before optimisation
    (unoptVerification, unoptLog) <- VerifyPass.run(unopt)(unoptLog).success
    unoptLog <- unoptLog.info(
      s"Verifier",
      unoptVerification.map((f, xs) => s"${f.signatureRepr}\nError = ${xs.map("\t->" + _).mkString("\n")}")*
    )
    _ <-
      if (unoptVerification.exists(_._2.nonEmpty))
        s"Unopt validation failed (error=${unoptVerification.map(_._2.size).sum})\n${unoptVerification
          .map { case (f, es) =>
            s"${f.repr.linesIterator.map("\t|" + _).mkString("\n")}\n\t[Errors]\n${es.map("\t -> " + _).mkString("\n")}"
          }
          .mkString("\n")}".fail
      else ().success
    log <- log ~+ unoptLog

    _ = println(log.render().mkString("\n"))

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
        s"Opt validation failed (error=${optVerification.map(_._2.size).sum})\n${optVerification
          .map { case (f, es) =>
            s"${f.repr.linesIterator.map("\t|" + _).mkString("\n")}\n\t[Errors]\n${es.map("\t -> " + _).mkString("\n")}"
          }
          .mkString("\n")}".fail
      else ().success
    log <- log ~+ optLog

    capturedNameTable = capturedNames.map((name, ref) => name.symbol -> (name.tpe, ref)).toMap
    captured <- opt.entry.captures.traverse { n =>
      // Struct type of symbols may have been modified through specialisation so we just validate whether it's still a struct for now
      capturedNameTable.get(n.symbol) match {
        case None                                       => (n -> captureNameToModuleRefTable(n.symbol)).success
        case Some((tpe, ref)) if tpe.kind == n.tpe.kind => (n -> ref).success
        case Some((tpe, ref)) =>
          s"Unexpected type conversion, capture was ${tpe.repr} for $ref but function expected ${n.repr}".fail
      }
    }
    log <- log.info(s"Final captures", captured.map((name, ref) => s"${name.repr} = ${ref.show}")*)
    _ = println(log.render().mkString("\n"))

  } yield (captured, opt, log)

}
