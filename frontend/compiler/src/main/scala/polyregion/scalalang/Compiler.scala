package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.pass.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.prism.StdLib

import java.nio.file.Paths
import scala.quoted.Expr

object Compiler {

  import Remapper.*
  import Retyper.*

  private def runLocalOptPass(using Quoted) = IntrinsifyPass.intrinsify

  private val ProgramPasses: List[ProgramPass] = List(
    FnInlinePass, //
    VarReducePass,
    UnitExprElisionPass,   //
    DeadArgEliminationPass //
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
      deps: q.FnWitnesses
  )(
      fnLut: Map[p.Signature, (p.Function, Set[p.StructDef])] = Map.empty
  ): Result[(List[p.Function], Set[p.StructDef], Set[q.Symbol], Log)] = for {
    log <- Log(s"Compile ${deps.size} dependent function(s)")
    log <- deps.toList.foldLeftM(log) { case (log, (fn, ivks)) =>
      log.info(q.DefDef(fn.symbol, rhsFn = _ => None).show, ivks.map(_.repr).toList.sorted*)
    }
    (_ /*input*/, fns, clsDeps, moduleSymDeps, logs) <- (
      deps,                   // seed
      List.empty[p.Function], // acc
      Set.empty[p.StructDef], // acc
      Set.empty[q.Symbol],    // acc
      List.empty[Log]         // acc
    ).iterateWhileM { (remaining, fnAcc, clsDepAcc, moduleSymDepAcc, logAcc) =>
      remaining.toList
        .foldLeftM((fnAcc, Map.empty: q.FnWitnesses, clsDepAcc, moduleSymDepAcc, logAcc)) {
          case ((xs, depss, clsDepss, moduleSymDepss, logs), (defDef, ivks)) =>
            for {
              // XXX Generic specialisation on invoke is done in a pass later so we can ignore it for now
              _ <-
                if (ivks.map(_.name).size != 1)
                  s"Cannot collapse multiple invocations (${ivks.map(_.repr)}), term compiler may have miscompiled".fail
                else ().success
              target: p.Expr.Invoke = ivks.head

              fnApOfSameArityAndName = fnLut.collect {
                case (sig, x)
                    if sig.name.last == target.name.last &&      // match name first, most methods are eliminated here
                      sig.tpeVars.size == target.tpeArgs.size && // then we check type arity
                      sig.args.size == target.args.size          // then finally term arity
                    =>
                  val appliedTypes =
                    sig.tpeVars.zip(target.tpeArgs).map((name, tpe) => (p.Type.Var(name): p.Type) -> tpe).toMap
                  (sig.modifyAll[p.Type](_.mapLeaf(t => appliedTypes.getOrElse(t, t))): p.Signature) -> x
              }

              fnApOfMatchingTypes = fnApOfSameArityAndName.filter {
                case (sig, _)
                    if sig.rtn == target.rtn &&
                      sig.args == target.args.map(_.tpe) &&
                      sig.receiver == target.receiver.map(_.tpe) =>
                  true
                case _ => false
              }

              r <- fnApOfMatchingTypes.toList match {
                case Nil =>
                  // We found no replacement, log it and keep going.
                  for {
                    l                   <- Log(s"Compile (no replacement): ${target.repr}")
                    ((fn0, deps), log0) <- compileFn(defDef)
                    (fn1, wit0, clsDeps, moduleDeps, log1) <- compileAndReplaceStructDependencies(fn0, deps)(
                      StdLib.StructDefs
                    )
                  } yield (
                    fn1 :: xs,
                    depss ++ wit0,
                    clsDeps ++ clsDepss,
                    moduleDeps ++ moduleSymDepss,
                    (l + log0 + log1) :: logs
                  )
                case (_, (fn, clsDeps)) :: Nil =>
                  // We found exactly one function matching the invocation!
                  println(s"Replace: replace ${target.repr}")
                  for {
                    log <- Log(s"${target.repr}")
                    _   <- log.info("Callsites", ivks.map(_.repr).toList: _*)
                    _   <- log.info("Replacing with impl:", fn.repr)
                    _   <- log.info("Additional structs:", clsDeps.map(_.repr).toList*)

                    originalFnOwner = defDef.symbol.owner

                    originalFnOwnerStructDef <- structDef0(originalFnOwner)
                      .adaptError(e =>
                        new CompilerException(s"Cannot resolve struct def for owner of function ${defDef}", e)
                      )
                    // _ <- if (clsDeps != Set(originalFnOwnerStructDef)) s"Bad clsDep (${clsDeps.map(_.repr)}.contains(${originalFnOwnerStructDef.repr}) == false)".fail else ().success
                  } yield (
                    fn.copy(name = target.name) :: xs,
                    depss,
                    clsDeps ++ clsDepss,
                    moduleSymDepss,
                    log :: logs
                  )
                case xs =>
                  // We found multiple ambiguous replacements, signal error
                  s"Ambiguous replacement for ${target.repr}, the following replacements all match the signiture:\n${xs
                      .map("\t" + _._1.repr)
                      .mkString("\n")}".fail
              }
            } yield r
        }
        .map((xs, deps, clsDeps, moduleSymDeps, log) =>
          (
            deps,
            xs,
            clsDeps,
            moduleSymDeps,
            log
          )
        )

    }(_._1.nonEmpty)

  } yield (fns, clsDeps, moduleSymDeps, log + Log("Replacements", logs.toVector))

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

      allTypeVars = (receiverTpeVars ::: fnTypeVars).distinct

      log <- log.info("DefDef arguments", fnArgs.map((a, n) => s"$a(symbol=${a.symbol}) ~> $n")*)
      log <- log.info("DefDef tpe vars (recv+fn)", allTypeVars*)

      // Finally, we compile the def body like a closure or just return the term if we have one.
      ((rhsStmts, rhsDeps, rhsThisCls), log) <- fnRtnTerm match {
        case Some(t) =>
          ((p.Stmt.Return(p.Expr.Alias(t)) :: Nil, q.Dependencies(), None), log).success
        case None =>
          compileTerm(
            term = rhs,
            root = f.symbol,
            scope = Map.empty, // We pass an empty scope table as we are compiling an independent fn.
            tpeArgs = allTypeVars.map(p.Type.Var(_)),
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
        tpeVars = allTypeVars,
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
      tpeArgs: List[p.Type],
      intrinsify: Boolean = true
  ): Result[((List[p.Stmt], p.Type, q.Dependencies, Option[(q.ClassDef, p.Type.Struct)]), Log)] = for {
    log <- Log(s"Compile term: ${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}")
//    log <- log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
    log <- log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))

    // FILL TERM ARGS here

    _ = println(log.render().mkString("\n"))

    (termValue, c)      <- q.RemapContext(root = root, refs = scope).mapTerm(term, tpeArgs)
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

  def findMatchingClassInHierarchy[A, B](using q: Quoted)(symbol: q.Symbol, clsLut: Map[p.Sym, B]): Option[B] = {
    val hierarchy: List[p.Sym] = structName0(symbol) ::
      q.TypeIdent(symbol).tpe.baseClasses.map(structName0(_))
    hierarchy.collectFirst(Function.unlift(clsLut.get(_)))
  }

  def compileAndReplaceStructDependencies(using q: Quoted)(
      fn: p.Function,
      deps: q.Dependencies
  )(clsLut: Map[p.Sym, p.StructDef]): Result[(p.Function, q.FnWitnesses, Set[p.StructDef], Set[q.Symbol], Log)] =
    for {
      log <- Log(s"Compile dependent structs & apply replacements")
      (replacedClasses, classes) <- deps.classes.toList.partitionEitherM { case (clsDef, tpeAps) =>
        findMatchingClassInHierarchy(clsDef.symbol, clsLut) match {
          case Some(x) => Left(tpeAps -> x).success
          case None    => structDef0(clsDef.symbol).map(x => Right(tpeAps -> x))
        }
      }
      (replacedModules, modules) <- deps.modules.toList.partitionEitherM { case (symbol, tpe) =>
        findMatchingClassInHierarchy(symbol, clsLut) match {
          case Some(x) => Left(tpe -> x).success
          case None    => structDef0(symbol).map(x => Right(tpe -> x))
        }
      }

      log <- log.info("Replaced modules", replacedModules.map((ap, s) => s"${ap.repr} => ${s.repr}")*)
      log <- log.info("Replaced classes", replacedClasses.map((aps, s) => s"${aps.map(_.repr)} => ${s.repr}")*)
      log <- log.info("Reflected modules", modules.map((ap, s) => s"${ap.repr} => ${s.repr}")*)
      log <- log.info("Reflected classes", classes.map((aps, s) => s"${aps.map(_.repr)} => ${s.repr}")*)

      structDefs =
        (replacedClasses :::
          classes :::
          replacedModules :::
          modules).map(_._2).toSet

      typeLut: Map[p.Type.Struct, p.Type.Struct] =
        replacedClasses
          .flatMap((tpeAps, sdef) => tpeAps.map { case ap @ p.Type.Struct(_, _, ts) => ap -> sdef.tpe.copy(args = ts) })
          .toMap ++ replacedModules.map((tpe, sdef) => tpe -> sdef.tpe).toMap

      replaceTpe = (t: p.Type) =>
        t match {
          case s @ p.Type.Struct(_, _, _) => typeLut.getOrElse(s, s)
          case t                          => t
        }

      replaceTpeForTerm = (t: p.Term) =>
        t match {
          case s @ p.Term.Select(_, _) => s.modifyAll[p.Type](replaceTpe(_))
          case t                       => t
        }

      mappedFn = fn.modifyAll[p.Type](replaceTpe(_))
      mappedFnDeps = deps.functions.map((defdef, ivks) =>
        defdef -> ivks.map { case p.Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
          p.Expr.Invoke(
            name,
            tpeArgs.map(replaceTpe(_)),
            receiver.map(replaceTpeForTerm(_)),
            args.map(replaceTpeForTerm(_)),
            replaceTpe(rtn)
          ): p.Expr.Invoke
        }
      )

      log <- log.info("Type replacements", typeLut.map((a, b) => s"${a.repr} => ${b.repr}").toList.sorted*)
      log <- log.info("All struct defs", structDefs.toList.map(_.repr).sorted*)
      log <- log.info(
        "Dependent defs",
        mappedFnDeps.map((f, ivks) => s"${f.show} \ncallsites: \n${ivks.map(_.repr).mkString("\n")}").toList*
      )
      log <- log.info("Fn after replacements", mappedFn.repr)

      _ = println(log.render(1).mkString("\n"))

    } yield (mappedFn, mappedFnDeps, structDefs, deps.modules.keySet, log)

  def compileExpr(using q: Quoted)(expr: Expr[Any]): Result[
    (                                                     //
        List[(p.Named, q.Term)],                          //
        Map[p.Sym, polyregion.prism.Prism], //
        Map[p.Sym, p.Sym], //
        p.Program,                                        //
        Log                                               //
    )
  ] = for {
    log <- Log("Expr compiler")
    term = expr.asTerm
    // Generate a name for the expr first.
    exprName = s"${term.pos.sourceFile.name}:${term.pos.startLine}-${term.pos.endLine}"
    log <- log.info(s"Expr name: `$exprName`")

    // And then outline the term.
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

    // Then compile the term.
    ((exprStmts, exprTpe, exprDeps, exprThisCls), termLog) <- compileTerm(
      term = term,
      root = q.Symbol.spliceOwner,
      scope = captureScope.toMap,
      tpeArgs = Nil // TODO work out how to get the owner's tpe vars
    )

    exprFn = p.Function(
      name = p.Sym(exprName),
      tpeVars = Nil,
      receiver = None,
      args = Nil,
      captures = capturedNames.map(_._1) ++
        deriveModuleStructCaptures(exprDeps) ++
        exprThisCls.map((_, tpe) => p.Named("this", tpe)),
      rtn = exprTpe,
      body = exprStmts
    )

    log <- log ~+ termLog
    log <- log.info("Expr Fn", exprFn.repr)
    log <- log.info(
      "External captures",
      capturedNames.map((n, r) => s"$r(symbol=${r.symbol}, ${r.symbol.pos}) ~> ${n.repr}")*
    )
    log <- log.info(
      "Scope replacements",
      captureScope.map((sym, term) => s"$sym => ${term.repr}")*
    )
    log <- log.info(s"Expr dependent methods", exprDeps.functions.values.map(_.map(_.repr).toString).toList*)
    log <- log.info(s"Expr dependent structs", exprDeps.classes.values.map(_.map(_.repr).toString).toList*)

    // The methods in a prism will have the receiver set to the mirrored class.
    // So before we compile any dependent methods, we must first replace all struct types.
    // And since we are visiting all structures, we also compile the non-replaced ones as well.
    (rootFn, rootDependentFns, rootDependentStructs, rootDependentModuleSymbols, rootLog) <-
      compileAndReplaceStructDependencies(exprFn, exprDeps)(StdLib.StructDefs)
    log <- log ~+ rootLog

    _ = println(log.render().mkString("\n"))

    // We got a compiled fn now, now compile all dependencies.
    (allRootDependentFns, allRootDependentStructs, allRootDependentModuleSymbols, allRootLogs) <-
      compileAllDependencies(rootDependentFns)(StdLib.Functions)
    log <- log ~+ allRootLogs

    //    log <- log.info(s"Expr+Deps Dependent methods", deps.functions.values.map(_.toString).toList*)
//    log <- log.info(s"Expr+Deps Dependent structs", deps.classes.values.map(_.toString).toList*)
//    log <- log.info(s"Replacement dependent structs", clsDeps.map(_.repr).toList*)

    log <- log.info("This instance", exprThisCls.map((cls, tpe) => s"${cls.symbol} ~> ${tpe.repr}").toList*)

    // sort the captures so that the order is stable for codegen
    // captures = deps.vars.toList.sortBy(_._1.symbol)
    // captures = capturedNames.toList.distinctBy(_._2).sortBy((name, ref) => ref.symbol.pos.map(_.start) -> name.symbol)
    // log                 <- log.info(s"Expr+Deps Dependent vars   ", captures.map(_.toString)*)

//    (sdefs, sdefLog) <- deriveAllStructs(deps)(StdLib.StructDefs)
//    log              <- log ~+ sdefLog

    captureNameToModuleRefTable = (rootDependentModuleSymbols ++ allRootDependentModuleSymbols).map { symbol =>
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
      symbol.fullName.replace('.', '_') -> removeThisFromRef(symbol)
    }.toMap ++ exprThisCls.map((clsDef, _) => "this" -> q.This(clsDef.symbol)).toMap

    unopt = p.Program(rootFn, allRootDependentFns, (rootDependentStructs ++ allRootDependentStructs).toList)
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
    (opt, log)                              <- runProgramOptPasses(unopt)(log)
    (monomorphicToPolymorphicSym, opt, log) <- MonoStructPass(opt, log).success

    _ = println("Opt done")

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
    _ = println(optLog.render().mkString("\n"))
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

    log <- log.info(
      s"Final captures",
      captured.map((name, ref) => s"${name.repr} <== ${ref.symbol}: ${ref.tpe.widenTermRefByName.show}")*
    )
    log <- log.info(
      s"Final name references (monomorphic -> polymorphic)",
      monomorphicToPolymorphicSym.map((m, p) => s"$m => $p").toList*
    )

    _ = println(log.render().mkString("\n"))

    // List[(p.StructDef, Option[Prism])]
  } yield (
    captured,
    // For each def, we find the original name before monomorphic specialisation and then resolve the term mirrors
    opt.defs.flatMap { monomorphicDef =>
      val polymorphicSym = monomorphicToPolymorphicSym(monomorphicDef.name)
      StdLib.Mirrors.collect { case prism@(m, _) if m.source == polymorphicSym => monomorphicDef.name -> prism }
    }.toMap,
    monomorphicToPolymorphicSym,
    opt,
    log
  )

}
