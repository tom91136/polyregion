package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.pass.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.prism.StdLib

import java.nio.file.Paths
import scala.quoted.Expr
import scala.util.Try

object Compiler {

  import Remapper.*
  import Retyper.*

//   private val ProgramPasses: List[ProgramPass] = List(
//     printPass(IntrinsifyPass),
//     // printPass(DynamicDispatchPass),
//     printPass(SpecialisationPass),

// //    FnInlinePass,
//     VarReducePass,
//     UnitExprElisionPass,
//     DeadArgEliminationPass
//   )

//   private def runProgramOptPasses(program: p.Program, log: Log): Result[(p.Program)] =
//     scala.Function.chain(ProgramPasses.map(p => p(_, log)))(program).success

  // This derives the signature based on the *owning* symbol only!
  // This means, for cases where the method's direct root can be different, the actual owner(i.e. the context of the definition site) is used.
  def deriveSignature(using q: Quoted)(f: q.DefDef): Result[p.Signature] = for {
    (_ -> fnRtnTpe, _)      <- typer0(f.returnTpt.tpe) // Ignore the class witness as we're deriving the signature.
    (fnArgsTpeWithTerms, _) <- typer0N(f.paramss.flatMap(_.params).collect { case d: q.ValDef => d.tpt.tpe })
    owningClass             <- clsSymTyper0(f.symbol.owner)
    fnArgsTpes = fnArgsTpeWithTerms.map(_._2)
    fnTypeVars = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }
    (receiver, receiverTpeVars) <- owningClass match {
      case t @ p.Type.Struct(_, args) =>
        (Some(t), args.collect { case p.Type.Var(n) => n }).success
      case x => s"Illegal receiver: $x".fail
    }
    // TODO run outliner here
  } yield p.Signature(
    p.Sym(f.symbol.fullName),
    receiverTpeVars ::: fnTypeVars,
    receiver,
    fnArgsTpes,
    Nil /* TODO outline module captures */,
    Nil /* TODO outline term captures */,
    fnRtnTpe
  )

  private def matchingSignatures(sigs: Iterable[p.Signature], target: p.Expr.Invoke) =
    sigs
      .collect {
        case sig
            if sig.name.last == target.name.last &&      // match name first, most methods are eliminated here
              sig.tpeVars.size == target.tpeArgs.size && // then we check type arity
              sig.args.size == target.args.size          // then finally term arity
            =>
          val appliedTypes = // we then apply any type variables to form an applied signature
            sig.tpeVars.zip(target.tpeArgs).map((name, tpe) => (p.Type.Var(name): p.Type) -> tpe).toMap
          sig.modifyAll[p.Type](_.mapLeaf(t => appliedTypes.getOrElse(t, t))) -> sig
      }
      .collect {
        case (applied, sig) // finally we look for the matching signature
            if applied.rtn == target.rtn &&
              applied.args == target.args.map(_.tpe) &&
              applied.receiver == target.receiver.map(_.tpe) =>
          sig // and remember NOT to return the applied one
      }

  def compileAllDependencies(using q: Quoted)(
      sink: Log,
      deps: q.FnWitnesses,
      missingDefDefs: Map[q.Symbol, q.DefDef],
      existing: List[p.Function],
      renameMap: Map[q.Symbol, p.Sym] = Map.empty
  )(
      fnLut: Map[p.Signature, (p.Function, Set[p.StructDef])] = Map.empty
  ): Result[(List[p.Function], Set[p.StructDef], Set[q.Symbol])] = for {
    log <- sink.subLog(s"Compile ${deps.size} dependent function(s)").success
    _ = deps.foreach { (fn, ivks) =>
      log.info(q.DefDef(fn, rhsFn = _ => None).show, ivks.map(_.repr).toList.sorted*)
    }

    (_ /*input*/, fns, clsDeps, moduleSymDeps) <- (
      deps,                   // seed
      List.empty[p.Function], // acc
      Set.empty[p.StructDef], // acc
      Set.empty[q.Symbol]     // acc
    ).iterateWhileM { (remaining, fnAcc, clsDepAcc, moduleSymDepAcc) =>
      remaining.toList
        .foldLeftM((fnAcc, Map.empty: q.FnWitnesses, clsDepAcc, moduleSymDepAcc)) {
          case ((xs, depss, clsDepss, moduleSymDepss), (sym, ivks)) =>
            for {
              // XXX Generic specialisation on invoke is done in a pass later so we can ignore it for now
              _ <-
                if (ivks.map(_.name).size != 1)
                  s"Cannot collapse multiple invocations (${ivks.map(_.repr)}), term compiler may have miscompiled".fail
                else ().success
              target: p.Expr.Invoke = ivks.head

              // First, see if it's already in the pile the existing compiled/resolved functions (happens to local definitions)
              existingFns = existing.map(f => f.signature -> f).toMap
              r <- matchingSignatures(existingFns.keys, target).toList match {
                case _ :: Nil => // We have a perfect match from existing functions, no further actions needed
                  (xs, depss, clsDepss, moduleSymDepss).success
                case Nil => // Not there, we now look at actual dependencies
                  val defDef = sym.tree.asInstanceOf[q.DefDef]
                  // scala.Array's apply/update/length etc. have stub `throw new Error` bodies in the
                  // stdlib (the JVM lowers their call sites specially). The term compiler can't
                  // handle `throw`, but IntrinsifyPass handles xs.apply/update/length on Ptr<T>
                  // receivers directly at IR level. So just skip compiling these — they'll never
                  // be called as functions.
                  val isScalaArrayMethod = sym.maybeOwner.fullName == "scala.Array"
                  if (isScalaArrayMethod)
                    (xs, depss, clsDepss, moduleSymDepss).success
                  else {
                    // For overrides synthesized by depsWithOverrides, the target ivk's receiver is the
                    // base type (e.g. Buffer for an xs.apply call), not the override's owner (ListBuffer).
                    // If a mirror exists for the override's owner, the target's name/receiver won't match
                    // any fnLut key. As a fallback (only used when the primary signature match finds
                    // nothing), also match by the sym's own name when sym differs from the target's name.
                    val symFullName    = p.Sym(sym.fullName)
                    val primaryMatches = matchingSignatures(fnLut.keys, target).map(s => fnLut(s)).toList
                    val symNameMatch =
                      if (primaryMatches.nonEmpty || symFullName == target.name) Nil
                      else
                        fnLut.keys
                          .filter(s =>
                            s.name == symFullName &&
                              s.tpeVars.size == target.tpeArgs.size &&
                              s.args.size == target.args.size
                          )
                          .map(fnLut(_))
                          .toList
                    (primaryMatches ::: symNameMatch).distinctBy(_._1.name) match {
                      case Nil => // We found no replacement, log it and keep going.
                        val actualDefDef = missingDefDefs.getOrElse(sym, defDef)

                        if (actualDefDef.rhs.isEmpty) {
                          log.info(
                            s"No implementation: ${target.repr} (${actualDefDef.symbol.flags.show}); ${actualDefDef.symbol
                                .hashCode()}",
                            fnLut.keys.map(r => s"Candidate: ${r.repr}").toList*
                          )

                          // if (actualDefDef.symbol.flags.is(q.Flags.Abstract)) {

                          // } else {
                          (
                            xs,
                            depss,
                            clsDepss,
                            moduleSymDepss
                          ).success
                          // }

                        } else {
                          // see if function is already there

                          for {
                            log <- log
                              .subLog(s"Compile (no replacement): ${target.repr} (${actualDefDef.symbol.fullName})")
                              .success
                            (fn0Raw, deps) <- compileFn(log, actualDefDef, Map.empty)
                            // Apply rename for symbols that collide on fullName (e.g. multiple
                            // `_$$anon` classes from given declarations) so each compiled override
                            // gets a unique IR name.
                            fn0 = renameMap.get(actualDefDef.symbol) match {
                              case Some(unique) => fn0Raw.copy(name = unique)
                              case None         => fn0Raw
                            }
                            (fn1, wit0, clsDeps, moduleDeps) <- compileAndReplaceStructDependencies(log, fn0, deps)(
                              StdLib.StructDefs
                            )
                          } yield (
                            fn1 :: xs ::: deps.resolvedFunctions,
                            // Add this function's body-discovered deps to next iteration's queue.
                            // Avoid duplicates: skip if already in `remaining` (this-iteration queue)
                            // or `depss` (already-queued-for-next-iter). Importantly, do NOT filter
                            // against `deps.functions` itself — that's the SOURCE of wit0, so
                            // filtering against it would empty wit0 entirely and starve the loop.
                            depss ++ wit0.filterNot(x => remaining.contains(x._1) || depss.contains(x._1)),
                            clsDeps ++ clsDepss,
                            moduleDeps ++ moduleSymDepss
                          )
                        }
                      case (fn, clsDeps) :: Nil => // We found exactly one function matching the invocation
                        for {
                          log <- log.subLog(s"${target.repr}").success
                          _ = log.info("Callsites", ivks.map(_.repr).toList*)
                          _ = log.info("Replacing with impl:", fn.repr)
                          _ = log.info("Additional structs:", clsDeps.map(_.repr).toList*)

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
                          moduleSymDepss
                        )
                      case xs => // We found multiple ambiguous replacements, signal error
                        s"Ambiguous replacement for ${target.repr}, the following replacements all match the signature:\n${xs
                            .map("\t" + _._1.repr)
                            .mkString("\n")}".fail
                    }
                  }
                case xs =>
                  s"Ambiguous overload for ${target.repr}, the following existing resolved function all match the signature:\n${xs
                      .map("\t" + _._1.repr)
                      .mkString("\n")}".fail
              }

            } yield r
        }
        .map((xs, deps, clsDeps, moduleSymDeps) =>
          (
            deps,
            xs,
            clsDeps,
            moduleSymDeps
          )
        )

    }(_._1.nonEmpty)
  } yield (fns, clsDeps, moduleSymDeps)

  def compileFn(using q: Quoted) //
  (
      sink: Log,
      f: q.DefDef,
      scope: Map[q.Symbol, p.Expr] = Map.empty,
      intrinsify: Boolean = true
  ): Result[(p.Function, q.Dependencies)] =
    for {
      log <- sink.subLog(s"Compile DefDef: ${f.name}").success
      _ = log.info(s"Body", f.show(using q.Printer.TreeAnsiCode))
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

      owningSymbol <- Remapper
        .owningClassSymbol(f.symbol)
        .failIfEmpty(s"${f.symbol} does not have an owning class symbol")
      owningClass <- Retyper.clsSymTyper0(owningSymbol)
      _ = log.info(s"Method owner: $owningClass")
      (receiver, receiverTpeVars) <- owningClass match {
        case t @ p.Type.Struct(_, args) =>
          (Some(p.Named("this", t)), args.collect { case p.Type.Var(n) => n }).success
        case x => s"Illegal receiver: $x".fail
      }

      allTypeVars = (receiverTpeVars ::: fnTypeVars).distinct

      _ = log.info("DefDef arguments", fnArgs.map((a, n) => s"$a(symbol=${a.symbol}) ~> $n")*)
      _ = log.info("DefDef tpe vars (recv+fn)", allTypeVars*)

      // Finally, we compile the def body like a closure or just return the term if we have one.
      (rhsStmts, rhsDeps, rhsThisCls) <- fnRtnTerm match {
        case Some(t) =>
          ((p.Stmt.Return(t) :: Nil, q.Dependencies(), None)).success
        case None =>
          compileTerm(
            sink = log,
            term = rhs,
            root = f.symbol,
            scope = scope, // We pass an empty scope table as we are compiling an independent fn.
            tpeArgs = allTypeVars.map(p.Type.Var(_)),
            intrinsify = intrinsify
          ).map((stmts, _, deps, thisCls) => (stmts, deps, thisCls))
      }

      // The rhs terms *may* contain a `this` reference, we check that it matches the receiver we computed previously
      // if such a reference exists at all.
      _ <-
        if (rhsThisCls.exists { case (_, tpe) => tpe != owningClass })
          s"In `${f.show}` ,`this` type mismatch (${rhsThisCls.map(_._2.repr)}(rhs) != ${owningClass.repr}(owner))".fail
        else ().success

      // Make sure we record any class witnessed from the params and return type together with rhs.
      deps = rhsDeps.copy(classes = rhsDeps.classes |+| fnRtnWit |+| fnArgsWit)

      compiledFn = p.Function(
        name = p.Sym(f.symbol.fullName),
        tpeVars = allTypeVars,
        receiver = receiver.map(p.Arg(_)),
        args = fnArgs.map(_._2).map(p.Arg(_)),
        moduleCaptures = deriveModuleStructCaptures(deps).map(p.Arg(_)),
        termCaptures = Nil,
        rtn = fnRtnTpe,
        body = rhsStmts,
        visibility = p.Function.Visibility.Exported,
        fpMode = p.Function.FpMode.Relaxed,
        isEntry = false
      )
      _ = log.info("Result", compiledFn.repr)
    } yield (compiledFn, deps)

  def compileTerm(using q: Quoted)(
      sink: Log,
      term: q.Term,
      root: q.Symbol,
      scope: Map[q.Symbol, p.Expr],
      tpeArgs: List[p.Type],
      intrinsify: Boolean = true
  ): Result[(List[p.Stmt], p.Type, q.Dependencies, Option[(q.ClassDef, p.Type.Struct)])] = for {
    log <- sink
      .subLog(
        s"Compile term: ${scala.util.Try(s"${term.pos.sourceFile.name}:${term.pos.startLine}~${term.pos.endLine}").getOrElse("<no pos>")}"
      )
      .success
    _ = log.info("Body (AST)", pprint.tokenize(term, indent = 1, showFieldNames = true).mkString)
    _ = log.info("Body (Ascii)", term.show(using q.Printer.TreeAnsiCode))

    (termValue, c)      <- q.RemapContext(root = root, refs = scope).mapTerm(term, None, tpeArgs)
    (_ -> termTpe, wit) <- Retyper.typer0(term.tpe)
    _ <-
      if (termTpe != termValue.tpe) {
        s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
      } else ().success
    statements          = c.stmts :+ p.Stmt.Return(termValue)
    (optStmts, optDeps) = (statements, c.deps)
    // if (intrinsify) runLocalOptPass(statements, c.deps)
    // else (statements, c.deps)
  } yield (optStmts, termTpe, optDeps, c.thisCls)

  def findMatchingClassInHierarchy[A, B](using q: Quoted)(symbol: q.Symbol, clsLut: Map[p.Sym, B]): Option[B] = {
    val hierarchy: List[p.Sym] = structName0(symbol) ::
      q.TypeIdent(symbol).tpe.baseClasses.map(structName0(_))
    hierarchy.collectFirst(Function.unlift(clsLut.get(_)))
  }

  def compileAndReplaceStructDependencies(using q: Quoted)(
      sink: Log,
      fn: p.Function,
      deps: q.Dependencies
  )(clsLut: Map[p.Sym, p.StructDef]): Result[(p.Function, q.FnWitnesses, Set[p.StructDef], Set[q.Symbol])] =
    for {
      log <- sink.subLog(s"Compile dependent structs & apply replacements").success
      (replacedClasses, classes) <- deps.classes.toList.partitionEitherM { case (clsSym, tpeAps) =>
        findMatchingClassInHierarchy(clsSym, clsLut) match {
          case Some(x) => Left(tpeAps -> x).success
          case None    => structDef0(clsSym).map(x => Right(tpeAps -> x))
        }
      }
      (replacedModules, modules) <- deps.modules.toList.partitionEitherM { case (symbol, tpe) =>
        findMatchingClassInHierarchy(symbol, clsLut) match {
          case Some(x) => Left(tpe -> x).success
          case None    => structDef0(symbol).map(x => Right(tpe -> x))
        }
      }

      _ = log.info("Replaced modules", replacedModules.map((ap, s) => s"${ap.repr} => ${s.repr}")*)
      _ = log.info("Replaced classes", replacedClasses.map((aps, s) => s"${aps.map(_.repr)} => ${s.repr}")*)
      _ = log.info("Reflected modules", modules.map((ap, s) => s"${ap.repr} => ${s.repr}")*)
      _ = log.info("Reflected classes", classes.map((aps, s) => s"${aps.map(_.repr)} => ${s.repr}")*)

      structDefs =
        (replacedClasses :::
          classes :::
          replacedModules :::
          modules).map(_._2).toSet

      typeLut: Map[p.Type.Struct, p.Type.Struct] =
        replacedClasses
          .flatMap { case (tpeAps, sdef) =>
            tpeAps.map { case ap @ p.Type.Struct(_, ts) =>
              ap -> (p.Type.Struct(sdef.name, ts): p.Type.Struct)
            }
          }
          .toMap ++ replacedModules
            .map { case (tpe, sdef) => tpe -> (p.Type.Struct(sdef.name, Nil): p.Type.Struct) }
            .toMap

      replaceTpe = (t: p.Type) =>
        t match {
          case s @ p.Type.Struct(_, _) => typeLut.getOrElse(s, s)
          case t                       => t
        }

      replaceTpeForTerm = (t: p.Term) =>
        t match {
          case s @ p.Term.Select(_, _, _) => s.modifyAll[p.Type](replaceTpe(_))
          case t                          => t
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

      _ = log.info("Type replacements", typeLut.map((a, b) => s"${a.repr} => ${b.repr}").toList.sorted*)
      _ = log.info("All struct defs", structDefs.toList.map(_.repr).sorted*)
      _ = log.info(
        "Dependent defs",
        mappedFnDeps.map((f, ivks) => s"${f.fullName} \ncallsites: \n${ivks.map(_.repr).mkString("\n")}").toList*
      )
      _ = log.info("Fn after replacements", mappedFn.repr)

    } yield (mappedFn, mappedFnDeps, structDefs, deps.modules.keySet)

  def compileExpr(using q: Quoted)(sink: Log, expr: Expr[Any]): Result[
    (                                       //
        List[(p.Named, q.Term)],            //
        Map[p.Sym, polyregion.prism.Prism], //
        Map[p.Sym, p.Sym],                  //
        p.Program                           //
    )
  ] = for {
    log <- sink.subLog("Expr compiler").success
    term = expr.asTerm
    // Generate a name for the expr first.
    exprName = s"${term.pos.sourceFile.name}:${term.pos.startLine}-${term.pos.endLine}"
    _        = log.info(s"Expr name: `$exprName`")

    // And then outline the term.
    (captures, wit) <- RefOutliner.outline(log, term)
    (capturedNames, captureScope) = captures.foldMap[(List[(p.Named, q.Ref)], List[(q.Symbol, p.Expr)])] {
      case (root, ref, value -> tpe) =>
        (value, tpe) match {
          case (Some(x), _) => (Nil, (ref.symbol -> x) :: Nil)
          case (None, t) =>
            val name = ref match {
              case s @ q.Select(_, _) => s"_capture_${s.show}_${s.pos.startLine}_"
              case i @ q.Ident(_)     => s"_capture_${i.name}_${i.pos.startLine}_"
            }
            val named = p.Named(name, t)
            ((named -> ref) :: Nil, (ref.symbol -> p.Expr.Alias(p.Term.Select(named, Nil, t))) :: Nil)
        }
    }

    // Then compile the term.
    (exprStmts, exprTpe, exprDeps, exprThisCls) <- compileTerm(
      log,
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
      moduleCaptures = deriveModuleStructCaptures(exprDeps).map(p.Arg(_)),
      termCaptures = (capturedNames.map(_._1) ++
        exprThisCls.map((_, tpe) => p.Named("this", tpe))).map(p.Arg(_)),
      rtn = exprTpe,
      body = exprStmts,
      visibility = p.Function.Visibility.Exported,
      fpMode = p.Function.FpMode.Relaxed,
      isEntry = true
    )

    symbolToMacroDefinedDefDefs = q
      .collectTree[(q.Symbol, q.DefDef)](term) {
        case dd: q.DefDef =>
          dd.symbol.tree match {
            case q.DefDef(_, _, _, None) => (dd.symbol -> dd) :: Nil
            case _                       => Nil
          }
        case _ => Nil
      }
      .toMap

    // For defdef symbols defined inside the macro, symbol.tree will be EmptyTree, so we associate it up front.

    _ = log.info("Expr Fn", exprFn.repr)
    _ = log.info("symbolToMacroDefinedDefDefs", symbolToMacroDefinedDefDefs.map((k, v) => s"${k} => ${v.show}").toList*)
    _ = log.info(
      "External captures",
      capturedNames.map((n, r) => s"$r(symbol=${r.symbol}, ${r.symbol.pos}) ~> ${n.repr}")*
    )
    _ = log.info(
      "Scope replacements",
      captureScope.map((sym, term) => s"$sym => ${term.repr}")*
    )
    _ = log.info(
      s"Expr dependent methods (${exprDeps.functions.size})",
      exprDeps.functions.values.map(_.map(_.repr).toString).toList*
    )
    _ = log.info(
      s"Expr dependent structs (${exprDeps.classes.size})",
      exprDeps.classes.values.map(_.map(_.repr).toString).toList*
    )
    _ = log.info(
      s"Expr resolved functions (${exprDeps.resolvedFunctions.size})",
      exprDeps.resolvedFunctions.map(_.repr).toList*
    )

    // We mark all potentially required *concrete* virtual methods in the universe.
    // Also walk each captured class's tree for nested ClassDefs (e.g. anonymous classes from
    // `given Monoid[Int] = new Monoid { ... }`); they're the concrete dispatch targets for
    // abstract trait method calls and don't show up directly in exprDeps.classes since they're
    // only the runtime types of given fields, not statically-named call receivers.
    allClassSymbols = {
      val direct = exprDeps.classes.map(_._1).toList
      val nested = direct.flatMap(c =>
        Try(q.collectTree[q.Symbol](c.tree) {
          case cd: q.ClassDef if cd.symbol != c => List(cd.symbol)
          case _                                => Nil
        }).getOrElse(Nil)
      )
      (direct ++ nested).distinct
    }
    depsWithOverrides = exprDeps.functions.flatMap { (sym, ivks) =>
      val symOwner = sym.maybeOwner
      val overridesFromHierarchy = allClassSymbols
        .filter(c =>
          !symOwner.isNoSymbol && c.isClassDef &&
            Try(c.typeRef.baseClasses.contains(symOwner)).getOrElse(false)
        )
        .flatMap { c =>
          val viaOverridingSymbol = sym.overridingSymbol(c) match {
            case s if !s.isNoSymbol && s.isDefDef => Some(s)
            case _                                => None
          }
          val viaBodyWalk = Try(c.tree).toOption.collect { case cd: q.ClassDef => cd }.flatMap { cd =>
            cd.body.collectFirst { case dd: q.DefDef if dd.name == sym.name => dd.symbol }
          }
          viaOverridingSymbol.orElse(viaBodyWalk).toList
        }
        .map(_.tree)
        .collect { case x: q.DefDef => x.symbol }
        .filter(_ != sym)
      (sym -> ivks) :: overridesFromHierarchy.map(_ -> ivks)
    }.toMap

    // Anonymous-class symbols (e.g. multiple `given Monoid[X] = new Monoid {...}` in the same
    // owner) all collapse to the same `fullName` (e.g. `polyregion.GivenSuite._$$anon`). The IR
    // function name is built from that fullName, so distinct concrete impls collide. Synthesise
    // a unique IR name per (anon class symbol, abstract method) using the symbol's identity.
    syntheticOverrideNames: Map[q.Symbol, p.Sym] = {
      val anonClassSymbols = allClassSymbols.filter(c => c.fullName.contains("$$anon"))
      anonClassSymbols.flatMap { c =>
        val classSyntheticName = s"${c.fullName}_${System.identityHashCode(c).toHexString}"
        Try(c.tree).toOption.collect { case cd: q.ClassDef => cd }.toList.flatMap { cd =>
          cd.body.collect { case dd: q.DefDef =>
            dd.symbol -> p.Sym(s"${classSyntheticName}.${dd.name}")
          }
        }
      }.toMap
    }

    // We also mark all parents of those virtual methods, these are abstract
    abstractDefs = exprDeps.functions
      .flatMap { (sym, ivks) =>
        allClassSymbols
          .map(sym.overridingSymbol(_))
          .filter(s => s.isDefDef)
          .map(_.tree)
          .collect { case x: q.DefDef => x }
          .filter(_.symbol.flags.is(q.Flags.Deferred))
      }
      .toList
      .distinctBy(_.symbol)

    abstractFnsWithAssertBody <- abstractDefs.traverse(deriveSignature(_).map { sig =>
      val fn = p.Function(
        name = sig.name,
        tpeVars = sig.tpeVars,
        receiver = sig.receiver.map(p.Named("this", _)).map(p.Arg(_)),
        args = sig.args.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
        moduleCaptures = sig.moduleCaptures.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
        termCaptures = sig.termCaptures.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
        rtn = sig.rtn,
        // Trait/abstract method body: synthesise a single Return of poison so the function is
        // well-formed. Actual implementation is dispatched dynamically; this body only matters
        // for the verifier (requires at least one statement and a Return).
        body = p.Stmt.Return(p.Expr.Alias(p.Term.Poison(sig.rtn))) :: Nil,
        visibility = p.Function.Visibility.Exported,
        fpMode = p.Function.FpMode.Relaxed,
        isEntry = false
      )
      log.info(s"Abstract (trait function)", sig.repr)
      fn
    })

    exprDeps <- exprDeps
      .copy(functions = depsWithOverrides)
      .success

    // The methods in a prism will have the receiver set to the mirrored class.
    // So before we compile any dependent methods, we must first replace all struct types.
    // And since we are visiting all structures, we also compile the non-replaced ones as well.
    (rootFn, rootDependentFns, rootDependentStructs, rootDependentModuleSymbols) <- compileAndReplaceStructDependencies(
      log,
      exprFn,
      exprDeps
    )(StdLib.StructDefs)

    // We got a compiled fn now, now compile all dependencies. Pass the synthetic-name rename
    // map so anonymous-class overrides get distinct IR names per (class symbol, method).
    (allRootDependentFns, allRootDependentStructs, allRootDependentModuleSymbols) <-
      compileAllDependencies(
        log,
        rootDependentFns,
        symbolToMacroDefinedDefDefs,
        exprDeps.resolvedFunctions,
        syntheticOverrideNames
      )(StdLib.Functions)

    _ = log.info("A ", rootDependentFns.map(x => x.toString).toList*)

    //    log <- log.info(s"Expr+Deps Dependent methods", deps.functions.values.map(_.toString).toList*)
//    log <- log.info(s"Expr+Deps Dependent structs", deps.classes.values.map(_.toString).toList*)
//    log <- log.info(s"Replacement dependent structs", clsDeps.map(_.repr).toList*)

    _ = log.info("This instance", exprThisCls.map((cls, tpe) => s"${cls.symbol} ~> ${tpe.repr}").toList*)

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

    // Build a dispatch table for abstract trait methods. For each (abstract method, concrete
    // class extending its declarer with applied type args), map (abstract method name, parent
    // applied as IR struct) → (concrete method name, concrete class struct). At IR rewrite
    // time this lets us redirect `monoid.mempty()` (where monoid: Monoid[Int]) to
    // `anon$1.mempty(monoid.cast[anon$1])`, treating the unique anonymous-class override as
    // the static dispatch target.
    dispatchTable: Map[(p.Sym, p.Type.Struct), (p.Sym, p.Type.Struct)] = {
      val entries = abstractDefs.flatMap { abstractDefDef =>
        val abstractOwnerSym = abstractDefDef.symbol.maybeOwner
        if (abstractOwnerSym.isNoSymbol) Nil
        else
          allClassSymbols.flatMap { classSym =>
            val baseClasses = Try(classSym.typeRef.baseClasses).getOrElse(Nil)
            val isSubclass = !classSym.isNoSymbol && classSym.isClassDef && classSym != abstractOwnerSym &&
              baseClasses.contains(abstractOwnerSym)
            if (!isSubclass) Nil
            else {
              val overrideSymOpt = Try(classSym.tree).toOption.collect { case cd: q.ClassDef => cd } match {
                case Some(cd) =>
                  cd.body.collectFirst {
                    case dd: q.DefDef if dd.name == abstractDefDef.name => dd.symbol
                  }
                case None => None
              }
              overrideSymOpt match {
                case Some(overrideSym) if !overrideSym.isNoSymbol && overrideSym != abstractDefDef.symbol =>
                  val parentRepr = classSym.typeRef.baseType(abstractOwnerSym)
                  val parentTry = Retyper.typer0(parentRepr).toOption.flatMap {
                    case ((_, s: p.Type.Struct), _) => Some(s)
                    case _                          => None
                  }
                  val classTry = Retyper.typer0(classSym.typeRef).toOption.flatMap {
                    case ((_, s: p.Type.Struct), _) => Some(s)
                    case _                          => None
                  }
                  (parentTry, classTry) match {
                    case (Some(parentStruct), Some(classStruct)) =>
                      // If the override symbol got a synthetic rename (because its class fullName
                      // collides with sibling anon classes), use the synthetic name as the dispatch
                      // target so it matches the renamed compiled function.
                      val concreteName =
                        syntheticOverrideNames.getOrElse(overrideSym, p.Sym(overrideSym.fullName))
                      val key: (p.Sym, p.Type.Struct) =
                        (p.Sym(abstractDefDef.symbol.fullName), parentStruct)
                      val value: (p.Sym, p.Type.Struct) = (concreteName, classStruct)
                      List(key -> value)
                    case _ => Nil
                  }
                case _ => Nil
              }
            }
          }
      }
      entries.toMap
    }
    _ = log.info(
      s"Dispatch table",
      dispatchTable.toList.map { case ((abs, recv), (concrete, cls)) =>
        s"${abs.repr} on ${recv.repr} -> ${concrete.repr} on ${cls.repr}"
      }*
    )

    // Apply dispatch: walk every Invoke and redirect abstract calls to their concrete
    // implementations, inserting an upcast on the receiver so the concrete fn's signature
    // accepts it. The verifier accepts struct downcasts when the source has the target as
    // a parent (anon$1 has Monoid as a parent).
    rewriteDispatch = (fn: p.Function) => {
      // Walk the body and rewrite Stmt.Var/Stmt.Return that wrap a dispatch-eligible Invoke. We
      // bind the upcast receiver to a fresh Stmt.Var (Cast is compound, so it cannot live in a
      // Term position) and rebind the Invoke to use the cast result.
      var counter = 0
      def freshName(tpe: p.Type): p.Named = { counter += 1; p.Named(s"_dispatch_recv_${counter}", tpe) }
      def patchInvoke(ivk: p.Expr.Invoke, prelude: scala.collection.mutable.ListBuffer[p.Stmt]): p.Expr.Invoke =
        ivk.receiver.map(_.tpe) match {
          case Some(s: p.Type.Struct) =>
            dispatchTable.get((ivk.name, s)) match {
              case Some((concreteName, classStruct)) =>
                ivk.receiver match {
                  case Some(r) =>
                    val tmp = freshName(classStruct)
                    prelude += p.Stmt.Var(tmp, Some(p.Expr.Cast(r, classStruct)), isMutable = false)
                    val tmpSel: p.Term.Select = p.Term.Select(tmp, Nil, classStruct)
                    ivk.copy(name = concreteName, receiver = Some(tmpSel))
                  case None => ivk.copy(name = concreteName)
                }
              case None => ivk
            }
          case _ => ivk
        }
      def walk(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
        case p.Stmt.Var(n, Some(ivk: p.Expr.Invoke), m) =>
          val prelude = scala.collection.mutable.ListBuffer.empty[p.Stmt]
          val newIvk = patchInvoke(ivk, prelude)
          prelude.toList :+ p.Stmt.Var(n, Some(newIvk), m)
        case p.Stmt.Return(ivk: p.Expr.Invoke) =>
          val prelude = scala.collection.mutable.ListBuffer.empty[p.Stmt]
          val newIvk = patchInvoke(ivk, prelude)
          prelude.toList :+ p.Stmt.Return(newIvk)
        case p.Stmt.Cond(c, t, f) => p.Stmt.Cond(c, walk(t), walk(f)) :: Nil
        case p.Stmt.While(c, b)   => p.Stmt.While(c, walk(b)) :: Nil
        case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, walk(b)) :: Nil
        case other                => other :: Nil
      }
      fn.copy(body = walk(fn.body))
    }

    rootFn0              = rewriteDispatch(rootFn)
    allRootDependentFns0 = allRootDependentFns.map(rewriteDispatch)
    resolvedFunctions0   = exprDeps.resolvedFunctions.map(rewriteDispatch)

    // The dispatch rewrite introduces Casts to anonymous-class structs that weren't part of
    // the original struct deps (the anon classes are only the runtime types of given fields,
    // not directly captured types). Synthesise their StructDefs (empty bodies, parents from
    // the class hierarchy) so the LLVM backend can resolve the type at the Cast site.
    dispatchAnonStructDefs <- dispatchTable.values
      .map(_._2)
      .toList
      .distinct
      .traverse { classStruct =>
        // We can't go from a Type.Struct back to a q.Symbol cleanly, so synthesise the StructDef
        // directly from the Type.Struct. Anon classes used in dispatch are opaque (no fields),
        // matching how Monoid is handled.
        p.StructDef(classStruct.name, Nil, Nil, Nil).success
      }

    unopt = p.Program(
      rootFn0,
      abstractFnsWithAssertBody ::: allRootDependentFns0 ::: resolvedFunctions0,
      ((rootDependentStructs ++ allRootDependentStructs).toList ++ dispatchAnonStructDefs).distinct
    )
    _ = log.info(
      s"Program compiled (unpot), structures = ${unopt.defs.size}, functions = ${unopt.functions.size}"
    )

    unoptLog <- log.subLog("Unopt").success
    _ = unoptLog.info(s"Structures = ${unopt.defs.size}", unopt.defs.map(_.repr)*)
    _ = unoptLog.info(s"Functions  = ${unopt.functions.size}", unopt.functions.map(_.repr)*)
    _ = unoptLog.info(s"Entry", unopt.entry.repr)

    // verify before optimisation
    unoptVerification <- VerifyPass(unopt, unoptLog, verifyFunction = false).success
    _ = unoptLog.info(
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

    // run the global optimiser

    optPassLog = log.subLog("Opt Passes")

    opt = scala.Function.chain(
      List[ProgramPass](
        printPass(IntrinsifyPass),
        printPass(SpecialisationPass),
        ConstantFoldPass,
        VarReducePass,
        UnitExprElisionPass,
        DeadArgEliminationPass
      ).map(p => p(_, optPassLog))
    )(unopt)
    (monomorphicToPolymorphicSym, opt) <- MonoStructPass(opt, log).success

    // Wire up Invoke captures and propagate captures transitively up to the entry. compileFn
    // doesn't know what captures a function will need until its body is compiled (in a separate
    // RemapContext from the entry/caller). After opt passes finalise per-function capture lists,
    // walk the program: ensure every Invoke passes the captures its callee needs, and that the
    // entry's own captures cover the transitive closure of all module captures any reachable
    // function needs.
    opt <- {
      val fnByName: Map[p.Sym, p.Function] = (opt.entry :: opt.functions).map(fn => fn.name -> fn).toMap
      // Transitive module-capture types each function needs, including those from its callees.
      // The call graph is invariant across iterations: only the per-fn capture sets grow;
      // we collect callees once up front rather than re-walking every body each iteration.
      def computeTransitiveModuleCaps(): Map[p.Sym, List[p.Arg]] = {
        val callGraph: Map[p.Sym, List[p.Sym]] =
          fnByName.view.mapValues(_.collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk.name }).toMap
        var current: Map[p.Sym, List[p.Arg]] = fnByName.view.mapValues(_.moduleCaptures).toMap
        var changed                          = true
        while (changed) {
          changed = false
          val next = current.map { case (sym, caps) =>
            val seen = scala.collection.mutable.Set.from(caps.map(_.named.tpe))
            val merged = callGraph(sym).foldLeft(caps) { (acc, callee) =>
              current.getOrElse(callee, Nil).foldLeft(acc) { (acc1, cap) =>
                if (seen.add(cap.named.tpe)) acc1 :+ cap else acc1
              }
            }
            sym -> merged
          }
          if (next != current) { changed = true; current = next }
        }
        current
      }
      val transitive   = computeTransitiveModuleCaps()
      val entryAllCaps = transitive.getOrElse(opt.entry.name, opt.entry.moduleCaptures)
      val newEntry     = opt.entry.copy(moduleCaptures = entryAllCaps)
      val entryCapByTpe: Map[p.Type, p.Named] =
        (newEntry.moduleCaptures ++ newEntry.termCaptures).map(arg => arg.named.tpe -> arg.named).toMap

      // Update each function's own moduleCaptures to its transitive set, so when its body
      // forwards a capture into a callee Invoke, the capture name is in scope.
      def updateFnCaps(fn: p.Function): p.Function = {
        val transitiveCaps = transitive.getOrElse(fn.name, fn.moduleCaptures)
        if (transitiveCaps == fn.moduleCaptures) fn
        else fn.copy(moduleCaptures = transitiveCaps)
      }

      // Rebuild the LUT from post-updateFnCaps functions so callee.moduleCaptures reflects
      // the transitive set the backend will actually register.
      val updatedFns                                = newEntry :: opt.functions.map(updateFnCaps)
      val updatedFnByName: Map[p.Sym, p.Function] = updatedFns.map(fn => fn.name -> fn).toMap

      def patchFn(fn: p.Function): p.Function = {
        val ownCapByTpe: Map[p.Type, p.Named] =
          (fn.moduleCaptures ++ fn.termCaptures).map(arg => arg.named.tpe -> arg.named).toMap
        def patchIvk(ivk: p.Expr.Invoke): p.Expr.Invoke = updatedFnByName.get(ivk.name) match {
          case None => ivk
          case Some(c) =>
            val expected = c.args.size + c.moduleCaptures.size + c.termCaptures.size
            if (ivk.args.size >= expected) ivk
            else {
              val needed = c.moduleCaptures ++ c.termCaptures
              if (needed.isEmpty) ivk
              else
                ivk.copy(args = needed.map { arg =>
                  val tpe   = arg.named.tpe
                  val named = ownCapByTpe.get(tpe).orElse(entryCapByTpe.get(tpe)).getOrElse(arg.named)
                  selectTerm(Nil, named)
                } ::: ivk.args)
            }
        }
        fn.modifyAll[p.Expr] {
          case ivk: p.Expr.Invoke => patchIvk(ivk)
          case x                  => x
        }
      }
      opt
        .copy(
          entry = patchFn(newEntry),
          functions = updatedFns.tail.map(patchFn)
        )
        .success
    }

    // verify again after optimisation
    optLog = log.subLog("Opt")
    optVerification <- VerifyPass(opt, optLog, verifyFunction = true).success

    _ = optLog.info(s"Structures = ${opt.defs.size}", opt.defs.map(_.repr)*)
    _ = optLog.info(s"Functions  = ${opt.functions.size}", opt.functions.map(_.repr)*)
    _ = optLog.info(s"Entry", opt.entry.repr)

    _ = optLog.info(
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

    capturedNameTable = capturedNames.map((name, ref) => name.symbol -> (name.tpe, ref)).toMap
    captured <- (opt.entry.moduleCaptures ++ opt.entry.termCaptures).traverse { n =>
      // Struct type of symbols may have been modified through specialisation so we just validate whether it's still a struct for now
      capturedNameTable.get(n.named.symbol) match {
        case None => (n -> captureNameToModuleRefTable(n.named.symbol)).success
        case Some((tpe, ref)) if tpe.kind == n.named.tpe.kind => (n -> ref).success
        case Some((tpe, ref)) =>
          s"Unexpected type conversion, capture was ${tpe.repr} for $ref but function expected ${n.repr}".fail
      }
    }

    _ = log.info(
      s"Final captures",
      captured.map((name, ref) => s"${name.repr} <== ${ref.symbol}: ${ref.tpe.widenTermRefByName.show}")*
    )
    _ = log.info(
      s"Final name references (monomorphic -> polymorphic)",
      monomorphicToPolymorphicSym.map((m, p) => s"$m => $p").toList*
    )

    // List[(StructDef, Option[Prism])]
  } yield (
    captured.map((arg, term) => arg.named -> term),
    // For each def, we find the original name before monomorphic specialisation and then resolve the term mirrors.
    // Some defs (LLVM-side opaque placeholders, abstract typeclasses) have no entry in the
    // monomorphic-to-polymorphic table — those have no prism, so just skip them.
    opt.defs.flatMap { monomorphicDef =>
      monomorphicToPolymorphicSym.get(monomorphicDef.name) match {
        case Some(polymorphicSym) =>
          StdLib.Mirrors.collect { case prism @ (m, _) if m.source == polymorphicSym => monomorphicDef.name -> prism }
        case None => Nil
      }
    }.toMap,
    monomorphicToPolymorphicSym,
    opt
  )

}
