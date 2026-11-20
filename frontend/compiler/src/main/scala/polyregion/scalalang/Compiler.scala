package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.pass.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.intrinsic.IntrinsicTable
import polyregion.javalang.{BytecodeResolver, Natives}
import polyregion.prism.{Prism, PrismTables}

import java.nio.file.Paths
import scala.annotation.tailrec
import scala.collection.mutable
import scala.quoted.Expr
import scala.util.Try

object Compiler {

  import Remapper.*
  import Retyper.*

  private[scalalang] def interfaceIdentity(using q: Quoted)(symbol: q.Symbol): Option[(String, String)] = {
    import q.underlying.reflect.*
    symbol.annotations.collectFirst(Function.unlift { annotation =>
      val annotationType = annotation.tpe
      if (annotationType.typeSymbol.fullName != "scala.annotation.compileTimeOnly") None
      else {
        def arguments(term: Term): List[Term] = term match {
          case Apply(_, args)           => args
          case Inlined(_, _, expansion) => arguments(expansion)
          case Typed(inner, _)          => arguments(inner)
          case _                        => Nil
        }
        arguments(annotation).collectFirst { case Literal(StringConstant(value)) => value }.flatMap { value =>
          value.stripPrefix("polyregion_interface:") match {
            case unchanged if unchanged == value => None
            case identity =>
              identity.split(":", 2).toList match {
                case packageName :: declaration :: Nil if packageName.nonEmpty && declaration.nonEmpty =>
                  Some(packageName -> declaration)
                case _ => None
              }
          }
        }
      }
    })
  }

  private[scalalang] def failIfVerificationErrors(
      label: String,
      verification: List[(p.Function, List[String])]
  ): Result[Unit] =
    if (verification.exists(_._2.nonEmpty))
      s"$label validation failed (error=${verification.map(_._2.size).sum})\n${verification
          .map { case (f, es) =>
            s"${f.repr.linesIterator.map("\t|" + _).mkString("\n")}\n\t[Errors]\n${es.map("\t -> " + _).mkString("\n")}"
          }
          .mkString("\n")}".fail
    else ().success

  // This derives the signature based on the *owning* symbol only!
  // This means, for cases where the method's direct root can be different, the actual owner(i.e. the context of the definition site) is used.
  private def logVerifier(log: Log, verification: List[(p.Function, List[String])]): Unit =
    log.info(
      "Verifier",
      verification.map((f, xs) => s"${f.signatureKey}\nError = ${xs.map("\t->" + _).mkString("\n")}")*
    )

  def deriveSignature(using q: Quoted)(f: q.DefDef): Result[p.Signature] = for {
    (_ -> fnRtnTpe, _) <- typer0(f.returnTpt.tpe) // Ignore the class witness as we're deriving the signature.
    params = f.paramss.flatMap(_.params)
    symbol = f.symbol
    (fnArgsTpeWithTerms, _) <- typer0N(params.collect { case d: q.ValDef => d.tpt.tpe })
    owningClass             <- clsSymTyper0(symbol.owner)
    callableVars = params.collect { case d: q.ValDef => Retyper.callableVar(d.symbol) }
    fnArgsTpes   = fnArgsTpeWithTerms.map(_._2).zip(callableVars).map((tpe, callable) => callable.getOrElse(tpe))
    fnTypeVars: List[p.Type.Var] =
      params.collect { case q.TypeDef(name, _) => p.Type.Var(name) } ::: callableVars.flatten
    (receiver, receiverTpeVars) <- owningClass match {
      case t @ p.Type.Struct(_, args) =>
        (Some(t), args.collect { case p.Type.Var(name, size) => p.Type.Var(name, size) }: List[p.Type.Var]).success
      case x => s"Illegal receiver: $x".fail
    }
    // TODO run outliner here
  } yield p.Signature(
    p.Sym(symbol.fullName),
    receiverTpeVars ::: fnTypeVars,
    receiver,
    fnArgsTpes,
    Nil /* TODO outline module captures */,
    Nil /* TODO outline term captures */,
    fnRtnTpe
  )

  private def matchingSignatures(sigs: Iterable[p.Signature], target: p.Expr.Invoke) = {
    val tpeArgCount = target.tpeArgs.size
    val argCount    = target.args.size
    val argTpes     = target.args.map(_.tpe)
    sigs
      .collect {
        case sig
            if sig.name.last == target.calleeName.last && // match name first, most methods are eliminated here
              sig.tpeVars.size == tpeArgCount &&          // then we check type arity
              sig.args.size == argCount                   // then finally term arity
            =>
          val appliedTypes = // we then apply any type variables to form an applied signature
            sig.tpeVars.zip(target.tpeArgs).map((variable, tpe) => (variable: p.Type) -> tpe).toMap
          sig.modifyAll[p.Type](_.mapLeaf(t => appliedTypes.getOrElse(t, t))) -> sig
      }
      .collect {
        case (applied, sig) // finally we look for the matching signature
            if applied.rtn == target.rtn &&
              applied.args == argTpes &&
              applied.receiver == target.receiver.map(_.tpe) =>
          sig // and remember NOT to return the applied one
      }
  }

  // a Java static method has no TASTy, so the bytecode tier lowers it from its class file unless an intrinsic covers it
  private def bytecodeFallback(using
      q: Quoted
  )(
      sym: q.Symbol,
      defDef: q.DefDef,
      target: p.Expr.Invoke
  ): Option[Result[List[p.Function]]] = {
    val flags = sym.flags
    Option
      .when(flags.is(q.Flags.JavaDefined) && flags.is(q.Flags.JavaStatic))(deriveSignature(defDef).toOption)
      .flatten
      .filter(sig => (sig.rtn :: sig.args).forall(BytecodeResolver.lowerable))
      .map { signature =>
        val bytes = BytecodeResolver.loaderClasses(getClass.getClassLoader, ClassLoader.getSystemClassLoader)
        for {
          owner <- BytecodeResolver.locate(sym.owner.fullName.stripSuffix("$"), bytes)
          ref   <- BytecodeResolver.descriptorFor(owner, sym.name, signature.args, signature.rtn, bytes)
          lowered <-
            if (Natives.covered(ref)) Nil.success
            else BytecodeResolver.resolve(List(ref), bytes)
        } yield lowered match {
          case plain :: _ if plain.name != target.calleeName =>
            BytecodeResolver.callableAs(plain, target.calleeName, None) :: lowered
          case _ => lowered
        }
      }
  }

  def compileAllDependencies(using
      q: Quoted
  )(
      sink: Log,
      deps: q.FnWitnesses,
      missingDefDefs: Map[q.Symbol, q.DefDef],
      existing: List[p.Function],
      renameMap: Map[q.Symbol, p.Sym] = Map.empty,
      callers: List[p.Function] = Nil,
      config: LaunchConfig = LaunchConfig.Default,
      target: String
  )(
      fnLut: Map[p.Signature, (p.Function, Set[p.StructDef])],
      structDefs: Map[p.Sym, p.StructDef]
  ): Result[(List[p.Function], Set[p.StructDef], Set[q.Symbol], Boolean)] = {
    val interfaceImports = List.newBuilder[InterfaceResolver.Import]
    for {
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
          .flatMap((sym, ivks) => ivks.groupBy(_.receiver.map(_.tpe)).values.map(group => sym -> Set(group.head)))
          .foldLeftM((fnAcc, Map.empty: q.FnWitnesses, clsDepAcc, moduleSymDepAcc)) {
            case ((xs, depss, clsDepss, moduleSymDepss), (sym, ivks)) =>
              for {
                // XXX Generic specialisation on invoke is done in a pass later so we can ignore it for now
                _ <-
                  if (ivks.map(_.callee).size != 1)
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
                    // handle `throw`, but Intrinsify handles xs.apply/update/length on Ptr<T>
                    // receivers directly at IR level. So just skip compiling these - they'll never
                    // be called as functions.
                    // intrinsics stubs are likewise replaced by Intrinsify at their call sites.
                    val isScalaArrayMethod = sym.maybeOwner.fullName == "scala.Array"
                    val isIntrinsicStub = sym.maybeOwner.fullName.stripSuffix("$") == "polyregion.scalalang.intrinsics"
                    if (isScalaArrayMethod || isIntrinsicStub)
                      (xs, depss, clsDepss, moduleSymDepss).success
                    else {
                      // For overrides synthesized by depsWithOverrides, the target ivk's receiver is the
                      // base type (e.g. Buffer for an xs.apply call), not the override's owner (ListBuffer).
                      // If a mirror exists for the override's owner, the target's name/receiver won't match
                      // any fnLut key. As a fallback (only used when the primary signature match finds
                      // nothing), also match by the sym's own name when sym differs from the target's name.
                      val symFullName    = p.Sym(sym.fullName)
                      val primaryMatches = matchingSignatures(fnLut.keys, target).map(s => fnLut(s)).toList
                      val tpeArgCount    = target.tpeArgs.size
                      val argCount       = target.args.size
                      val symNameMatch =
                        if (primaryMatches.nonEmpty || symFullName == target.calleeName) Nil
                        else
                          fnLut.keys
                            .filter(s =>
                              s.name == symFullName &&
                                s.tpeVars.size == tpeArgCount &&
                                s.args.size == argCount
                            )
                            .map(fnLut(_))
                            .toList
                      (primaryMatches ::: symNameMatch).distinctBy(_._1.name) match {
                        case Nil => // We found no replacement, log it and keep going.
                          val actualDefDef = missingDefDefs.getOrElse(sym, defDef)
                          val actualSym    = actualDefDef.symbol

                          interfaceIdentity(actualSym) match {
                            case Some((packageName, declaration)) =>
                              interfaceImports += InterfaceResolver.Import(
                                packageName,
                                declaration,
                                target.copy(receiver = None)
                              )
                              (xs, depss, clsDepss, moduleSymDepss).success
                            case None if actualDefDef.rhs.isEmpty =>
                              bytecodeFallback(actualSym, actualDefDef, target) match {
                                case Some(lowered) =>
                                  lowered.map(fs => (fs ::: xs, depss, clsDepss, moduleSymDepss))
                                case None =>
                                  log.info(
                                    s"No implementation: ${target.repr} (${actualSym.flags.show}); ${actualSym.hashCode()}",
                                    fnLut.keys.map(r => s"Candidate: ${r.repr}").toList*
                                  )
                                  (xs, depss, clsDepss, moduleSymDepss).success
                              }

                            case None =>
                              // see if function is already there

                              for {
                                log <- log
                                  .subLog(s"Compile (no replacement): ${target.repr} (${actualSym.fullName})")
                                  .success
                                (fn0Raw, deps) <- compileFn(log, actualDefDef, Map.empty)
                                // Apply rename for symbols that collide on fullName (e.g. multiple
                                // `_$$anon` classes from given declarations) so each compiled override
                                // gets a unique IR name.
                                fn0 = renameMap.get(actualSym) match {
                                  case Some(unique) => fn0Raw.copy(decl = fn0Raw.decl.copy(name = unique))
                                  case None         => fn0Raw
                                }
                                (fn1, wit0, clsDeps, moduleDeps) <- compileAndReplaceStructDependencies(log, fn0, deps)(
                                  structDefs
                                )
                              } yield (
                                fn1 :: xs ::: deps.resolvedFunctions,
                                // Add this function's body-discovered deps to next iteration's queue.
                                // Avoid duplicates: skip if already in `remaining` (this-iteration queue)
                                // or `depss` (already-queued-for-next-iter). Importantly, do NOT filter
                                // against `deps.functions` itself - that's the SOURCE of wit0, so
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
                                CompilerException(s"Cannot resolve struct def for owner of function ${defDef}", e)
                              )
                            // _ <- if (clsDeps != Set(originalFnOwnerStructDef)) s"Bad clsDep (${clsDeps.map(_.repr)}.contains(${originalFnOwnerStructDef.repr}) == false)".fail else ().success
                          } yield (
                            fn.copy(decl = fn.decl.copy(name = target.calleeName)) :: xs,
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
      requested = interfaceImports.result()
      imports <- InterfaceResolver
        .importAll(
          requested,
          callers ::: existing ::: fns,
          config.roots,
          config.capabilities.toSet,
          Map.empty,
          List(target)
        )
        .map(_(target))
        .leftMap(errors => CompilerException(errors.mkString("\n")))
    } yield (imports._1 ::: fns, imports._2 ++ clsDeps, moduleSymDeps, requested.nonEmpty)
  }

  private def showTree(using q: Quoted)(t: q.Tree): String =
    Try(t.show(using q.Printer.TreeAnsiCode)).getOrElse(t.toString)

  @SuppressWarnings(Array("bonsai:repeated-subexpression"))
  def compileFn(using
      q: Quoted
  ) //
  (
      sink: Log,
      f: q.DefDef,
      scope: Map[q.Symbol, p.Expr] = Map.empty,
      intrinsify: Boolean = true
  ): Result[(p.Function, q.Dependencies)] =
    for {
      log <- sink.subLog(s"Compile DefDef: ${f.name}").success
      _      = log.info("Body", showTree(f))
      symbol = f.symbol
      rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${symbol.maybeOwner}) ${showTree(f)}")

      // First we run the typer on the return type to see if we can just return a term based on the type.
      (fnRtnTerm -> fnRtnTpe, fnRtnWit) <- Retyper.typer0(f.returnTpt.tpe)

      // We also run the typer on all the def's arguments, all of which should come in the form of a ValDef.
      // TODO handle default value of args (ValDef.rhs)
      (fnArgs, fnArgsWit) <- Remapper.typedTermParams(f)

      fnTypeVars: List[p.Type.Var] =
        f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => p.Type.Var(name) }

      // And then work out whether this def is part of a class/object instance or free-standing (e.g. local methods),
      // class defs will have a `this` receiver arg with the appropriate type.

      owningSymbol <- Retyper
        .owningClassSymbol(symbol)
        .failIfEmpty(s"${symbol} does not have an owning class symbol")
      owningClass <- Retyper.clsSymTyper0(owningSymbol)
      _ = log.info(s"Method owner: $owningClass")
      (receiver, receiverTpeVars) <- Remapper.receiverOf(owningClass)

      allTypeVars = (receiverTpeVars ::: fnTypeVars ::: Remapper.callableParams(f)).distinct

      _ = log.info("DefDef arguments", fnArgs.map((a, n) => s"$a(symbol=${a.symbol}) ~> $n")*)
      _ = log.info("DefDef tpe vars (recv+fn)", allTypeVars.map(_.repr)*)

      // Finally, we compile the def body like a closure or just return the term if we have one.
      (rhsStmts, rhsDeps, rhsThisCls) <- fnRtnTerm match {
        case Some(t) =>
          ((p.Stmt.Return(t) :: Nil, q.Dependencies(), None)).success
        case None =>
          compileTerm(
            sink = log,
            term = rhs,
            root = symbol,
            scope = scope, // We pass an empty scope table as we are compiling an independent fn.
            tpeArgs = Nil,
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
        decl = p.FunctionDecl(
          name = p.Sym(symbol.fullName),
          tpeVars = allTypeVars,
          receiver = receiver.map(p.Arg(_)),
          args = fnArgs.map(_._2).map(p.Arg(_)),
          moduleCaptures = deriveModuleStructCaptures(deps).map(p.Arg(_)),
          termCaptures = Nil,
          rtn = fnRtnTpe,
          affinity = p.Function.Affinity.Offload
        ),
        body = rhsStmts,
        visibility = p.Function.Visibility.Exported,
        fpMode = p.Function.FpMode.Relaxed,
        convention = p.CallConvention.RegularCall
      )
      _ = log.info("Result", compiledFn.repr)
    } yield (compiledFn, deps)

  def compileTerm(using
      q: Quoted
  )(
      sink: Log,
      term: q.Term,
      root: q.Symbol,
      scope: Map[q.Symbol, p.Expr],
      tpeArgs: List[p.Type],
      intrinsify: Boolean = true
  ): Result[(List[p.Stmt], p.Type, q.Dependencies, Option[(q.ClassDef, p.Type.Struct)])] = for {
    log <- sink.subLog {
      val label = scala.util
        .Try(term.pos)
        .map(pos => s"${pos.sourceFile.name}:${pos.startLine}~${pos.endLine}")
        .getOrElse("<no pos>")
      s"Compile term: $label"
    }.success
    _ = log.info(
      "Body (AST)",
      Try(pprint.tokenize(term, indent = 1, showFieldNames = true).mkString).getOrElse(term.toString)
    )
    _ = log.info("Body (Ascii)", showTree(term))

    (termValue, c)      <- q.RemapContext(root = root, refs = scope).mapTerm(term, None, tpeArgs)
    (_ -> termTpe, wit) <- Retyper.typer0(term.tpe)
    _ <-
      if (
        termTpe != termValue.tpe && !((termTpe, termValue.tpe) match {
          case (p.Type.Exec(_, _, _), p.Type.FnRef(_)) => true
          case _                                       => false
        })
      ) {
        s"Term type ($termTpe) is not the same as term value type (${termValue.tpe}), term was $termValue".fail
      } else ().success
    statements          = c.stmts :+ p.Stmt.Return(termValue)
    (optStmts, optDeps) = (statements, c.deps)
    // if (intrinsify) runLocalOptPass(statements, c.deps)
    // else (statements, c.deps)
  } yield (optStmts, termTpe, optDeps, c.thisCls)

  def findMatchingClassInHierarchy(using
      q: Quoted
  )(symbol: q.Symbol, clsLut: Map[p.Sym, p.StructDef]): Option[p.StructDef] = {
    val hierarchy: List[p.Sym] = structName0(symbol) ::
      q.TypeIdent(symbol).tpe.baseClasses.map(structName0(_))
    hierarchy.collectFirst(Function.unlift(clsLut.get(_)))
  }

  def compileAndReplaceStructDependencies(using
      q: Quoted
  )(
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
        replacedClasses.flatMap { case (tpeAps, sdef) =>
          tpeAps.map { case ap @ p.Type.Struct(_, ts) =>
            ap -> (p.Type.Struct(sdef.name, ts): p.Type.Struct)
          }
        }.toMap ++ replacedModules.map { case (tpe, sdef) =>
          tpe -> (p.Type.Struct(sdef.name, Nil): p.Type.Struct)
        }.toMap

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
        defdef -> ivks.map(ivk =>
          ivk.copy(
            tpeArgs = ivk.tpeArgs.map(replaceTpe(_)),
            receiver = ivk.receiver.map(replaceTpeForTerm(_)),
            args = ivk.args.map(replaceTpeForTerm(_)),
            rtn = replaceTpe(ivk.rtn)
          )
        )
      )

      _ = log.info("Type replacements", typeLut.map((a, b) => s"${a.repr} => ${b.repr}").toList.sorted*)
      _ = log.info("All struct defs", structDefs.toList.map(_.repr).sorted*)
      _ = log.info(
        "Dependent defs",
        mappedFnDeps.map((f, ivks) => s"${f.fullName} \ncallsites: \n${ivks.map(_.repr).mkString("\n")}").toList*
      )
      _ = log.info("Fn after replacements", mappedFn.repr)

    } yield (mappedFn, mappedFnDeps, structDefs, deps.modules.keySet)

  private def liftClosureReceivers(program: p.Program): p.Program = {
    def usesReceiver(fn: p.Function, recv: p.Arg): Boolean =
      fn.collectFirst_[p.Term] { case p.Term.Select(root, _, _) if root == recv.named => true }.isDefined
    val closures = (program.entry.toList ::: program.functions).filter { fn =>
      fn.name.fqn.lastOption.exists(_.contains("$anonfun")) && fn.receiver.nonEmpty
    }
    val (lifted, dropped) = closures.partition(fn => fn.receiver.exists(usesReceiver(fn, _)))
    val liftedNames       = lifted.map(_.name).toSet
    val droppedNames      = dropped.map(_.name).toSet
    def fix(fn: p.Function): p.Function = {
      val base =
        if (droppedNames(fn.name)) fn.copy(decl = fn.decl.copy(receiver = None))
        else if (liftedNames(fn.name))
          fn.copy(decl = fn.decl.copy(receiver = None, termCaptures = fn.receiver.toList ::: fn.termCaptures))
        else fn
      base.modifyAll[p.Expr] {
        case ivk @ p.Expr.Invoke(p.Type.FnRef(name), _, Some(_), _, _) if droppedNames(name) =>
          ivk.copy(receiver = None)
        case ivk @ p.Expr.Invoke(p.Type.FnRef(name), _, Some(recv), args, _) if liftedNames(name) =>
          ivk.copy(receiver = None, args = recv :: args)
        case expr => expr
      }
    }
    program.copy(entry = program.entry.map(fix(_)), functions = program.functions.map(fix(_)))
  }

  def wireCaptures(program: p.Program, includeModules: Boolean = true): Result[p.Program] = program.entry match {
    case None => "compiler optimisation removed the program entry".fail
    case Some(entry) =>
      val fnByName: Map[p.Sym, p.Function] = (entry :: program.functions).map(fn => fn.name -> fn).toMap
      // Transitive module-capture types each function needs, including those from its callees.
      // The call graph is invariant across iterations: only the per-fn capture sets grow;
      // we collect callees once up front rather than re-walking every body each iteration.
      def computeTransitiveModuleCaps(): Map[p.Sym, List[p.Arg]] = {
        val callGraph: Map[p.Sym, List[p.Sym]] =
          fnByName.view
            .mapValues(_.collectWhere[p.Expr] { case p.Expr.Invoke(p.Type.FnRef(name), _, _, _, _) => name })
            .toMap
        @tailrec def fixpoint(current: Map[p.Sym, List[p.Arg]]): Map[p.Sym, List[p.Arg]] = {
          val next = current.map { case (sym, caps) =>
            val seen = mutable.Set.from(caps.map(_.named.tpe))
            val merged = callGraph(sym)
              .foldLeft(caps.reverse) { (acc, callee) =>
                current.getOrElse(callee, Nil).foldLeft(acc) { (acc1, cap) =>
                  if (seen.add(cap.named.tpe)) cap :: acc1 else acc1
                }
              }
              .reverse
            sym -> merged
          }
          if (next == current) current else fixpoint(next)
        }
        fixpoint(fnByName.view.mapValues(_.moduleCaptures).toMap)
      }
      def declaredNames(fn: p.Function): Set[p.Named] =
        (fn.receiver.toList ::: fn.args).map(_.named).toSet ++ fn
          .collectWhere[p.Stmt] {
            case p.Stmt.Var(named, _, _)                => named :: Nil
            case p.Stmt.ForRange(induction = induction) => induction :: Nil
            case p.Stmt.Try(_, handlers, _)             => handlers.flatMap(_.binder)
          }
          .flatten
      def computeTransitiveTermCaps(): Map[p.Sym, List[p.Arg]] = {
        val callGraph: Map[p.Sym, List[p.Sym]] =
          fnByName.view
            .mapValues(_.collectWhere[p.Expr] { case p.Expr.Invoke(p.Type.FnRef(name), _, _, _, _) => name })
            .toMap
        val declared = fnByName.view.mapValues(declaredNames(_)).toMap
        @tailrec def fixpoint(current: Map[p.Sym, List[p.Arg]]): Map[p.Sym, List[p.Arg]] = {
          val next = current.map { case (sym, caps) =>
            if (sym == entry.name) sym -> caps
            else {
              val inherited = callGraph(sym).flatMap(callee => current.getOrElse(callee, Nil))
              val extra     = inherited.filterNot(cap => declared(sym)(cap.named)).distinctBy(_.named)
              sym -> (caps ::: extra.filterNot(cap => caps.exists(_.named == cap.named)))
            }
          }
          if (next == current) current else fixpoint(next)
        }
        fixpoint(fnByName.view.mapValues(_.termCaptures).toMap)
      }
      val transitive   = if (includeModules) computeTransitiveModuleCaps() else Map.empty[p.Sym, List[p.Arg]]
      val termCaps     = computeTransitiveTermCaps()
      val entryAllCaps = transitive.getOrElse(entry.name, entry.moduleCaptures)
      val newEntry     = entry.copy(decl = entry.decl.copy(moduleCaptures = entryAllCaps))
      val entryCapByTpe: Map[p.Type, p.Named] =
        (newEntry.moduleCaptures ++ newEntry.termCaptures).map(arg => arg.named.tpe -> arg.named).toMap

      // Update each function's own moduleCaptures to its transitive set, so when its body
      // forwards a capture into a callee Invoke, the capture name is in scope.
      def updateFnCaps(fn: p.Function): p.Function = {
        val moduleCaps = if (includeModules) transitive.getOrElse(fn.name, fn.moduleCaptures) else fn.moduleCaptures
        val termCaps0  = termCaps.getOrElse(fn.name, fn.termCaptures)
        if (moduleCaps == fn.moduleCaptures && termCaps0 == fn.termCaptures) fn
        else fn.copy(decl = fn.decl.copy(moduleCaptures = moduleCaps, termCaptures = termCaps0))
      }

      // Rebuild the LUT from post-updateFnCaps functions so callee.moduleCaptures reflects
      // the transitive set the backend will actually register.
      val updatedFns                              = newEntry :: program.functions.map(updateFnCaps(_))
      val updatedFnByName: Map[p.Sym, p.Function] = updatedFns.map(fn => fn.name -> fn).toMap

      def patchFn(fn: p.Function): p.Function = {
        val ownCapByTpe: Map[p.Type, p.Named] =
          (fn.moduleCaptures ++ fn.termCaptures).map(arg => arg.named.tpe -> arg.named).toMap
        val inScope = declaredNames(fn) ++ (fn.moduleCaptures ++ fn.termCaptures).map(_.named)
        def patchIvk(ivk: p.Expr.Invoke): p.Expr.Invoke = (ivk.callee match {
          case p.Type.FnRef(name) => updatedFnByName.get(name)
          case _                  => None
        }) match {
          case None    => ivk
          case Some(c) =>
            // captures are always a leading prefix; a pre-opt patch may have added ones the
            // optimiser has since dropped from the callee, so normalise to the current set
            val existing = math.max(0, ivk.args.size - c.args.size)
            val modules  = if (includeModules) c.moduleCaptures else Nil
            if (modules.size + c.termCaptures.size == existing) ivk
            else {
              val prefix = ivk.args.take(existing)
              val actual = ivk.args.drop(existing)
              def lookup(arg: p.Arg): p.Term = {
                val tpe = arg.named.tpe
                selectTerm(
                  Nil,
                  if (inScope(arg.named)) arg.named
                  else ownCapByTpe.get(tpe).orElse(entryCapByTpe.get(tpe)).getOrElse(arg.named)
                )
              }
              val terms =
                if (prefix.size >= c.termCaptures.size) prefix.takeRight(c.termCaptures.size)
                else c.termCaptures.map(lookup(_))
              ivk.copy(args = modules.map(lookup(_)) ::: terms ::: actual)
            }
        }
        fn.modifyAll[p.Expr] {
          case ivk: p.Expr.Invoke => patchIvk(ivk)
          case x                  => x
        }
      }
      program
        .copy(
          entry = Some(patchFn(newEntry)),
          functions = updatedFns.tail.map(patchFn(_))
        )
        .success
  }

  private def lambdaEntry(
      region: p.Function,
      functions: List[p.Function]
  ): Result[(p.Function, List[p.Function])] =
    region.collectWhere[p.Type] { case p.Type.FnRef(name) => name }.distinct match {
      case name :: Nil =>
        functions.filter(_.name == name) match {
          case lambda :: Nil =>
            (
              lambda.copy(
                decl = lambda.decl.copy(affinity = p.Function.Affinity.Offload),
                visibility = p.Function.Visibility.Exported,
                convention = p.CallConvention.OffloadEntry
              ),
              functions.filterNot(_.name == name)
            ).success
          case other => s"callable `${name.repr}` has ${other.size} definitions".fail
        }
      case other => s"expected the expression to be a single lambda, it references ${other.map(_.repr)}".fail
    }

  def compileExpr(using
      q: Quoted
  )(
      sink: Log,
      expr: Expr[Any],
      prisms: PrismTables,
      config: LaunchConfig,
      callable: Boolean = false
  ): Result[
    (                             //
        List[(p.Named, q.Term)],  //
        Map[p.Sym, Prism],        //
        Map[p.Sym, p.Sym],        //
        List[(String, p.Program)] //
    )
  ] = {
    val targets = config.targetNames
    compileExprFor(sink, expr, prisms, config, callable, targets.head).flatMap {
      case (captures, prismRefs, monoMap, program, imported) =>
        val others =
          if (!imported) targets.tail.map(_ -> program.success)
          else
            targets.tail.map(target => target -> compileExprFor(sink, expr, prisms, config, callable, target).map(_._4))
        others
          .traverse((target, result) => result.map(target -> _))
          .map(rest => (captures, prismRefs, monoMap, (targets.head -> program) :: rest))
    }
  }

  private def compileExprFor(using
      q: Quoted
  )(
      sink: Log,
      expr: Expr[Any],
      prisms: PrismTables,
      config: LaunchConfig,
      callable: Boolean,
      target: String
  ): Result[
    (                                       //
        List[(p.Named, q.Term)],            //
        Map[p.Sym, polyregion.prism.Prism], //
        Map[p.Sym, p.Sym],                  //
        p.Program,                          //
        Boolean                             //
    )
  ] = for {
    log <- sink.subLog("Expr compiler").success
    term = expr.asTerm
    pos  = term.pos
    // Generate a name for the expr first.
    exprName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"
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
      decl = p.FunctionDecl(
        name = p.Sym(exprName),
        tpeVars = Nil,
        receiver = None,
        args = Nil,
        moduleCaptures = deriveModuleStructCaptures(exprDeps).map(p.Arg(_)),
        termCaptures = (capturedNames.map(_._1) ++
          exprThisCls.map((_, tpe) => p.Named("this", tpe))).map(p.Arg(_)),
        rtn = exprTpe,
        affinity = p.Function.Affinity.Offload
      ),
      body = exprStmts,
      visibility = p.Function.Visibility.Exported,
      fpMode = p.Function.FpMode.Relaxed,
      convention = p.CallConvention.OffloadEntry
    )

    symbolToMacroDefinedDefDefs = q
      .collectTree[(q.Symbol, q.DefDef)](term) {
        case dd: q.DefDef =>
          val sym = dd.symbol
          sym.tree match {
            case q.DefDef(_, _, _, None) => (sym -> dd) :: Nil
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
      capturedNames.map { (n, r) =>
        val sym = r.symbol
        s"$r(symbol=$sym, ${sym.pos}) ~> ${n.repr}"
      }*
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
      exprDeps.resolvedFunctions.map(_.repr)*
    )

    // We mark all potentially required *concrete* virtual methods in the universe.
    // Also walk each captured class's tree for nested ClassDefs (e.g. anonymous classes from
    // `given Monoid[Int] = new Monoid { ... }`); they're the concrete dispatch targets for
    // abstract trait method calls and don't show up directly in exprDeps.classes since they're
    // only the runtime types of given fields, not statically-named call receivers.
    declaredClasses = {
      val direct = exprDeps.classes.keys.toList
      val nested = direct.flatMap(c =>
        Try(q.collectTree[q.Symbol](c.tree) {
          case cd: q.ClassDef =>
            val sym = cd.symbol
            if (sym != c) List(sym) else Nil
          case _ => Nil
        }).getOrElse(Nil)
      )
      (direct ++ nested).distinct
    }
    virtualCalls = {
      val known = (declaredClasses ++ Dispatch.compilationUnit).distinct
      def referent(t: p.Type): Option[p.Sym] = t match {
        case p.Type.Struct(name, _)                => Some(name)
        case p.Type.Ptr(p.Type.Struct(name, _), _) => Some(name)
        case _                                     => None
      }
      exprDeps.functions.toList.flatMap { (method, ivks) =>
        val receivers = ivks.flatMap(_.receiver.flatMap(r => referent(r.tpe)))
        val arms      = Dispatch.classesOf(method, known)
        Option.when(!receivers.exists(prisms.structDefs.contains) && Dispatch.isVirtual(method, arms))(
          (method, arms, receivers)
        )
      }
    }
    genericArms = virtualCalls
      .flatMap((method, arms, _) =>
        arms.collect {
          case (cls, _) if Dispatch.isGeneric(cls) => s"${cls.fullName} (candidate of ${method.fullName})"
        }
      )
      .distinct
    _ <-
      if (genericArms.isEmpty) ().success
      else
        s"Cannot devirtualise generic candidate classes, a tag identifies a class but not its type arguments: ${genericArms.mkString(", ")}".fail
    armClasses      = virtualCalls.flatMap(_._2.map(_._1))
    implClasses     = virtualCalls.flatMap(_._2.map(_._2.maybeOwner)).filter(_.isClassDef)
    allClassSymbols = (declaredClasses ++ armClasses ++ implClasses).distinct
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
    }

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
        decl = p.FunctionDecl(
          name = sig.name,
          tpeVars = sig.tpeVars,
          receiver = sig.receiver.map(p.Named("this", _)).map(p.Arg(_)),
          args = sig.args.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
          moduleCaptures = sig.moduleCaptures.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
          termCaptures = sig.termCaptures.zipWithIndex.map((t, i) => p.Named(s"arg$i", t)).map(p.Arg(_)),
          rtn = sig.rtn,
          affinity = p.Function.Affinity.Offload
        ),
        // Trait/abstract method body: synthesise a single Return of poison so the function is
        // well-formed. Actual implementation is dispatched dynamically; this body only matters
        // for the verifier (requires at least one statement and a Return).
        body = p.Stmt.Return(p.Expr.Alias(p.Term.Poison(sig.rtn))) :: Nil,
        visibility = p.Function.Visibility.Exported,
        fpMode = p.Function.FpMode.Relaxed,
        convention = p.CallConvention.RegularCall
      )
      log.info(s"Abstract (trait function)", sig.repr)
      fn
    })

    dispatchSites = virtualCalls.map { (method, arms, receivers) =>
      Dispatch.Site(
        p.Sym(method.fullName),
        receivers,
        arms.map((cls, impl) =>
          Dispatch.Arm(
            structName0(cls),
            syntheticOverrideNames.getOrElse(impl, p.Sym(impl.fullName)),
            structName0(impl.maybeOwner)
          )
        )
      )
    }
    dispatchTags  = Dispatch.tagsOf(dispatchSites)
    dispatchTable = Dispatch.table(dispatchSites, dispatchTags)

    exprDeps <- exprDeps
      .copy(
        functions = depsWithOverrides,
        classes = exprDeps.classes ++ (armClasses ++ implClasses)
          .filterNot(exprDeps.classes.contains)
          .distinct
          .map(c => c -> Set[p.Type.Struct](p.Type.Struct(structName0(c), Nil)))
      )
      .success

    // The methods in a prism will have the receiver set to the mirrored class.
    // So before we compile any dependent methods, we must first replace all struct types.
    // And since we are visiting all structures, we also compile the non-replaced ones as well.
    (rootFn, rootDependentFns, rootDependentStructs, rootDependentModuleSymbols) <- compileAndReplaceStructDependencies(
      log,
      exprFn,
      exprDeps
    )(prisms.structDefs)

    // We got a compiled fn now, now compile all dependencies. Pass the synthetic-name rename
    // map so anonymous-class overrides get distinct IR names per (class symbol, method).
    (allRootDependentFns, allRootDependentStructs, allRootDependentModuleSymbols, imported) <-
      compileAllDependencies(
        log,
        rootDependentFns,
        symbolToMacroDefinedDefDefs,
        exprDeps.resolvedFunctions,
        syntheticOverrideNames,
        rootFn :: Nil,
        config,
        target
      )(prisms.functions, prisms.structDefs)

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
      def removeThisFromRef(t: q.Symbol): q.Ref = {
        val companion = t.companionModule
        q.Ref(companion) match {
          case select @ q.Select(root @ q.This(_), _) if companion.maybeOwner.flags.is(q.Flags.Module) =>
            removeThisFromRef(root.symbol).select(select.symbol)
          case x => x
        }
      }
      symbol.fullName.replace('.', '_') -> removeThisFromRef(symbol)
    }.toMap ++ exprThisCls.map((clsDef, _) => "this" -> q.This(clsDef.symbol)).toMap

    (entryFn, programFns) <-
      if (!callable) (rootFn, abstractFnsWithAssertBody ::: allRootDependentFns ::: exprDeps.resolvedFunctions).success
      else lambdaEntry(rootFn, abstractFnsWithAssertBody ::: allRootDependentFns ::: exprDeps.resolvedFunctions)

    unopt = liftClosureReceivers(
      Devirtualise(dispatchTable)(
        p.Program(
          Some(entryFn),
          IntrinsicTable.dedupe(programFns),
          Dispatch.decorate(
            (rootDependentStructs ++ allRootDependentStructs).toList.distinct,
            dispatchSites,
            dispatchTags
          ),
          metadata = Dispatch.tagMetadata(dispatchTags)
        ),
        log
      )
    )
    candidatesWithoutDef = dispatchTags.keys.filterNot(unopt.defs.map(_.name).toSet).toList
    _ <-
      if (candidatesWithoutDef.isEmpty) ().success
      else
        s"Devirtualisation candidates without a struct definition: ${candidatesWithoutDef.map(_.repr).mkString(", ")}".fail
    _ = log.info(
      s"Program compiled (unpot), structures = ${unopt.defs.size}, functions = ${unopt.functions.size}"
    )

    unoptLog <- log.subLog("Unopt").success
    _ = unoptLog.info(s"Structures = ${unopt.defs.size}", unopt.defs.map(_.repr)*)
    _ = unoptLog.info(s"Functions  = ${unopt.functions.size}", unopt.functions.map(_.repr)*)
    _ = unoptLog.info(s"Entry", unopt.entry.fold("<none>")(_.repr))

    // verify before optimisation
    unoptVerification <- Verify(unopt, unoptLog, verifyFunction = false).success
    _ = logVerifier(unoptLog, unoptVerification)
    _ <- failIfVerificationErrors("Unopt", unoptVerification)

    // run the global optimiser

    optPassLog = log.subLog("Opt Passes")

    unoptWired <- wireCaptures(unopt, includeModules = false)
    specialised = scala.Function.chain(
      List[ProgramPass](printPass(Intrinsify), printPass(Specialisation)).map(p => p(_, optPassLog))
    )(unoptWired)
    specialisedTerms <- wireCaptures(specialised, includeModules = false)
    specialisedWired <- wireCaptures(DeadArgElimination(specialisedTerms, optPassLog))
    opt = scala.Function.chain(
      List[ProgramPass](printPass(CallableInline), ArithmeticGuard, PartialEval(), DeadArgElimination).map(p =>
        p(_, optPassLog)
      )
    )(specialisedWired)
    (monomorphicToPolymorphicSym, opt) <- MonoStruct(opt, log).success

    opt <- wireCaptures(opt)

    // verify again after optimisation
    optLog = log.subLog("Opt")
    optVerification <- Verify(opt, optLog, verifyFunction = true).success

    _ = optLog.info(s"Structures = ${opt.defs.size}", opt.defs.map(_.repr)*)
    _ = optLog.info(s"Functions  = ${opt.functions.size}", opt.functions.map(_.repr)*)
    _ = optLog.info(s"Entry", opt.entry.fold("<none>")(_.repr))

    _ = logVerifier(optLog, optVerification)
    _ <- failIfVerificationErrors("Opt", optVerification)

    capturedNameTable = capturedNames.map((name, ref) => name.symbol -> (name.tpe, ref)).toMap
    finalEntry <- opt.entry.failIfEmpty("compiler optimisation removed the program entry")
    captured <- (finalEntry.moduleCaptures ++ finalEntry.termCaptures).traverse { n =>
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

    mirrorsBySource = prisms.mirrors.groupBy(_._1.source)

    // List[(StructDef, Option[Prism])]
  } yield (
    captured.map((arg, term) => arg.named -> term),
    // For each def, we find the original name before monomorphic specialisation and then resolve the term mirrors.
    // Some defs (LLVM-side opaque placeholders, abstract typeclasses) have no entry in the
    // monomorphic-to-polymorphic table - those have no prism, so just skip them.
    opt.defs.flatMap { monomorphicDef =>
      monomorphicToPolymorphicSym.get(monomorphicDef.name) match {
        case Some(polymorphicSym) =>
          mirrorsBySource.getOrElse(polymorphicSym, Nil).map(monomorphicDef.name -> _)
        case None => Nil
      }
    }.toMap,
    monomorphicToPolymorphicSym,
    opt,
    imported
  )

}
