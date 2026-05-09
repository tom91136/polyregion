package polyregion.scalalang

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

import scala.annotation.tailrec

object Remapper {

  private def fullyApplyGenExec(exec: p.Type.Exec, tpeArgs: List[p.Type]): p.Type.Exec = {
    val tpeTable = exec.tpeVars.zip(tpeArgs).toMap
    // Recurse fully so type vars nested inside Struct args, Ptr components, or curried inner Execs
    // (e.g. `<T>(Int) => <>(ClassTag[T]) => Array[T]`) are all rewritten.
    def resolve(t: p.Type): p.Type = t match {
      case p.Type.Var(name) =>
        tpeTable.get(name) match {
          case Some(value) => value
          case None        => ???
        }
      case p.Type.Struct(name, args) =>
        p.Type.Struct(name, args.map(resolve))
      case p.Type.Ptr(comp, space)         => p.Type.Ptr(resolve(comp), space)
      case p.Type.Arr(comp, length, space) => p.Type.Arr(resolve(comp), length, space)
      case p.Type.Exec(innerVars, args, rtn) =>
        p.Type.Exec(innerVars, args.map(resolve), resolve(rtn))
      case x => x
    }
    p.Type.Exec(Nil, exec.args.map(resolve), resolve(exec.rtn).asInstanceOf[p.Type])
  }

  @tailrec private def collectExecArgLists(exec: p.Type, out: List[List[p.Type]] = Nil): List[List[p.Type]] =
    exec match {
      case p.Type.Exec(_, args, rtn) => collectExecArgLists(rtn, out :+ args)
      case _                         => out
    }

  @tailrec private def resolveExecRtnTpe(tpe: p.Type): p.Type = tpe match {
    case p.Type.Exec(_, _, rtn) => resolveExecRtnTpe(rtn)
    case x                      => x
  }

  def ownersList(using q: Quoted)(sym: q.Symbol): LazyList[q.Symbol] =
    LazyList.unfold(sym.maybeOwner)(s => if (s.isNoSymbol) None else Some(s, s.maybeOwner))

  def owningClassSymbol(using q: Quoted)(sym: q.Symbol): Option[q.Symbol] = ownersList(sym).find(_.isClassDef)

  def deriveModuleStructCaptures(using q: Quoted)(d: q.Dependencies): List[p.Named] =
    d.modules.values.toList.map(t => p.Named(t.name.fqn.mkString("_"), t))

  extension (using q: Quoted)(c: q.RemapContext) {

    private def typerAndWitness(repr: q.TypeRepr): Result[(q.Retyped, q.RemapContext)] =
      Retyper.typer0(repr).map((x, wit) => (x, c.updateDeps(d => d.copy(classes = d.classes |+| wit))))

    private def typerNAndWitness(reprs: List[q.TypeRepr]): Result[(List[(q.Retyped)], q.RemapContext)] =
      Retyper.typer0N(reprs).map((xs, wit) => (xs, c.updateDeps(d => d.copy(classes = d.classes |+| wit))))

    def mapTrees(args: List[q.Tree]): Result[(p.Expr, q.RemapContext)] = args match {
      case Nil => (p.Expr.Alias(p.Term.Unit0Const), c).pure
      case x :: xs =>
        c
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            c0.mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Result[(p.Expr, q.RemapContext)] = tree match {
      case q.ValDef(name, tpeTree, Some(rhs)) =>
        for {
          (name, c)        <- c.down(tree).mkName(tree.symbol).success
          (term -> tpe, c) <- c.typerAndWitness(tpeTree.tpe)
          // if tpe is singleton, substitute with constant directly
          (ref, c) <- term.fold((c !! tree).mapTerm(rhs, Some(tpe)))(x => (x, c).success)
          // term
        } yield (
          p.Expr.Alias(p.Term.Unit0Const),
          c ::= p.Stmt.Var(p.Named(name, tpe), Some(ref))
          // case (v: q.ErasedMethodVal, tpe: q.ErasedFnTpe) => c.suspend(name -> tpe)(v)
        )
      case q.ValDef(name, tpe, None) => c.fail(s"Unexpected variable $name:$tpe")
      // TODO DefDef here comes from general closures ( (a:A) => ??? )

      case defDef @ q.DefDef(_, _, _, _) =>
        // The outliner would consider function args as foreign so we need to collect them first and discard.
        val argSymbols = defDef.termParamss.flatMap(_.params).map(_.symbol).toSet

        for {
          (captures, c) <- RefOutliner
            .outline(RenderedLog(""), defDef.rhs.get)
            .map((captures, v) => captures.filterNot(cap => argSymbols.contains(cap._2.symbol)))
            .map(xs =>
              xs.foldLeft(List.empty[(q.Symbol, p.Named)] -> c) { case ((acc, c0), (_, ref, (term, tpe))) =>
                val (name, c) = c0.mkName(ref.symbol)
                ((ref.symbol, p.Named(name, tpe)) :: acc) -> c
              }
            )
          (fn, c0) <- c.mapFn(defDef, RenderedLog(""))
          allCaptureNames = (fn.termCaptures ++ captures
            .map(_._2)
            .map(p.Arg(_))).distinct // Include outlined and propagated
          c <- c
            .withInvokeCapture(
              defDef.symbol,
              allCaptureNames.map(arg => p.Term.Select(arg.named, Nil, arg.named.tpe))
            )
            .updateDeps(_ |+| c0.deps)
            .updateDeps(_.witness(fn.copy(termCaptures = allCaptureNames)))
            .success

        } yield (
          p.Expr.Alias(p.Term.Unit0Const),
          c // FIXME restore statements!!!
        )

      case q.Import(_, _)        => (p.Expr.Alias(p.Term.Unit0Const), c).success // ignore
      case q.TypeDef(name, tree) => (p.Expr.Alias(p.Term.Unit0Const), c).success // ignore
      case q.ClassDef(name, ctor, parents, selfTpeValDef, body) => (p.Expr.Alias(p.Term.Unit0Const), c).success
      case t: q.Term                                            => (c !! tree).mapTerm(t)

      // def f(x...) : A = ??? === val f : x... => A = ???

      case tree => c.fail(s"Unhandled: $tree\nSymbol:\n${tree.symbol}")
    }

    def mapFn(f: q.DefDef, log: Log): Result[(p.Function, q.RemapContext)] =
      for {

        rhs <- f.rhs.failIfEmpty(s"Function does not contain an implementation: (in ${f.symbol.maybeOwner}) ${f.show}")
        // First we run the typer on the return type to see if we can just return a term based on the type.
        (fnRtnTerm -> fnRtnTpe, fnRtnWit) <- Retyper.typer0(f.returnTpt.tpe)

        // We also run the typer on all the def's arguments, all of which should come in the form of a ValDef.
        // TODO handle default value of args (ValDef.rhs)
        (fnArgs, fnArgsWit) <- f.termParamss.flatMap(_.params).foldMapM { arg =>
          Retyper.typer0(arg.tpt.tpe).map { case (_ -> t, wit) => ((arg, p.Named(arg.name, t)) :: Nil, wit) }
        }

        // And then work out whether this def is part of a class/object instance or free-standing (e.g. local methods),
        //   class defs will have a `this` receiver arg with the appropriate type.
        //   FIXME wording: all methods should have a receiver no?
        owningSymbol <- owningClassSymbol(f.symbol).failIfEmpty(s"${f.symbol} does not have an owning class symbol")
        owningClass  <- Retyper.clsSymTyper0(owningSymbol)
        (receiver, receiverTpeVars) <- owningClass match {
          case t @ p.Type.Struct(_, args) =>
            (Some(p.Named("this", t)), args.collect { case p.Type.Var(n) => n }).success
          case x => s"Illegal receiver: $x".fail
        }
        // Fuse receiver's tpe vars with what's explicitly defined
        fnTpeVars  = f.paramss.flatMap(_.params).collect { case q.TypeDef(name, _) => name }
        allTpeArgs = (receiverTpeVars ::: fnTpeVars).distinct

        // Finally, we compile the def body like a closure or return the term if we have one.
        (fnStmts, c) <- fnRtnTerm match {
          case Some(t) =>
            (p.Stmt.Return(t) :: Nil, c).success
          case None =>
            for {
              // We reuse the same context, but reset anything related to the *current* scope.
              // So, delete any existing statement and `this` witness first
              c <- c
                .copy(depth = 0, stmts = Nil, thisCls = None)
                .success
              (term, c)         <- c.mapTerm(rhs, Some(fnRtnTpe), allTpeArgs.map(p.Type.Var(_)))
              ((_, termTpe), _) <- Retyper.typer0(rhs.tpe)
              _ <-
                if (termTpe == term.tpe) ().success
                else
                  s"Dotty term type ($termTpe) is not the same as PolyAst term value type (${term.tpe}), term was $term".fail
            } yield (c.stmts :+ p.Stmt.Return(term), c)

        }

        fn = p.Function(
          name = p.Sym(f.symbol.fullName),
          tpeVars = allTpeArgs,
          receiver = receiver.map(p.Arg(_)),
          args = fnArgs.map(_._2).map(p.Arg(_)),
          moduleCaptures = deriveModuleStructCaptures(c.deps).map(p.Arg(_)),
          termCaptures = Nil, // filled by Compiler.patchFn from the call sites

          rtn = fnRtnTpe,
          body = fnStmts,
          visibility = p.Function.Visibility.Exported,
          fpMode = p.Function.FpMode.Relaxed,
          isEntry = false
        )

      } yield fn -> c

    def mapTerms(args: List[q.Term]): Result[(List[p.Expr], q.RemapContext)] = args match {
      case Nil => (Nil, c).pure
      case x :: xs =>
        c.mapTerm(x)
          .map((ref, c) => (ref :: Nil, c))
          .flatMap(xs.foldLeftM(_) { case ((rs, c0), term) =>
            c0.mapTerm(term).map((r, c) => (rs :+ r, c))
          })
    }

    private def mkInvoke(
        fn: q.DefDef,
        tpeArgs: List[p.Type],
        receiver: Option[p.Expr],
        args: List[p.Expr],
        rtnTpe: p.Type
    ): Result[(p.Expr, q.RemapContext)] = (fn.symbol, fn.rhs, receiver) match {
      // We handle synthesis of the primary ctor here: primary ctors have the following invariant:
      //  * `fn.name` == "<init>"
      //  * `fn.rhs` is empty
      //  * `receiver` is not an non-empty select and matches the owning class
      case (sym, None, Some(instance)) if sym.maybeOwner.primaryConstructor == sym && sym.name == "<init>" =>
        // For primary ctors, we synthesise the assignment of ctor args to the corresponding fields
        val ctorArgs = fn.termParamss.flatMap(_.params)
        for {
          (ctorArgTpes, c) <- c.typerNAndWitness(ctorArgs.map(_.tpt.tpe))
          fieldNames = ctorArgs.map(_.name) // args in primary ctor are fields

          // Make sure we're requested a struct type here, the class may have generic types so we create a LUT of
          // type vars to type args so we can resolve the concrete field types later.
          tpeVarTable <- rtnTpe match {
            case p.Type.Struct(_, ts) =>
              // Without parents/tpeVars on Type.Struct, we extract Var-named args from the
              // ctor's expected return type (`rtnTpe.args`) and align with the call-site's
              // tpeArgs. Concrete (non-Var) args don't bind: the caller may have already
              // monomorphised them.
              val varNames = ts.collect { case p.Type.Var(n) => n }
              val table    = varNames.zip(tpeArgs).toMap
              table.success
            case bad => s"Requested a ctor with a non-struct type: ${bad.repr}".fail
          }

          instanceSelect <- instance match {
            case p.Expr.Alias(s: p.Term.Select) => s.success
            case _                              => "Ctor invocation on instance must be a Select term".fail
          }

          stmts = fieldNames.zip(args).map { (name, rhs) =>
            val appliedTpe = rhs.tpe.mapLeaf {
              case p.Type.Var(name) => tpeVarTable.getOrElse(name, p.Type.Var(name))
              case x                => x
            }
            val fieldSelect: p.Term.Select = p.Term.Select(
              instanceSelect.root,
              instanceSelect.steps :+ p.PathStep.Field(name),
              appliedTpe
            )
            p.Stmt.Mut(fieldSelect, rhs)
          }
        } yield (instance, c.::=(stmts*))
      case (sym, _, _) => // Anything else is a normal invoke.
        if (receiver.isEmpty)
          throw new RuntimeException("Why is recv empty???")

        val receiverTpeArgs = receiver.map(_.tpe).fold(Nil) {
          case p.Type.Struct(_, tpeArgs) => tpeArgs
          case _                         => Nil
        }

        // termCaptures filled by Compiler.patchFn after this pass.
        val ivk: p.Expr.Invoke = p.Expr.Invoke(
          p.Sym(sym.fullName),
          receiverTpeArgs ::: tpeArgs,
          receiver.map(asTerm),
          args.map(asTerm),
          rtnTpe
        )
        val named = c.named(rtnTpe)
        val c0 = c
          .down(fn)
          .updateDeps(_.witness(fn, ivk)) ::= p.Stmt.Var(named, Some(ivk))
        (selectExpr(Nil, named), c0).success
    }

    // The general idea is that idents are either used as-is or it goes through a series of type/term applications (e.g `{ foo.fn[A](b)(c) }`).
    // We collect the types and term lists in `mapTerm` and then fully apply it here.
    private def mapRef0(
        ref: q.Ref,
        tpeArgs: List[p.Type],
        termArgss: List[List[p.Expr]]
    ): Result[(p.Expr, q.RemapContext)] = c.typerAndWitness(ref.tpe).flatMap {
      case (Some(term) -> tpe, c) => (term, c).success
      case (None -> tpe0, c)      =>
        // Apply any unresolved type vars first.
        val tpe = tpe0 match {
          case exec: p.Type.Exec => fullyApplyGenExec(exec, tpeArgs)
          case x                 => x
        }

        // Call no-arg functions (i.e. `x.toDouble` or just `def fn = ???; fn` ) directly or pass-through if not no-arg
        def invokeOrSelect(
            c: q.RemapContext
        )(sym: q.Symbol, receiver: Option[p.Expr])(select: => Result[p.Expr]) = {

          def isFnTpe(t: p.Type, u: p.Type) = (t, u) match {
            case (p.Type.Struct(structSym, _), p.Type.Exec(tpeVars, args, rtnTpe)) =>
              // Without parents on Type.Struct, we approximate by matching Function$n by sym only.
              structSym match {
                case p.Sym("scala" :: s"Function$n" :: Nil) if args.size == n.toInt => true
                case _                                                              => false
              }
            case _ => false
          }

          sym.tree match {
            case fn: q.DefDef => // `receiver?.$fn`
              // Assert that the term list matches Exec's nested (recursive) types.
              // Note that Exec treats both empty args `()` and no-args as `Nil` where as the collected arg lists through
              // `Apply` will give empty args as `Nil` and not collect no-args at all because no application took place.
              val termTpess = termArgss.map(_.map(_.tpe)) // Poison type???
              val execTpess = collectExecArgLists(tpe)
              for {
                // First we validate the application by checking whether types of the positional arguments line up.
                _ <- (fn.termParamss.isEmpty, termTpess, execTpess) match {
                  case (true, Nil, (Nil :: Nil) | Nil) =>
                    ().success // no-ap; no-arg Exec (`Nil::Nil`) or no Exec (`Nil`)
                  case (false, tss, ess) if tss == ess => ().success // everything else, do the assertion
                  case (ap, tss, ess) => // we may have unapplied generic types from the exec side
                    (tss.flatten, ess.flatten) match {
                      case (ts, es) if ts.size == es.size =>
                        ts.zip(es).traverse {
                          case (t, e) if t == e   => t.success // same type
                          case (t, p.Type.Var(_)) => t.success // applied type
                          case (p.Type.Struct(termSym, termArgs), p.Type.Struct(execSym, execArgs)) =>
                            if (termSym != execSym) s"Class type mismatch: $termSym != $execSym ".fail
                            else if (termArgs.size != execArgs.size)
                              s"Class generic type arity mismatch: ${termArgs.size} != ${execArgs.size} ".fail
                            else ().success
                          case (t, u) if isFnTpe(t, u) || isFnTpe(u, t) => t.success
                          case (t, u)                                   => s"Cannot match $t with $u".fail
                        }
                      case (ts, es) =>
                        s"Argument size mismatch ${ts.map(_.repr)}(size=${ts.size}) != ${es.map(_.repr)}(size=${es.size})".fail
                    }
                }
                rtnTpe = resolveExecRtnTpe(tpe)
                ivk <- c.mkInvoke(fn, tpeArgs, receiver, termArgss.flatten, rtnTpe)
              } yield ivk
            case local: q.ValDef => // sym.$local
              for (s <- select) yield s -> c
            case illegal => s"unexpected invoke/select receiver ${illegal}".fail
          }
        }

        // We handle any reference to arbitrarily nested objects/modules (including direct ident with no nesting, as produced by `inline` calls)
        // directly because they are singletons (i.e. can appear anywhere with no dependencies, even the owner).
        // We traverse owners closes to the ref first (inside out) until we hit a method/package and work out the nesting.
        def handleObjectSelect(ref: q.Ref, named: p.Named) = {

          def mkSelect(owner: q.Symbol) = for {
            tpe <- Retyper.clsSymTyper0(owner)
            (term, c) <- tpe match {
              case s @ p.Type.Struct(rootName, Nil) =>
                for {
                  c <- ref match {
                    case q.Select(qualifier, _) => c.updateDeps(_.witness(qualifier.tpe.typeSymbol, s)).success
                    case q.Ident(_)             => c.updateDeps(_.witness(owner, s)).success
                  }
                  directRoot = p.Named(rootName.fqn.mkString("_"), s)
                  result <- invokeOrSelect(c)(ref.symbol, Some(selectExpr(Nil, directRoot)))(
                    selectExpr(directRoot :: Nil, named).success
                  )
                } yield result

              case bad =>
                s"Illegal type ($bad) derived for owner of ${named.repr}, expecting an object class with no generics and type vars".fail
            }
          } yield (term, c)

          ownersList(ref.symbol)
            .takeWhile(Retyper.isModuleClass(_))
            .toList match {
            case Nil =>
              // Here, an object select can still original from a class, for example:
              //   `class  X { def foo: Unit }` // `foo` is owned by `X` which is a class and not object
              //   `object Y extends X`
              //   `Y.foo`
              // The only way to tell is to check the type of the ref, which should be a TermRef such that the receiver is an object.

              // ref.tpe => MethodType
              // scala.Predef
              // RichInt

              ref.tpe match {
                case q.TermRef(root, _) =>
                  root.classSymbol.filter(_.flags.is(q.Flags.Module)).map(mkSelect(_))
                case _ => // Not nested, we're not dealing with an object select after all.
                  None
              }

            case owner :: _ => Some(mkSelect(owner)) // Owner would be the closes symbol (head) to ref.symbol here.
          }
        }

        def mkThisVar(clsSymbol: q.Symbol) = for {
          tpe <- Retyper.clsSymTyper0(clsSymbol) // TODO what about generics???
          c <- (clsSymbol.tree, tpe) match {
            case (clsDef: q.ClassDef, s @ p.Type.Struct(_, _)) if !clsSymbol.flags.is(q.Flags.Module) =>
              c.bindThis(clsDef, s)
            case _ => c.success
          }
        } yield (p.Named("this", tpe), c)

        // When a ref is owned by a class/object (i.e. `c.root`), we add an implicit `this` reference.
        // Only fire for bare idents and `this.x`-style selects — for `param.x` (e.g. `that.a` in
        // `def +(that: Float2) = ... + that.a`) the qualifier is the parameter itself, NOT this,
        // and rewriting it to `this.a` silently drops the parameter access.
        def handleThisRef(ref: q.Ref, named: p.Named) = {
          val rootIsThisOrAbsent = ref match {
            case _: q.Ident             => true
            case q.Select(_: q.This, _) => true
            case _                      => false
          }
          if (rootIsThisOrAbsent && owningClassSymbol(c.root).contains(ref.symbol.maybeOwner)) {
            val ownerSym = ref.symbol.owner
            Some(for {
              (thisCls, c) <- mkThisVar(ownerSym)
              (invoke, c) <- invokeOrSelect(c)(ref.symbol, Some(selectExpr(Nil, thisCls)))(
                selectExpr(thisCls :: Nil, named).success
              )
            } yield (invoke, c))
          } else None
        }

        // First, we check if the ident's symbol is an object.
        // This handle cases like `ObjA.ObjB` or `($x: ObjA).ObjB`, both should resolve to `ObjB` directly
        if (Retyper.isModuleClass(ref.tpe.typeSymbol)) {
          tpe match { // Object references regardless of nesting can be direct so we use the generated reference name here.
            case s @ p.Type.Struct(name, _) =>
              (
                selectExpr(Nil, p.Named(name.fqn.mkString("_"), tpe)),
                c.updateDeps(_.witness(ref.tpe.typeSymbol, s))
              ).success
            case bad =>
              s"Type assertion failed; ref `${ref.show}` is a module but typer disagrees (tpe = ${bad.repr})".fail
          }
        } else
          ref match {
            case ident @ q.Ident(s) => // this is the root of q.Select, it can appear at top-level as well
              // We may encounter a case where the ident's name is different from the TermRef's name.
              // This is likely a result of inline where we end up with synthetic names, we use the TermRef's name in this case.
              val name = ident.tpe match {
                case q.TermRef(_, name) if name != s => name
                case _                               => s
              }
              val local = p.Named(name, tpe)

              handleThisRef(ident, local)
                .orElse(handleObjectSelect(ident, local))
                .getOrElse {
                  val sym = ident.symbol

                  def synthesiseSelectOnMethodWithReceiver(classSym: q.Symbol) =
                    Retyper.clsSymTyper0(classSym).map(classSym.tree -> _).flatMap {
                      case (clsDef: q.ClassDef, s @ p.Type.Struct(_, _)) =>
                        c.bindThis(clsDef, s).flatMap { c =>
                          val self = p.Named("this", s)
                          invokeOrSelect(c)(sym, Some(selectExpr(Nil, self)))(
                            selectExpr(self :: Nil, local).success
                          )
                        }
                      case (illegal, tpe) => s"Unexpected tree $illegal with type $tpe when resolving local dummy".fail
                    }

                  if (sym.maybeOwner.isLocalDummy) {
                    // We're seeing a def/val defined in the class scope: `class X{ { def a()  } }`.
                    // Method (or any kind of Def) `a` in this case is owned by a local dummy where the parent is `X`.
                    // We happen to be referring to `a`  locally (i.e. via ident, and not select) so we're in the same scope as `a`.
                    // As local dummies don't exist, we need to synthesize the receiver for this invoke to be the owning class of the local dummy, which is `X`.
                    val classSym = sym.maybeOwner.maybeOwner // Up two levels to skip over the local dummy.
                    if (!classSym.isNoSymbol) synthesiseSelectOnMethodWithReceiver(classSym)
                    else s"The owner of $sym does not contain an implementation ".fail
                  } else {
                    // In any other case, we're probably referencing a local ValDef/DefDef that appeared before this.

                    if (sym.isValDef) { // For ValDef, we can ignore the receiver

                      val (name_, c_) = c.mkName(sym)

                      invokeOrSelect(c_)(sym, None)(selectExpr(Nil, p.Named(name_, local.tpe)).success)
                    } else if (sym.isDefDef) { // For DefDef, we need to synthesise one like the local dummy case
                      owningClassSymbol(sym)
                        .failIfEmpty(s"$sym does not contain an implementation")
                        .flatMap(synthesiseSelectOnMethodWithReceiver(_))
                    } else {
                      throw new RuntimeException(s"Unexpected ${sym}, not a valdef or defdef!")
                    }

                  }
                }
            case select @ q.Select(q.Super(term, _), name) =>
              // We handle `super.m` separately: https://www.scala-lang.org/files/archive/spec/2.13/06-expressions.html#this-and-super
              // "The statically referenced member m must be a type or a method"
              for {
                // Here, we're certain the owner of this method (i.e the select) will be the actual super class so we just use the same
                // logic as witnessing `this`.
                // We discard the context here because it binds `this` to the wrong class.
                (thisCls, _) <- mkThisVar(select.symbol.owner)
                c <- (select.symbol.owner.tree, thisCls.tpe) match {
                  case (clsDef: q.ClassDef, tpe: p.Type.Struct) => c.updateDeps(_.witness(clsDef, tpe)).success
                  case (bad, tpe) => s"super class $tpe is not backed by a class tree: $bad".fail
                }

                // we're gonna case this down to the base
                superSubclassSymbol <- term.tpe.classSymbol
                  .failIfEmpty(s"subclass of superclass ${term.tpe.show} does not have a class symbol")
                superSubclassTpe <- Retyper.clsSymTyper0(superSubclassSymbol)

                c <- c.down(select).success
                castResult = c.down(select).named(thisCls.tpe)
                c <- (c ::= p.Stmt.Var(
                  castResult,
                  Some(p.Expr.Cast(selectTerm(Nil, thisCls.copy(tpe = superSubclassTpe)), thisCls.tpe))
                )).success
                (term, c) <- invokeOrSelect(c)(select.symbol, Some(selectExpr(Nil, castResult)))(
                  "illegal selection of a non DefDef symbol from super".fail
                )
              } yield (term, c)
            case select @ q.Select(root, name) => // we have qualifiers before the actual name
              val named = p.Named(name, tpe)
              handleThisRef(select, named)
                .orElse(handleObjectSelect(select, named))
                .getOrElse {
                  // Otherwise we go through the usual path of resolution  (nested classes where each instance has an `this` reference to the owning class)
                  c.mapTerm(root).flatMap {
                    case (root @ p.Expr.Alias(prev: p.Term.Select), c) =>
                      // fuse with previous select if we got one: append a Field step.
                      val fused: p.Term.Select =
                        p.Term.Select(prev.root, prev.steps :+ p.PathStep.Field(named.symbol), named.tpe)
                      invokeOrSelect(c)(select.symbol, Some(root))(p.Expr.Alias(fused).success)
                    case (root, c) =>
                      invokeOrSelect(c)(select.symbol, Some(root))(
                        "illegal selection of a non DefDef symbol from a primitive term (i.e `{ 1.$name }` )".fail
                      )
                  }
                }
          }
    }

    private def mkMut(t: q.Tree, lhs: p.Expr, rhs: p.Expr): (List[p.Stmt], q.RemapContext) = {
      // Stmt.Mut.name must be a Term.Select. Pull it out of the Alias wrapper if the caller
      // already produced one; otherwise this is a programming error at the IR boundary.
      val lhsSelect: p.Term.Select = lhs match {
        case p.Expr.Alias(s: p.Term.Select) => s
        case other => throw new IllegalStateException(s"mkMut lhs must be an Alias(Term.Select); got ${other.repr}")
      }
      if (lhs.tpe == rhs.tpe) {
        (p.Stmt.Mut(lhsSelect, rhs) :: Nil, c)
      } else {
        val c0       = c.down(t)
        val tempName = c0.named(rhs.tpe)
        (
          p.Stmt.Var(tempName, Some(rhs)) ::
            p.Stmt.Mut(lhsSelect, p.Expr.Cast(selectTerm(Nil, tempName), lhs.tpe)) :: Nil,
          c0
        )
      }
    }

    def mapTerm(
        term: q.Term,
        eventualTpe: Option[p.Type] = None,
        tpeArgs: List[p.Type] = Nil,
        termArgss: List[List[p.Expr]] = Nil
    ): Result[(p.Expr, q.RemapContext)] = {

      val c1 = c // c.withDefs(term)

      // Short-circuit: scala.reflect.ClassTag[T] construction. When Array.ofDim[T](n) summons
      // an implicit ClassTag, Scala 3.7's stdlib inlines `ClassTag.apply` into the call site,
      // producing a Block(If(WeakReference cache lookup)) that the term mapper can't traverse
      // (and shouldn't — the kernel never reads ClassTag fields). Replace any sub-expression of
      // ClassTag[T] type with an empty ClassTag struct, which matches the StdLib mirror.
      val isClassTagTpe = term.tpe.widenTermRefByName.dealias match {
        case q.AppliedType(ctor, _) => ctor.typeSymbol.fullName == "scala.reflect.ClassTag"
        case _                      => false
      }
      if (isClassTagTpe && tpeArgs.isEmpty && termArgss.isEmpty) {
        return c1.typerAndWitness(term.tpe).map { case (_ -> tpe, c) =>
          val name = c.named(tpe)
          (selectExpr(Nil, name), c.down(term) ::= p.Stmt.Var(name, None))
        }
      }

      (tpeArgs, termArgss, term) match {
        case (Nil, Nil, q.NamedArg(name, rhs)) => (c1 !! term).mapTerm(rhs) // named argument: `$name = $rhs`
        case (Nil, Nil, q.Typed(x, _))         => (c1 !! term).mapTerm(x)   // type ascription: `value : T`
        case (Nil, Nil, q.Inlined(call, bindings, expansion)) => // inlined DefDef
          for {
            // For non-inlined args, bindings will contain all relevant arguments with rhs.
            // TODO I think call is safe to ignore here? It looks like a subtree from the expansion...
            (_, c) <- (c1 !! term).mapTrees(bindings)
            (v, c) <- (c !! term).mapTerm(expansion)
          } yield (v, c)
        case (Nil, Nil, q.Literal(q.BooleanConstant(v))) => (p.Expr.Alias(p.Term.Bool1Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.IntConstant(v)))     => (p.Expr.Alias(p.Term.IntS32Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.FloatConstant(v)))   => (p.Expr.Alias(p.Term.Float32Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.DoubleConstant(v)))  => (p.Expr.Alias(p.Term.Float64Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.LongConstant(v)))    => (p.Expr.Alias(p.Term.IntS64Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.ShortConstant(v)))   => (p.Expr.Alias(p.Term.IntS16Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.ByteConstant(v)))    => (p.Expr.Alias(p.Term.IntS8Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.CharConstant(v)))    => (p.Expr.Alias(p.Term.IntU16Const(v)), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.UnitConstant()))     => (p.Expr.Alias(p.Term.Unit0Const), c1 !! term).pure
        case (Nil, Nil, q.Literal(q.StringConstant(v))) =>
          ??? // XXX alloc new string instance
        case (Nil, Nil, q.Literal(q.ClassOfConstant(_))) =>
          c1.typerAndWitness(term.tpe).map { case (_ -> tpe, c) =>
            val name = c.named(tpe)
            (selectExpr(Nil, name), (c.down(term)) ::= p.Stmt.Var(name, None))
          }
        case (Nil, Nil, l @ q.Literal(q.NullConstant())) =>
          c1.typerAndWitness(l.tpe).map { case (_ -> tpe, c) =>
            (p.Expr.Alias(p.Term.Poison(tpe)), c !! term)
          }
        case (Nil, Nil, q.This(_)) => // reference to the current class: `this.???`
          // XXX Don't use typerAndWitness here, we need to record the witnessing of `this` separately.
          (c1 !! term).typerAndWitness(term.tpe).flatMap {
            case (None -> (s @ p.Type.Struct(_, _)), c) =>
              // There may be more than one
              term.tpe.classSymbol.map(_.tree) match {
                case Some(clsDef: q.ClassDef) =>
                  c.bindThis(clsDef, s).map(c => (selectExpr(Nil, p.Named("this", s)), c !! term))
                case Some(bad) => s"`this` type symbol points to a non-ClassDef tree: $bad".fail
                case None      => "`this` does not contain a class symbol".fail
              }
            case (Some(value) -> tpe, _) => "`this` isn't supposed to have a value".fail /*(value, c).success*/
            case (None -> illegal, _)    => "`this` isn't typed as a struct type".fail
          }
        case (tpeArgs, termArgss, q.TypeApply(term, args)) => // *single* application of some types: `$term[$args...]`
          for {
            (args, c) <- c1.typerNAndWitness(args.map(_.tpe))
            (term, c) <- c.mapTerm(term, tpeArgs = args.map(_._2), termArgss = termArgss)
          } yield (term, c)
        case (tpeArgs, termArgs, r: q.Ref) =>
          c1.refs.get(r.symbol) match {
            case Some(term) =>
              c1.typerAndWitness(r.tpe).flatMap { case (_ -> tpe, c) =>
                if (term.tpe != tpe) s"Ref type mismatch (${term.tpe} != $tpe)".fail
                else (term, c !! r).success
              }
            case None => (c !! r).mapRef0(r, tpeArgs, termArgs)
          }
        case (Nil, Nil, q.New(tpt)) => // new instance *without* arg application: `new $tpt`
          c1.typerAndWitness(tpt.tpe).map { case (_ -> tpe, c) =>
            val name = c.named(tpe)
            (selectExpr(Nil, name), (c.down(term)) ::= p.Stmt.Var(name, None))
          }
        case (tpeArgs, termArgs0, ap @ q.Apply(fun, args)) => // *single* application of some terms: `$fun($args...)`
          for {
            (args, c) <- c1.down(ap).mapTerms(args)
            (fun, c)  <- (c !! ap).mapTerm(fun, tpeArgs = tpeArgs, termArgss = args :: termArgs0)
          } yield (fun, c)
        case (_, _, q.Block(stat, expr)) => // block expression: `{ $stmts...; $expr }`
          // Type/term args may carry through when an inlined polymorphic call (e.g. ClassTag.apply
          // for an Array.ofDim implicit) lands as a Block result. Pass them down to the result expr.
          for {
            (_, c)   <- (c1 !! term).mapTrees(stat)
            (ref, c) <- c.mapTerm(expr, tpeArgs = tpeArgs, termArgss = termArgss)
          } yield (ref, c)
        case (Nil, Nil, q.Assign(lhs, rhs)) => // simple assignment: `$lhs = $rhs`
          for {
            (lhsRef, c) <- c1.down(term).mapTerm(lhs) // go down here
            (rhsRef, c) <- (c !! term).mapTerm(rhs)
            r <- (lhsRef, rhsRef) match {
              case (s @ p.Expr.Alias(_: p.Term.Select), rhs) =>
                val (xs, c0) = c.mkMut(term, s, rhs)
                (p.Expr.Alias(p.Term.Unit0Const), c0.::=(xs*)).success
              case (lhs, rhs) => c.fail(s"Illegal assign LHS,RHS: lhs=${lhs.repr} rhs=$rhs")
            }
          } yield r
        case (Nil, Nil, q.If(cond, thenTerm, elseTerm)) => // conditional: `if($cond) then $thenTerm else $elseTerm`
          for {
            (_ -> tpe, c)     <- c1.typerAndWitness(term.tpe) // TODO  return term value if already known at type-level
            (condTerm, ifCtx) <- c.down(term).mapTerm(cond)
            (thenTerm, thenCtx) <- ifCtx.noStmts.down(thenTerm).mapTerm(thenTerm)
            (elseTerm, elseCtx) <- thenCtx.noStmts.down(elseTerm).mapTerm(elseTerm)

            _ <-
              if (condTerm.tpe != p.Type.Bool1) s"Cond must be a Bool ref, got ${condTerm}".fail
              else ().success

            mkCondStmts = (tpe: p.Type) => {
              val c      = elseCtx.down(term)
              val name   = c.named(tpe)
              val result = p.Stmt.Var(name, None)

              val (thenStmts, c0) = c.mkMut(term, selectExpr(Nil, name), thenTerm)
              val (elseStmts, c1) = c0.mkMut(term, selectExpr(Nil, name), elseTerm)

              val cond = p.Stmt.Cond(
                asTerm(condTerm),
                thenCtx.stmts ++ thenStmts,
                elseCtx.stmts ++ elseStmts
              )
              (selectExpr(Nil, name), c1.down(term).replaceStmts(ifCtx.stmts :+ result :+ cond)).success
            }

            // See https://dotty.epfl.ch/docs/reference/new-types/union-types-spec.html#erasure
            // If we encounter a union type on the RHS and a struct on the LHS, use the struct type
            // TODO synthesise coproduct proxy on the spot for unions
            cond <- (thenTerm.tpe, elseTerm.tpe, eventualTpe) match {
              case (`tpe`, `tpe`, _) => // Same on both side, perfect
                mkCondStmts(tpe)

              case (
                    p.Type.Struct(_, _),
                    p.Type.Struct(_, _),
                    Some(widened @ p.Type.Struct(_, _))
                  ) =>
                // We got a something like:
                // `val a : Base = if(???) ClassA() else ClassB() # ClassA <: Base, ClassB <: Base`
                // Where the eventual type is widened by the Scala compiler.
                // Without parents on Type.Struct we can't verify the subtype relation here; defer
                // to the verifier downstream.
                mkCondStmts(widened)
              case _ =>
                s"condition unification failure, then=${thenTerm.repr} else=${elseTerm.repr}, expr tpe=${tpe.repr} actual=${term.tpe.widen.simplified.show}".fail
            }
          } yield cond
        case (Nil, Nil, q.While(cond, body)) => // loop: `while($cond) {$body...}`
          for {
            (condTerm, condCtx) <- c1.noStmts.down(term).mapTerm(cond)
            (_, bodyCtx)        <- condCtx.noStmts.mapTerm(body)
          } yield {
            // ANF: hoist the cond compute -- condCtx.stmts holds Var bindings (and Cond branches
            // for short-circuit `&&`/`||`) that produce condTerm. To re-evaluate against
            // body-mutated state we declare every Var up front with mutable=true (one alloca per
            // slot) and re-emit the sequence as Muts inside the body each iteration. Without
            // hoisting, declaring Vars inside Cond branches every iteration creates a fresh
            // alloca and the loop test reads stale state.
            def hoistDecls(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
              case p.Stmt.Var(n, _, _)  => List(p.Stmt.Var(n, None, isMutable = true))
              case p.Stmt.Cond(c, t, f) => hoistDecls(t) ++ hoistDecls(f)
              case _                    => Nil
            }
            def varsToMuts(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
              case p.Stmt.Var(n, Some(e), _) => List(p.Stmt.Mut(p.Term.Select(n, Nil, n.tpe), e))
              case p.Stmt.Var(_, None, _)    => Nil // already declared in hoistedDecls
              case p.Stmt.Cond(c, t, f)      => List(p.Stmt.Cond(c, varsToMuts(t), varsToMuts(f)))
              case other                     => List(other)
            }
            val hoistedDecls = hoistDecls(condCtx.stmts).distinctBy {
              case p.Stmt.Var(n, _, _) => n.symbol
              case other               => other.toString
            }
            val condBody = varsToMuts(condCtx.stmts)
            (
              p.Expr.Alias(p.Term.Unit0Const),
              bodyCtx.replaceStmts(
                c.stmts ++ hoistedDecls ++ condBody :+ p.Stmt.While(asTerm(condTerm), bodyCtx.stmts ++ condBody)
              )
            )
          }
        case (Nil, Nil, q.Closure(rhs, None)) => ???
        case _ =>
          c1.fail(
            s"Unhandled: <${tpeArgs.map(_.repr)}>`$term`(${termArgss}), show=`${term.show}`\nSymbol:\n${term.symbol}"
          )
      }
    }
  }

}
