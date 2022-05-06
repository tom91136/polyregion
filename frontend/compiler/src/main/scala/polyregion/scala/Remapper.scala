package polyregion.scala

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object Remapper {

  import Retyper.{clsSymTyper0, typer0, typer0N}

  private def fullyApplyGenExec(exec: p.Type.Exec, tpeArgs: List[p.Type]): p.Type.Exec = {
    val tpeTable = exec.tpeVars.zip(tpeArgs).toMap
    def resolve(t: p.Type) = t.map {
      case p.Type.Var(name) => tpeTable(name)
      case x                => x
    }
    p.Type.Exec(Nil, exec.args.map(resolve(_)), resolve(exec.rtn))
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

  def selectObject(using q: Quoted)(objectSymbol: q.Symbol): q.Ref = {
    if (!objectSymbol.flags.is(q.Flags.Module)) {
      q.report.error(s"Cannot select Ref from non-object type: ${objectSymbol}")
    }
    println(
      s"chain ${objectSymbol} ${objectSymbol.companionModule} = ${ownersList(objectSymbol.companionModule).toList}"
    )

    //  def removeThisRef(r : q.Ref) =
    //    r match {
    //      case q.Select(q.This(x), y) => q.Select(removeThisRef(x), y)
    //      case x => x
    //    }

    q.Ref.apply(objectSymbol.companionModule)
  }

  extension (using q: Quoted)(c: q.RemapContext) {

    def mapTrees(args: List[q.Tree]): Result[(p.Term, q.RemapContext)] = args match {
      case Nil => (p.Term.UnitConst, c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show.replace("\n", " ; ")))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show.replace("\n", " ; "))).mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Result[(p.Term, q.RemapContext)] = tree match {
      case q.ValDef(name, tpeTree, Some(rhs)) =>
        for {
          (term, tpe) <- typer0(tpeTree.tpe)
          // if tpe is singleton, substitute with constant directly
          (ref, c) <- term.fold((c !! tree).mapTerm(rhs))(x => (x, c).success)
          // term
        } yield (
          p.Term.UnitConst,
          c ::= p.Stmt.Var(p.Named(name, tpe), Some(p.Expr.Alias(ref)))
          // case (v: q.ErasedMethodVal, tpe: q.ErasedFnTpe) => c.suspend(name -> tpe)(v)
        )
      case q.ValDef(name, tpe, None) => c.fail(s"Unexpected variable $name:$tpe")
      // TODO DefDef here comes from general closures ( (a:A) => ??? )
      case q.Import(_, _) => (p.Term.UnitConst, c).success // ignore
      case t: q.Term      => (c !! tree).mapTerm(t)

      // def f(x...) : A = ??? === val f : x... => A = ???

      case tree => c.fail(s"Unhandled: $tree\nSymbol:\n${tree.symbol}")
    }

    def mapTerms(args: List[q.Term]): Result[(List[p.Term], q.RemapContext)] = args match {
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
        receiver: Option[p.Term],
        args: List[p.Term],
        rtnTpe: p.Type
    ): Result[(p.Term, q.RemapContext)] = (fn.symbol, fn.rhs, receiver) match {
      // We handle synthesis of the primary ctor here: primary ctors have the following invariant:
      //  * `fn.name` == "<init>"
      //  * `fn.rhs` is empty
      //  * `receiver` is not an non-empty select and matches the owning class
      case (sym, None, Some(instance)) if sym.maybeOwner.primaryConstructor == sym && sym.name == "<init>" =>
        // For primary ctors, we synthesise the assignment of ctor args to the corresponding fields
        val ctorArgs = fn.termParamss.flatMap(_.params)
        for {
          ctorArgTpes <- typer0N(ctorArgs.map(_.tpt.tpe)).map(_.map(_._2))
          fieldNames = ctorArgs.map(_.name) // args in primary ctor are fields

          // Make sure we're requested a struct type here, the class may have generic types so we create a LUT of
          // type vars to type args so we can resolve the concrete field types later.
          tpeVarTable <- rtnTpe match {
            case p.Type.Struct(_, tpeVars, `tpeArgs`) =>
              if (tpeVars.length != tpeArgs.length) {
                s"Requested a ctor with different type arg length: vars=$tpeVars, args=${tpeArgs}".fail
              } else {
                tpeVars.zip(tpeArgs).toMap.success
              }
            case p.Type.Struct(_, _, bad) =>
              s"Requested a ctor with different type args, ctor=${bad}, requested=${tpeArgs}".fail
            case bad => s"Requested a ctor with a non-struct type: ${bad.repr}".fail
          }

          instancePath <- instance match {
            case p.Term.Select(xs, x) => (xs :+ x).success
            case _                    => "Ctor invocation on instance must be a Select term".fail
          }

          stmts = fieldNames.zip(args).map { (name, rhs) =>
            val appliedTpe = rhs.tpe.map {
              case p.Type.Var(name) => tpeVarTable(name)
              case x                => x
            }
            p.Stmt.Mut(p.Term.Select(instancePath, p.Named(name, appliedTpe)), p.Expr.Alias(rhs), copy = false)
          }
        } yield (instance, c.::=(stmts*))
      case (sym, _, _) => // Anything else is a normal invoke.
        val ivk: p.Expr.Invoke = p.Expr.Invoke(p.Sym(sym.fullName), tpeArgs, receiver, args, rtnTpe)
        val named              = c.named(rtnTpe)
        val c0 = c
          .down(fn)
          .updateDeps(_.witness(fn, ivk)) ::= p.Stmt.Var(named, Some(ivk))
        (p.Term.Select(Nil, named), c0).success
    }

    // The general idea is that idents are either used as-is or it goes through a series of type/term applications (e.g `{ foo.fn[A](b)(c) }`).
    // We collect the types and term lists in `mapTerm` and then fully apply it here.
    private def mapRef0(
        ref: q.Ref,
        tpeArgs: List[p.Type],
        termArgss: List[List[p.Term]]
    ): Result[(p.Term, q.RemapContext)] = typer0(ref.tpe).flatMap {
      case (Some(term), tpe) => (term, c).success
      case (None, tpe0)      =>
        // Apply any unresolved type vars first.
        val tpe = tpe0 match {
          case exec: p.Type.Exec => fullyApplyGenExec(exec, tpeArgs)
          case x                 => x
        }

        println(s"[mapRef0] tpe=${tpe.repr} ${ref.symbol.fullName}=${ref.symbol.flags.show} ")

        def witnessClassTpe(c: q.RemapContext)(sym: q.Symbol, tpe: p.Type) =
          if (sym.isPackageDef) c.success
          else {
            (sym.tree, tpe) match {
              case (cls: q.ClassDef, s @ p.Type.Struct(_, _, _)) =>
                c.updateDeps(_.witness(cls, s)).success
              case (x, s) =>
                c.success
            }
          }

        // Call no-arg functions (i.e. `x.toDouble` or just `def fn = ???; fn` ) directly or pass-through if not no-arg
        def invokeOrSelect(
            c: q.RemapContext
        )(sym: q.Symbol, receiver: Option[p.Term])(select: => Result[p.Term.Select]) = sym.tree match {
          case fn: q.DefDef => // `sym.$fn`
            // Assert that the term list matches Exec's nested (recursive) types.
            // Note that Exec treats both empty args `()` and no-args as `Nil` where as the collected arg lists through
            // `Apply` will give empty args as `Nil` and not collect no-args at all because no no application took place.
            val termTpess = termArgss.map(_.map(_.tpe))
            val execTpess = collectExecArgLists(tpe)
            println(s"Invoke ${receiver} . ${fn}")
            for {
              _ <- (fn.termParamss.isEmpty, termTpess, execTpess) match {
                case (true, Nil, (Nil :: Nil) | Nil) => ().success // no-ap; no-arg Exec (`Nil::Nil`) or no Exec (`Nil`)
                case (false, ts, es) if ts == es     => ().success // everything else, do the assertion
                case (ap, ts, es)                    => ???        // TODO raise failure
              }
              rtnTpe = resolveExecRtnTpe(tpe)
              c   <- witnessClassTpe(c)(sym.maybeOwner, rtnTpe) // TODO this won't work right?
              ivk <- c.mkInvoke(fn, tpeArgs, receiver, termArgss.flatten, rtnTpe)
            } yield ivk
          case _ => // sym.$select
            for {
              s <- select
              c <- witnessClassTpe(c)(sym.maybeOwner, s.tpe)
            } yield (s -> c)
        }

        // We handle any reference to arbitrarily nested objects/modules (including direct indent with no nesting, as produced by `inline` calls)
        // directly because they are singletons (i.e. can appear anywhere with no dependencies, even the owner).
        // We traverse owners closes to the ref first (inside out) until we hit a method/package and work out the nesting.
        def handleObjectSelect(ref: q.Ref, named: p.Named) = ownersList(ref.symbol)
          .takeWhile(Retyper.isModuleClass(_))
          .toList match {
          case Nil => None // not nested
          case xs @ (owner :: _) => // owner would be the closes symbol to ref.symbol here
            Some(for {
              tpe <- clsSymTyper0(owner)
              (term, c) <- tpe match {
                case s @ p.Type.Struct(rootName, Nil, Nil) =>
                  for {
                    c <- ref match {
                      case q.Select(qualifier, _) => c.updateDeps(_.witness(qualifier.tpe.typeSymbol, s)).success
                      case q.Ident(_)             => c.updateDeps(_.witness(owner, s)).success
                    }
                    directRoot = p.Named(rootName.fqn.mkString("_"), s)
                    result <- invokeOrSelect(c)(ref.symbol, Some(p.Term.Select(Nil, directRoot)))(
                      p.Term.Select(directRoot :: Nil, named).success
                    )
                  } yield result

                case bad =>
                  s"Illegal type ($bad) derived for owner of ${named.repr}, expecting an object class with no generics and type vars".fail
              }
            } yield (term, c))
        }

        // When an ref is owned by the current class/object (i.e. `c.root`), we add an implicit `this` reference.
        def handleThisRef(ref: q.Ref, named: p.Named) = if (owningClassSymbol(c.root).contains(ref.symbol.maybeOwner)) {
          Some(for {
            tpe <- clsSymTyper0(ref.symbol.owner) // TODO what about generics???
            cls = p.Named("this", tpe)
            (invoke, c) <- invokeOrSelect(c)(ref.symbol, Some(p.Term.Select(Nil, cls)))(
              p.Term.Select(cls :: Nil, named).success
            )
          } yield (invoke, c))
        } else None

        // First, we check if the ident's symbol is an object.
        // This handle cases like `ObjA.ObjB` or `($x: ObjA).ObjB`, both should resolve to `ObjB` directly
        if (Retyper.isModuleClass(ref.tpe.typeSymbol)) {
          tpe match { // Object references regardless of nesting can be direct so we use the generated reference name here.
            case s @ p.Type.Struct(name, _, _) =>
              (
                p.Term.Select(Nil, p.Named(name.fqn.mkString("_"), tpe)),
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
                  // In any other case, we're probably referencing a local ValDef that appeared before before this.
                  invokeOrSelect(c)(ident.symbol, None)(p.Term.Select(Nil, local).success)
                }

            case select @ q.Select(root, name) => // we have qualifiers before the actual name
              val named = p.Named(name, tpe)
              handleThisRef(select, named)
                .orElse(handleObjectSelect(select, named))
                .getOrElse {
                  // Otherwise we go through the usual path of resolution  (nested classes where each
                  // instance has an `this` reference to the owning class)
                  c.mapTerm(root).flatMap {
                    case (root @ p.Term.Select(xs, x), c) => // fuse with previous select if we got one
                      invokeOrSelect(c)(select.symbol, Some(root))(p.Term.Select(xs :+ x, named).success)
                    case (root, c) => // or simply return whatever it's referring to
                      // `$root.$name` becomes `$root.${select.symbol}()` so we don't need the `$name` here
                      invokeOrSelect(c)(select.symbol, Some(root))(
                        "illegal selection of a non DefDef symbol from a primitive term (i.e `{ 1.$name }` )".fail
                      )
                  }
                }
          }
    }

    // def witnessTpe(tpe: q.TypeRepr) = (tpe.classSymbol) match {
    //   case Some(clsSym) if !clsSym.isNoSymbol =>
    //     println(s" D=${clsSym.tree}")
    //   // c.updateDeps(_.witness())
    //   case bad =>
    //     println(s" D=${bad}")
    // }

    def mapTerm(
        term: q.Term,
        tpeArgs: List[p.Type] = Nil,
        termArgss: List[List[p.Term]] = Nil
    ): Result[(p.Term, q.RemapContext)] = {
      println(s">>${term.show} = ${c.stmts.size} ~ ${term}")
      (tpeArgs, termArgss, term) match {
        case (Nil, Nil, q.Typed(x, _)) => (c !! term).mapTerm(x) // type ascription: `value : T`
        case (Nil, Nil, q.Inlined(call, bindings, expansion)) => // inlined DefDef
          for {
            // for non-inlined args, bindings will contain all relevant arguments with rhs
            // TODO I think call is safe to ignore here? It looks like a subtree from the expansion
            (_, c) <- (c !! term).mapTrees(bindings)
            (v, c) <- (c !! term).mapTerm(expansion)
          } yield (v, c)
        case (Nil, Nil, q.Literal(q.BooleanConstant(v))) => (p.Term.BoolConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.IntConstant(v)))     => (p.Term.IntConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.FloatConstant(v)))   => (p.Term.FloatConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.DoubleConstant(v)))  => (p.Term.DoubleConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.LongConstant(v)))    => (p.Term.LongConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.ShortConstant(v)))   => (p.Term.ShortConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.ByteConstant(v)))    => (p.Term.ByteConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.CharConstant(v)))    => (p.Term.CharConst(v), c !! term).pure
        case (Nil, Nil, q.Literal(q.UnitConstant()))     => (p.Term.UnitConst, c !! term).pure
        case (Nil, Nil, q.This(_)) => // reference to the current class: `this.???`
          typer0(term.tpe).flatMap {
            case (Some(value), tpe) => (value, c).success
            case (None, tpe)        => (p.Term.Select(Nil, p.Named("this", tpe)), c).success
          }
        case (Nil, termArgss, q.TypeApply(term, args)) => // *single* application of some types: `$term[$args...]`
          println(s"[mapper] tpeAp = `${term.show}`")
          for {
            (args)    <- typer0N(args.map(_.tpe))
            (term, c) <- c.mapTerm(term, tpeArgs = args.map(_._2), termArgss = termArgss)
          } yield (term, c)
        case (tpeArgs, termArgs, r: q.Ref) =>
          println(
            s"[mapper] ref = `${r}` termArgs={${termArgs.flatten.map(_.repr).mkString(",")}} tpeArgs=<${tpeArgs.map(_.repr).mkString(",")}>"
          )
          (c.refs.get(r.symbol)) match {
            case Some(term) => (term, (c !! r)).success
            case None       => (c !! r).mapRef0(r, tpeArgs, termArgs)
          }
        case (Nil, Nil, q.New(tpt)) => // new instance *without* arg application: `new $tpt`
          println(s"[mapper] new = `${term.show}`")
          typer0(tpt.tpe).map { (_, tpe) =>
            val name = c.named(tpe)
            (p.Term.Select(Nil, name), (c !! term) ::= p.Stmt.Var(name, None))
          }
        case (Nil, termArgs0, ap @ q.Apply(fun, args)) => // *single* application of some terms: `$fun($args...)`
          println(s"[mapper] ap = `${ap.show}`")
          for {
            (args, c) <- c.down(ap).mapTerms(args)
            (fun, c)  <- (c !! ap).mapTerm(fun, termArgss = args :: termArgs0)
          } yield (fun, c)
        case (Nil, Nil, q.Block(stat, expr)) => // block expression: `{ $stmts...; $expr }`
          for {
            (_, c)   <- (c !! term).mapTrees(stat)
            (ref, c) <- c.mapTerm(expr)
          } yield (ref, c)
        case (Nil, Nil, q.Assign(lhs, rhs)) => // simple assignment: `$lhs = $rhs`
          for {
            (lhsRef, c) <- c.down(term).mapTerm(lhs) // go down here
            (rhsRef, c) <- (c !! term).mapTerm(rhs)
            r <- (lhsRef, rhsRef) match {
              case (s @ p.Term.Select(_, _), rhs) =>
                (
                  p.Term.UnitConst,
                  c ::= p.Stmt.Mut(s, p.Expr.Alias(rhs), copy = false)
                ).success
              case (lhs, rhs) => c.fail(s"Illegal assign LHS,RHS: lhs=${lhs.repr} rhs=$rhs")
            }
          } yield r
        case (Nil, Nil, q.If(cond, thenTerm, elseTerm)) => // conditional: `if($cond) then $thenTerm else $elseTerm`
          for {
            (_, tpe)            <- typer0(term.tpe) // TODO just return the value if result is known at type level
            (condTerm, ifCtx)   <- c.down(term).mapTerm(cond)
            (thenTerm, thenCtx) <- ifCtx.noStmts.mapTerm(thenTerm)
            (elseTerm, elseCtx) <- thenCtx.noStmts.mapTerm(elseTerm)

            _ <-
              (if (condTerm.tpe != p.Type.Bool) s"Cond must be a Bool ref, got ${condTerm}".fail
               else ().success)
            cond <- (thenTerm, elseTerm) match {
              case ((thenTerm), (elseTerm)) if thenTerm.tpe == tpe && elseTerm.tpe == tpe =>
                val name   = ifCtx.named(tpe)
                val result = p.Stmt.Var(name, None)
                val cond = p.Stmt.Cond(
                  p.Expr.Alias(condTerm),
                  thenCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(thenTerm), copy = false),
                  elseCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(elseTerm), copy = false)
                )
                (p.Term.Select(Nil, name), elseCtx.replaceStmts(ifCtx.stmts :+ result :+ cond)).success
              case _ =>
                s"condition unification failure, then=${thenTerm} else=${elseTerm}, expr tpe=${tpe}".fail
            }
          } yield cond
        case (Nil, Nil, q.While(cond, body)) => // loop: `while($cond) {$body...}`
          for {
            (condTerm, condCtx) <- c.noStmts.down(term).mapTerm(cond)
            (_, bodyCtx)        <- condCtx.noStmts.mapTerm(body)
          } yield (
            p.Term.UnitConst,
            bodyCtx.replaceStmts(c.stmts :+ p.Stmt.While(condCtx.stmts, condTerm, bodyCtx.stmts))
          )
        case _ => c.fail(s"Unhandled: `$term`, show=`${term.show}`\nSymbol:\n${term.symbol}")
      }
    }
  }

}
