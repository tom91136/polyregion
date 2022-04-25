package polyregion.scala

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object Remapper {

  import Retyper.*

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

  extension (using q: Quoted)(c: q.FnContext) {

    def mapTrees(args: List[q.Tree]): Result[(p.Term, q.FnContext)] = args match {
      case Nil => (p.Term.UnitConst, c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show.replace("\n", " ; ")))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show.replace("\n", " ; "))).mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Result[(p.Term, q.FnContext)] = tree match {
      case q.ValDef(name, tpeTree, Some(rhs)) =>
        for {
          (term, tpe, c) <- c.typer(tpeTree.tpe)
          // if tpe is singleton, substitute with constant directly
          (ref, c) <- term.fold((c !! tree).mapTerm(rhs))(x => (x, c).success)

        } yield (
          p.Term.UnitConst,
          (ref, tpe) match {
            case (term: p.Term, tpe: p.Type) => c ::= p.Stmt.Var(p.Named(name, tpe), Some(p.Expr.Alias(term)))
            // case (v: q.ErasedMethodVal, tpe: q.ErasedFnTpe) => c.suspend(name -> tpe)(v)
            case (ref, tpe) =>
              println(s"tree= ${tree.show}")
              println(s"ref= $ref")
              println(s"tpe= $tpe")
              ???
          }
        )
      case q.ValDef(name, tpe, None) => c.fail(s"Unexpected variable $name:$tpe")
      // DefDef here comes from general closures ( (a:A) => ??? )
      case t: q.Term => (c !! tree).mapTerm(t)

      // def f(x...) : A = ??? === val f : x... => A = ???

      case tree => c.fail(s"Unhandled: $tree\nSymbol:\n${tree.symbol}")

    }

    def mapTerms(args: List[q.Term]): Result[(List[p.Term], q.FnContext)] = args match {
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
    ) = (fn.symbol, fn.rhs, receiver) match {
      // We handle synthesis of the primary ctor here: primary ctors have the following invariant:
      //  * `fn.name` == "<init>"
      //  * `fn.rhs` is empty
      //  * `receiver` is not an non-empty select and matches the owning class
      case (sym, None, Some(instance)) if sym.maybeOwner.primaryConstructor == sym && sym.name == "<init>" =>
        // For primary ctors, we synthesise the assignment of ctor args to the corresponding fields
        val fields = fn.termParamss.flatMap(_.params)
        for {
          (fieldTpes, c) <- c.typerN(fields.map(_.tpt.tpe))
          fieldNames = fields.map(_.name)
          _ <- if (args.map(_.tpe) != fieldTpes.map(_._2)) "Ctor application type mismatch".fail else ().success
          instancePath <- instance match {
            case p.Term.Select(xs, x) => (xs :+ x).success
            case _                    => "Ctor invocation on instance must be a Select term".fail
          }
          stmts = fieldNames.zip(args).map { (name, rhs) =>
            p.Stmt.Mut(p.Term.Select(instancePath, p.Named(name, rhs.tpe)), p.Expr.Alias(rhs), copy = false)
          }
        } yield (instance, c.::=(stmts*))
      case (sym, _, _) => // Anything else is a normal invoke.
        val ivk: p.Expr.Invoke = p.Expr.Invoke(p.Sym(sym.fullName), tpeArgs, receiver, args, rtnTpe)
        val named              = c.named(rtnTpe)
        val c0 = c
          .down(fn)
          .mark(ivk.signature, fn) ::= p.Stmt.Var(named, Some(ivk))
        (p.Term.Select(Nil, named), c0).success
    }

    // The general idea is that idents are either used as-is or it goes through a series of type/term applications (e.g `{ foo.fn[A](b)(c) }`).
    // We collect the types and term lists in `mapTerm` and then fully apply it here.
    private def mapRef0(
        ref: q.Ref,
        tpeArgs: List[p.Type],
        termArgss: List[List[p.Term]]
    ): Result[(p.Term, q.FnContext)] =
      c.typer(ref.tpe).flatMap {
        case (Some(term), tpe, c) => (term, c).success
        case (None, tpe0, c)      =>
          // Apply any unresolved type vars first.
          val tpe = tpe0 match {
            case exec: p.Type.Exec => fullyApplyGenExec(exec, tpeArgs)
            case x                 => x
          }
          // call no-arg functions (i.e. `x.toDouble` or just `def fn = ???; fn` ) directly or pass-through if not no-arg
          def invoke(sym: q.Symbol, receiver: Option[p.Term], c: q.FnContext) = sym.tree match {
            case fn: q.DefDef =>
              // Assert that the term list matches Exec's nested (recursive) types.
              // Note that Exec treats both empty args `()` and no-args as `Nil` where as the collected arg lists through
              // `Apply` will give empty args as `Nil` and not collect no-args at all because no no application took place.
              val termTpess = termArgss.map(_.map(_.tpe))
              val execTpess = collectExecArgLists(tpe)
              (fn.termParamss.isEmpty, termTpess, execTpess) match {
                case (true, Nil, (Nil :: Nil) | Nil) => ()  // no-ap, no-arg Exec (`Nil::Nil`) or no Exec at all (`Nil`)
                case (false, ts, es) if ts == es     => ()  // everything else, do the assertion
                case (ap, ts, es)                    => ??? // TODO raise failure
              }
              c.mkInvoke(fn, tpeArgs, receiver, termArgss.flatten, resolveExecRtnTpe(tpe)).map(Some(_))
            case _ => None.success
          }
          ref match {
            case ident @ q.Ident(s) => // this is the root of q.Select, it can appear at top-level as well
              // We've encountered a case where the ident's name is different from the TermRef's name.
              // This is likely a result of inline where we end up with synthetic names, we use the TermRef's name in this case.
              val name = ident.tpe match {
                case q.TermRef(_, name) if name != s => name
                case _                               => s
              }
              val local = p.Named(name, tpe)
              // When an ident is owned by a class/object (i.e. not owned by a DefDef) we add an implicit `this` reference;
              // this is valid because the current scope (we're probably a class method) of this ident permits such access.
              // In any other case, we're probably referencing a local ValDef that appeared before before this.
              if (ident.symbol.maybeOwner.isClassDef) {
                for {
                  (tpe, c) <- c.clsSymTyper(ident.symbol.owner)
                  cls = p.Named("this", tpe)
                  invoke <- invoke(ident.symbol, Some(p.Term.Select(Nil, cls)), c)
                } yield invoke.getOrElse((p.Term.Select(cls :: Nil, local), c))
              } else invoke(ident.symbol, None, c).map(_.getOrElse((p.Term.Select(Nil, local), c)))
            case select @ q.Select(qualifierTerm, name) =>
              c.mapTerm(qualifierTerm).flatMap {
                case (recv @ p.Term.Select(xs, x), c) => // fuse with previous select if we got one
                  invoke(select.symbol, Some(recv), c).map(_.getOrElse((p.Term.Select(xs :+ x, p.Named(name, tpe)), c)))
                case (term, c) => // or simply return whatever it's referring to
                  // `$qualifierTerm.$name` becomes `$term.${select.symbol}()` so we don't need the `$name` here
                  invoke(select.symbol, Some(term), c).flatMap {
                    case Some(x) => x.success
                    case None =>
                      "illegal selection of a non DefDef symbol from a primitive term (i.e `{ 1.$name }` )".fail
                  }
              }
          }
      }

    def mapTerm(
        term: q.Term,
        tpeArgs: List[p.Type] = Nil,
        termArgss: List[List[p.Term]] = Nil
    ): Result[(p.Term, q.FnContext)] = {
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
          c.typer(term.tpe).flatMap {
            case (Some(value), tpe, c) => (value, c).success
            case (None, tpe, c)        => (p.Term.Select(Nil, p.Named("this", tpe)), c).success
          }
        case (Nil, termArgss, q.TypeApply(term, args)) => // *single* application of some types: `$term[$args...]`
          println(s"[mapper] tpeAp = `${term.show}`")
          for {
            (args, c) <- c.typerN(args.map(_.tpe))
            (term, c) <- c.mapTerm(term, tpeArgs = args.map(_._2), termArgss = termArgss)
          } yield (term, c)
        case (tpeArgs, termArgs, r: q.Ref) =>
          println(s"[mapper] ref = `${r.show}` <${tpeArgs.map(_.repr).mkString(",")}>")
          (c.refs.get(r)) match {
            case Some(term) => (term, (c !! r)).success
            case None       => (c !! r).mapRef0(r, tpeArgs, termArgs)
          }
        case (Nil, Nil, q.New(tpt)) => // new instance *without* arg application: `new $tpt`
          println(s"[mapper] new = `${term.show}`")
          (c !! term).typer(tpt.tpe).map { (_, tpe, c) =>
            val name = c.named(tpe)
            (p.Term.Select(Nil, name), c ::= p.Stmt.Var(name, None))
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
              case (s @ p.Term.Select(Nil, _), rhs) =>
                (
                  p.Term.UnitConst,
                  c ::= p.Stmt.Mut(s, p.Expr.Alias(rhs), copy = false)
                ).success
              case bad => c.fail(s"Illegal assign LHS,RHS: ${bad}")
            }
          } yield r
        case (Nil, Nil, q.If(cond, thenTerm, elseTerm)) => // conditional: `if($cond) then $thenTerm else $elseTerm`
          for {
            (_, tpe, c)         <- c.typer(term.tpe) // TODO just return the value if result is known at type level
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
