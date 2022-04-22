package polyregion.scala

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object Remapper {

  import Retyper.*
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

    def mapRef0(ref: q.Ref): Result[(p.Term, q.FnContext)] = c.typer(ref.tpe).flatMap {
      case (Some(term), tpe, c) => (term, c).success
      case (None, tpe, c)       =>
        // call no-arg functions (e.g) directly or pass-through if not no-arg
        def invokeNoArg(sym: q.Symbol, receiver: Option[p.Term], c: q.FnContext): Option[(p.Term, q.FnContext)] =
          sym.tree match {
            case fn: q.DefDef
                if fn.termParamss.isEmpty => // no-arg def call (i.e. `x.toDouble` or just `def fn = ???; fn` )

              val (tpeArgs, rtnTpe) = tpe match {
                case p.Type.Exec(tpeArgs, Nil, rtn) => (tpeArgs.map(p.Type.Var(_)), rtn)
                case p.Type.Exec(_, xs, _)          => ???
                case x                              => (Nil, x)
              }

              val ivk: p.Expr.Invoke = p.Expr.Invoke(p.Sym(sym.fullName), tpeArgs, receiver, Nil, rtnTpe)
              val named              = c.named(rtnTpe)
              val c0 = c
                .down(fn)
                .mark(ivk.signature, fn) ::= p.Stmt.Var(named, Some(ivk))

              Some((p.Term.Select(Nil, named), c0))
            case _ => None
          }

        ref match {
          case ident @ q.Ident(s) => // this is the root of q.Select, it can appear at top-level as well
            // we've encountered a case where the ident's name is different from the TermRef's name
            // this is likely a result of inline where we end up with synthetic names
            // we use the TermRef's name in this case
            val name = ident.tpe match {
              case q.TermRef(_, name) if name != s => name
              case _                               => s
            }
            val local = p.Named(name, tpe)

            // When an ident is owned by a class/object (i.e. not owned by a DefDef) we add an implicit `this` reference;
            // this is valid because the current scope (we're probably a class method) of this ident permits such access.
            // In any other case, we're probably referencing a local ValDef that appeared before before this.
            if (ident.symbol.maybeOwner.isClassDef) {
              c.clsSymTyper(ident.symbol.owner).map { (tpe, c) =>
                val cls = p.Named("this", tpe)
                invokeNoArg(ident.symbol, Some(p.Term.Select(Nil, cls)), c)
                  .getOrElse((p.Term.Select(cls :: Nil, local), c))
              }
            } else invokeNoArg(ident.symbol, None, c).getOrElse((p.Term.Select(Nil, local), c)).success
          case select @ q.Select(qualifierTerm, name) =>
            // fuse with previous select if we got one or simply return the
            c.mapTerm(qualifierTerm).map {
              case (recv @ p.Term.Select(xs, x), c) =>
                invokeNoArg(select.symbol, Some(recv), c).getOrElse((p.Term.Select(xs :+ x, p.Named(name, tpe)), c))
              case (term, c) =>
                // `$qualifierTerm.$name` becomes `$term.${select.symbol}()` so we don't need the `$name` here
                invokeNoArg(select.symbol, Some(term), c) match {
                  case Some(x) => x
                  case None =>
                    ??? // illegal selection of a non DefDef symbol from a primitive term (i.e `{ 1.$name }` )
                }
            }
        }
    }

    def mapTerm(term: q.Term): Result[(p.Term, q.FnContext)] = {
      println(s">>${term.show} = ${c.stmts.size} ~ ${term}")
      term match {
        case q.Typed(x, _)                        => (c !! term).mapTerm(x) // type ascription: `value : T`
        case q.Inlined(call, bindings, expansion) =>
          // for non-inlined args, bindings will contain all relevant arguments with rhs
          // TODO I think call is safe to ignore here? It looks like a subtree from the expansion
          for {
            (_, c) <- (c !! term).mapTrees(bindings)
            (v, c) <- (c !! term).mapTerm(expansion) // simple-inline
          } yield (v, c)
        case q.Literal(q.BooleanConstant(v)) => (p.Term.BoolConst(v), c !! term).pure
        case q.Literal(q.IntConstant(v))     => (p.Term.IntConst(v), c !! term).pure
        case q.Literal(q.FloatConstant(v))   => (p.Term.FloatConst(v), c !! term).pure
        case q.Literal(q.DoubleConstant(v))  => (p.Term.DoubleConst(v), c !! term).pure
        case q.Literal(q.LongConstant(v))    => (p.Term.LongConst(v), c !! term).pure
        case q.Literal(q.ShortConstant(v))   => (p.Term.ShortConst(v), c !! term).pure
        case q.Literal(q.ByteConstant(v))    => (p.Term.ByteConst(v), c !! term).pure
        case q.Literal(q.CharConstant(v))    => (p.Term.CharConst(v), c !! term).pure
        case q.Literal(q.UnitConstant())     => (p.Term.UnitConst, c !! term).pure
        case q.This(_) => // reference to the current class: `this.???`
          c.typer(term.tpe).flatMap {
            case (Some(value), tpe, c) => (value, c).success
            case (None, tpe, c)        => (p.Term.Select(Nil, p.Named("this", tpe)), c).success
          }
        case a @ q.TypeApply(term, args) =>
          // Apply(TypeApply(select, Ts), args...)
          // TypeApply(select, Ts)
          // println("@ "+a.tpe.widenTermRefByName)

          // TODO replace all Type.Var in context subtree with concrete type
          // then, replace all Type.Exec
          c.mapTerm(term).foreach { (term, c) =>

            println("@ " + term)
            println(s"args=${c.stmts.map(_.repr).mkString("\n")}")
            println(s"args=${c.defs.keys}")
          }

          // for {
          //   (_, tpe, c) <- c.typer(a.tpe)
          //   (v, c) <- tpe match {
          //     case ect: q.ErasedFnTpe =>
          //       a.symbol.tree match {
          //         case f: q.DefDef => // (Symbol...).(x: X=>Y)
          //           val sym      = p.Sym(a.symbol.name)
          //           val receiver = p.Sym(a.symbol.maybeOwner.fullName)
          //           println(s"[mapper] type apply of erased fn: ${receiver ~ sym}")
          //           (
          //             q.ErasedMethodVal(receiver, sym, ect, f),
          //             c       /* .mark(receiver ~ sym, f)*/
          //           ).success // ofDim
          //         case _ => ???
          //       }
          //     case bad => c.fail[(q.Val, q.FnContext)](s"Saw ${bad}")
          //   }
          // } yield (v, c)
          ???

        case r: q.Ref =>
          println(s"[mapper] ref @ ${r}")
          (c.refs.get(r)) match {
            case Some(term) => (term, (c !! r)).success
            case None       => (c !! r).mapRef0(r)
          }
        case q.New(tpt) => // new instance *without* arg application: `new $tpt`
          println(s"New=${term.show}")
          (c !! term).typer(tpt.tpe).map {
            case (_, tpe: p.Type, c) =>
              val name = c.named(tpe)
              (p.Term.Select(Nil, name), c ::= p.Stmt.Var(name, None))
            case (_, tpe, c) => ???
          }
        case ap @ q.Apply(fun, args) =>
          val receiverSym        = p.Sym(fun.symbol.fullName)
          val receiverOwner      = fun.symbol.maybeOwner
          val receiverOwnerFlags = receiverOwner.flags

          println(s"A=${args} ${ap.show}")

          // ap.args(0) match {
          //   case q.Block(dd::Nil, _) =>

          //     dd match {
          //       case d : q.DefDef =>
          //         val r = d.rhs.get
          //         println(s"dd=$d")
          //         println(s"r=$r")
          //         println(s"R=${q.Term.betaReduce(r.appliedToArgs (q.Literal(q.IntConstant(1))   ::Nil  ) )}"         )

          //     }

          // }

//          println(s"[mapper] Apply = ${ap}")
          for {

            (_, rtnTpe, c) <- c.typer(ap.tpe)
            (_, funTpe, c) <- c.typer(fun.tpe)
            (argTpes, c)   <- c.typerN(args.map(_.tpe))

            (funVal, c)  <- (c !! ap).mapTerm(fun)
            (argVals, c) <- c.down(ap).mapTerms(args)

            // discard application of erased types for ClassTag[A]
            // argsNoErasedTpe = argTpes.zip(args).flatMap {
            //   case ((_, _: q.ErasedClsTpe), x) => Nil
            //   case ((_, _), x)                 => x :: Nil
            // }
//            _ = println(s"M=${funVal} (...) ")

            // XXX although this behaviour wasn't explicitly documented, it appears that with named args,
            // the compiler will order the args correctly beforehand so we don't have to deal with it here
            argTerms <- argVals.traverse {
              // case q.ErasedNamedArg(_, t: p.Term) => t.success
              case t: p.Term => t.success
              case bad       => c.fail(s"Illegal ${bad}")
            }
//            _ = println("=== " + argTpes)

            mkReturn = (expr: p.Expr, c: q.FnContext) => {

              val name = c.named(expr.tpe)
              (p.Term.Select(Nil, name), c.down(term) ::= p.Stmt.Var(name, Some(expr)))
            }

            //  val ivk: p.Expr.Invoke = p.Expr.Invoke(sym, Nil, Some(receiver), argTerms, t)
            // mkReturn(ivk, c.mark(ivk.signature, underlying)).success

            // _ = println(s"saw apply -> ${ap.symbol.tree.show}")
            // _ = println(s"inner is  -> ${defdef.show}")
            // _ = println(s"inner is  -> ${funVal} AP ${argTerms}")

            _ = println(s"[mapper] apply function value: ${funVal}")

//             (ref, c) <- (argTerms, funVal) match {
//               case (Nil, x) => (x, c).success
//               case (_, q.ErasedMethodVal(receiver: p.Term.Select, p.Sym("<init>" :: Nil), fnTpe, _)) => // ctor call
//                 (for {
//                   sdef <- (fnTpe.rtn match {
//                     case p.Type.Struct(s, _) => c.clss.get(s)
//                     case _                   => None
//                   }).failIfEmpty(s"No StructDef found for type ${fnTpe.rtn}")
//                   _ = println(s"[mapper] Ctor !")

//                   // we need to also make sure the ctor has no impl here
//                   // that would mean it's in the form of `class X(val field : Y)`
//                   structTpes  = sdef.members.map(_.tpe)
//                   ctorArgTpes = fnTpe.args

// //                  a = ctorArgTpes.x
//                   _ <-
//                     if (structTpes == argTpes.map(_._2) && structTpes == ctorArgTpes.map(_._2)) ().success
//                     else
//                       s"Ctor args mismatch, class ${sdef.name} expects ${structTpes}, fn expects ${ctorArgTpes} and was applied with ${argTpes}".fail

//                   setMemberExprs = sdef.members.zip(argTerms).map { (member, value) =>
//                     p.Stmt.Mut(p.Term.Select(receiver.init :+ receiver.last, member), p.Expr.Alias(value), copy = false)
//                   }
//                 } yield (receiver, c.::=(setMemberExprs*)))
//               case (_, m @ q.ErasedMethodVal(module: p.Sym, sym, tpe, underlying)) => // module call
//                 val t = tpe.rtn match {
//                   case t: p.Type => t
//                   case q.ErasedFnTpe(args, rtn: p.Type) if args.forall {
//                         case (_, _: q.ErasedClsTpe) => true
//                         case _                      => false
//                       } =>
//                     rtn

//                   //
//                   case x: q.ErasedClsTpe =>
//                     println(x)
//                     // x
//                     ???
//                 } // TODO handle multiple arg list methods
//                 val ivk: p.Expr.Invoke = p.Expr.Invoke(module ~ sym, Nil, None, argTerms, t)
//                 mkReturn(ivk, c.mark(ivk.signature, underlying)).success
//               case (_, q.ErasedMethodVal(receiver: p.Term, sym, tpe, underlying)) => // instance call
//                 val t = tpe.rtn match {
//                   case t: p.Type => t
//                   case _         => ???
//                 }
//                 val ivk: p.Expr.Invoke = p.Expr.Invoke(sym, Nil, Some(receiver), argTerms, t)
//                 mkReturn(ivk, c.mark(ivk.signature, underlying)).success
//             }

          } yield ??? // (ref, c)
        case q.Block(stat, expr) => // block expression: `{ $stmts...; $expr }`
          for {
            (_, c)   <- (c !! term).mapTrees(stat)
            (ref, c) <- c.mapTerm(expr)
          } yield (ref, c)
        case q.Assign(lhs, rhs) => // assignment: `$lhs = $rhs`
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
        case q.If(cond, thenTerm, elseTerm) => // conditional: `if($cond) then { $thenTerm } else { $elseTerm }`
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
        case q.While(cond, body) => // loop: `while($cond) {$body...}`
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
