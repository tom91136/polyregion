package polyregion.compiler

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*

object TreeMapper {

  import Retyper.*
  extension (using q: Quoted)(c: q.FnContext) {

    def mapTrees(args: List[q.Tree]): Deferred[(q.Value, q.FnContext)] = args match {
      case Nil => (q.Value(p.Term.UnitConst), c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show)).mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Deferred[(q.Value, q.FnContext)] = tree match {
      case q.ValDef(name, tpe, Some(rhs)) =>
        for {
          (term, t, c) <- c.typer(tpe.tpe)
          // if tpe is singleton, substitute with constant directly
          (ref, c) <- term.fold((c !! tree).mapTerm(rhs))(x => (q.Value(x), c).pure)
          c2 = ref.actual match {
            case term: p.Term         => c ::= p.Stmt.Var(p.Named(name, t), Some(p.Expr.Alias(term)))
            case partial: q.Suspended => c.suspend(p.Term.Select(Nil, p.Named(name, t)))(partial)
          }
        } yield (q.Value(p.Term.UnitConst), c2)
      case q.ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
      // DefDef here comes from general closures ( (a:A) => ??? )
      case t: q.Term => (c !! tree).mapTerm(t)

      // def f(x...) : A = ??? === val f : x... => A = ???

      case tree =>
        s"[depth=${c.depth}] Unhandled: $tree\nSymbol:\n${tree.symbol}\nTrace was:\n${(tree :: c.traces)
          .map(x => "\t" + x.show + "\n\t" + x)
          .mkString("\n---\n")}".fail.deferred
    }

    def mapTerms(args: List[q.Term]): Deferred[(List[q.Value], q.FnContext)] = args match {
      case Nil => (Nil, c).pure
      case x :: xs =>
        c.mapTerm(x)
          .map((ref, c) => (ref :: Nil, c))
          .flatMap(xs.foldLeftM(_) { case ((rs, c0), term) =>
            c0.mapTerm(term).map((r, c) => (rs :+ r, c))
          })
    }

    def mapTerm(term: q.Term): Deferred[(q.Value, q.FnContext)] = term match {
      case q.Typed(x, _)                        => (c !! term).mapTerm(x)
      case q.Inlined(call, bindings, expansion) => (c !! term).mapTerm(expansion) // simple-inline
      case q.Literal(q.BooleanConstant(v))      => (q.Value(p.Term.BoolConst(v)), c !! term).pure
      case q.Literal(q.IntConstant(v))          => (q.Value(p.Term.IntConst(v)), c !! term).pure
      case q.Literal(q.FloatConstant(v))        => (q.Value(p.Term.FloatConst(v)), c !! term).pure
      case q.Literal(q.DoubleConstant(v))       => (q.Value(p.Term.DoubleConst(v)), c !! term).pure
      case q.Literal(q.LongConstant(v))         => (q.Value(p.Term.LongConst(v)), c !! term).pure
      case q.Literal(q.ShortConstant(v))        => (q.Value(p.Term.ShortConst(v)), c !! term).pure
      case q.Literal(q.ByteConstant(v))         => (q.Value(p.Term.ByteConst(v)), c !! term).pure
      case q.Literal(q.CharConstant(v))         => (q.Value(p.Term.CharConst(v)), c !! term).pure
      case q.Literal(q.UnitConstant())          => (q.Value(p.Term.UnitConst), c !! term).pure
      case r: q.Ref =>
        (c.refs.get(r.symbol), r) match {
          case (Some(q.Reference(value, tpe)), _) =>
            val term = value match {
              case name: String => p.Term.Select(Nil, p.Named(name, tpe))
              case term: p.Term => term
            }
            (q.Value(term), c !! r).pure
          case (None, i @ q.Ident(s)) =>
            val name = i.tpe match {
              // we've encountered a case where the ident's name is different from the TermRef's name
              // this is likely a result of inline where we end up with synthetic names
              // we use the TermRefs name in this case
              case q.TermRef(_, name) if name != s => name
              case _                               => s
            }
            c.typer(i.tpe).map { (_, tpe, c) =>
              (q.Value(p.Term.Select(Nil, p.Named(name, tpe))), c.!!(r))
            }
          case (None, s @ q.Select(root, name)) =>
            for {
              (_, tpe, c) <- c.typer(s.tpe)
              (lhsRef, c) <- (c !! term).mapTerm(root)
              ref <- lhsRef match {
                case (p.Term.Select(xs, x)) =>
                  p.Term.Select(xs :+ x, p.Named(name, tpe)).success.deferred
                case bad => s"illegal select root ${bad}".fail.deferred
              }
            } yield (q.Value(ref), c)

          case (None, x) =>
            s"[depth=${c.depth}] Ref ${x} with tpe=${x.tpe} was not identified at closure args stage, ref pool:\n->${c.refs
              .mkString("\n->")} ".fail.deferred

        }

      case ap @ q.Apply(_, _) =>
        val receiverSym        = p.Sym(ap.fun.symbol.fullName)
        val receiverOwner      = ap.fun.symbol.maybeOwner
        val receiverOwnerFlags = receiverOwner.flags

        //
        println(s"A=${ap.args}")

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

        for {
          (_, tpe, c)  <- c.typer(ap.tpe)
          (funRef, c)  <- c.mapTerm(ap.fun)
          (argRefs, c) <- c.down(ap).mapTerms(ap.args)

          _ = funRef.actual match {
            case t: p.Term         => 
            case part: q.Suspended =>
          }

          defdef <- ap.fun.symbol.tree match {
            case d: q.DefDef => d.success.deferred // if we see this then the call is probably fully applied at the end
            case bad         => s"Unexpected ap symbol ${bad.show}".fail.deferred
          }

          _ = println(s"receiverFlags:${receiverSym} = ${receiverOwnerFlags.show} (${receiverOwner})")

          // method calls on a module
          // * Symbol.companionClass gives the case class symbol if this is a module, otherwise Symbol.isNoSymbol
          // * Symbol.companionClass gives the module symbol if this is a case class, otherwise Symbol.isNoSymbol
          // * Symbol.companionModule on a module gives the val def of the singleton

          mkReturn = (expr: p.Expr, c: q.FnContext) => {
            val name = c.named(tpe)
            (p.Term.Select(Nil, name), c ::= p.Stmt.Var(name, Some(expr)))
          }

          _ = println(s"saw apply -> ${ap.symbol.tree.show}")
          _ = println(s"inner is  -> ${defdef.show}")

          withoutErasedTerms = argRefs
          // .filter(_.tpe match {
          //   case p.Type.Erased(_, _) => false
          //   case _                   => true
          // })

          (ref, c) <-
            if (receiverOwnerFlags.is(q.Flags.Module)) // Object.x(ys)(erased?)
              mkReturn(
                p.Expr.Invoke(receiverSym, None, withoutErasedTerms, tpe),
                c.mark(receiverSym, defdef)
              ).success.deferred
            else
              ap.fun match {
                case q.Select(q.New(tt), "<init>") => // new X
                  ap.symbol.tree match {
                    case q.DefDef(_, _, _, None) =>
                      // println(s"impl=${ap.symbol.tree.asInstanceOf[q.DefDef].rhs}")
                      println(s"args=${ap.args.map(_.tpe.dealias.widenTermRefByName)}")

                      (for {
                        sdef <- (tpe match {
                          case p.Type.Struct(s) => c.clss.get(s)
                          case _                => None
                        }).failIfEmpty(s"No StructDef found for type $tpe")

                        structTpes = sdef.members.map(_.tpe)
                        argTpes    = withoutErasedTerms.map(_.tpe)

                        name = p.Named(s"v${c.depth}_new", tpe)
                        expr = p.Stmt.Var(name, None)

                        _ <-
                          if (structTpes == argTpes) ().success
                          else s"Ctor args mismatch, class expects ${sdef.members} but was given ${argTpes}".fail

                        setMemberExprs = sdef.members.zip(withoutErasedTerms).map { (member, value) =>
                          p.Stmt.Mut(p.Term.Select(name :: Nil, member), p.Expr.Alias(value), copy = false)
                        }
                      } yield ((p.Term.Select(Nil, name)), c.::=(expr +: setMemberExprs*))).deferred
                    case x => s"Found ctor signature, expecting def with no rhs but got: $x".fail.deferred
                  }
                case s @ q.Select(q, n) => // s.y(zs)
                  (c !! s)
                    .mark(receiverSym, defdef)
                    .mapTerm(q)
                    .map((receiverRef, c) =>
                      mkReturn(p.Expr.Invoke(receiverSym, Some(receiverRef), withoutErasedTerms, tpe), c)
                    )
                case _ => ??? // (ctx.depth, None, Nil).success.deferred
              }
        } yield (q.Value(ref), c)
      case q.Block(stat, expr) =>
        for {
          (_, c)   <- (c !! term).mapTrees(stat)
          (ref, c) <- c.mapTerm(expr)
        } yield (ref, c)
      case q.Assign(lhs, rhs) =>
        for {
          (lhsRef, c) <- c.down(term).mapTerm(lhs) // go down here
          (rhsRef, c) <- (c !! term).mapTerm(rhs)
          r <- (lhsRef, rhsRef) match {
            case (s @ p.Term.Select(Nil, _), rhs) =>
              (
                q.Value(p.Term.UnitConst),
                rhs.actual match {
                  case t: p.Term      => c ::= p.Stmt.Mut(s, p.Expr.Alias(t), copy = false)
                  case p: q.Suspended => c.suspend(s)(p)
                }
              ).pure
            case bad => s"Illegal assign LHS,RHS: ${bad}".fail.deferred
          }
        } yield r
      case q.If(cond, thenTerm, elseTerm) =>
        for {
          (_, tpe, c)        <- c.typer(term.tpe) // TODO just return the value if result is known at type level
          (condVal, ifCtx)   <- c.down(term).mapTerm(cond)
          (thenVal, thenCtx) <- ifCtx.noStmts.mapTerm(thenTerm)
          (elseVal, elseCtx) <- thenCtx.noStmts.mapTerm(elseTerm)

          // cond, then, else, must be fully applied here
          failIfPartial = (name: String, v: q.Value) =>
            v.actual match {
              case t: p.Term => t.success.deferred
              case partial: q.Suspended =>
                s"$name term of an if-then-else statement (${term.show}) is partial ($partial)".fail.deferred
            }

          condRef <- failIfPartial("condition", condVal)
          thenRef <- failIfPartial("then", thenVal)
          elseRef <- failIfPartial("else", elseVal)

          _ <- (if (condRef.tpe != p.Type.Bool) s"Cond must be a Bool ref, got ${condRef}".fail
                else ().success).deferred
          cond <- (thenRef, elseRef) match {
            case ((thenRef), (elseRef)) if thenRef.tpe == tpe && elseRef.tpe == tpe =>
              val name   = ifCtx.named(tpe)
              val result = p.Stmt.Var(name, None)
              val cond = p.Stmt.Cond(
                p.Expr.Alias(condRef),
                thenCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(thenRef), copy = false),
                elseCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(elseRef), copy = false)
              )
              (q.Value(p.Term.Select(Nil, name)), elseCtx.replaceStmts(ifCtx.stmts :+ result :+ cond)).success.deferred
            case _ =>
              s"condition unification failure, then=${thenRef} else=${elseRef}, expr tpe=${tpe}".fail.deferred
          }
        } yield cond
      case q.While(cond, body) =>
        for {
          (condVal, condCtx) <- c.noStmts.down(term).mapTerm(cond)
          (_, bodyCtx)       <- condCtx.noStmts.mapTerm(body)
          condRef <- condVal match {
            case t: p.Term => t.success.deferred
            case partial: q.Suspended =>
              s"condition term of a while expression (${term.show}) is partial ($partial)".fail.deferred
          }

        } yield {
          val block = condCtx.stmts match {
            case Nil                              => ??? // this is illegal, while needs a bool predicate
            case p.Stmt.Var(_, Some(cond)) :: Nil =>
              // simple condition:
              // var cond := true
              // while(cond) { ...; cond := false }
              p.Stmt.While(cond, bodyCtx.stmts)
            case xs =>
              // complex condition:
              // while(true) {  stmts...; if(!condRef) break;  }
              println(">>>>>" + term.show)
              println(">>>>>" + condRef)
              println(">>>>>" + xs)
              ???

              val body = (xs :+ p.Stmt.Cond(
                p.Expr.Alias(condRef),
                Nil,
                p.Stmt.Break :: Nil
              )) ++ bodyCtx.stmts

              p.Stmt.While(p.Expr.Alias(p.Term.BoolConst(true)), body)
          }
          (q.Value(p.Term.UnitConst), bodyCtx.replaceStmts(c.stmts :+ block))
        }
      case _ =>
        s"[depth=${c.depth}] Unhandled: $term\nSymbol:\n${term.symbol}\nTrace was:\n${(term :: c.traces)
          .map(x => "\t" + x.show + "\n\t" + x)
          .mkString("\n---\n")}".fail.deferred
    }

  }

}
