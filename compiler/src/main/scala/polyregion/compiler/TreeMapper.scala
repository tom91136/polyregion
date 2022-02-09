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

    def mapTrees(args: List[q.Tree]): Deferred[(q.Val, q.FnContext)] = args match {
      case Nil => (p.Term.UnitConst, c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show)).mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Deferred[(q.Val, q.FnContext)] = tree match {
      case q.ValDef(name, tpeTree, Some(rhs)) =>
        for {
          (term, tpe, c) <- c.typer(tpeTree.tpe)
          // if tpe is singleton, substitute with constant directly
          (ref, c) <- term.fold((c !! tree).mapTerm(rhs))(x => (x, c).pure)
        } yield (
          p.Term.UnitConst,
          (ref, tpe) match {
            case (term: p.Term, tpe: p.Type) => c ::= p.Stmt.Var(p.Named(name, tpe), Some(p.Expr.Alias(term)))
            case (v: q.ErasedClosureVal, tpe: q.ErasedClosureTpe) => c.suspend(name -> tpe)(v)
          }
        )
      case q.ValDef(name, tpe, None) => c.fail(s"Unexpected variable $name:$tpe")
      // DefDef here comes from general closures ( (a:A) => ??? )
      case t: q.Term => (c !! tree).mapTerm(t)

      // def f(x...) : A = ??? === val f : x... => A = ???

      case tree => c.fail(s"Unhandled: $tree\nSymbol:\n${tree.symbol}")

    }

    def mapTerms(args: List[q.Term]): Deferred[(List[q.Val], q.FnContext)] = args match {
      case Nil => (Nil, c).pure
      case x :: xs =>
        c.mapTerm(x)
          .map((ref, c) => (ref :: Nil, c))
          .flatMap(xs.foldLeftM(_) { case ((rs, c0), term) =>
            c0.mapTerm(term).map((r, c) => (rs :+ r, c))
          })
    }

    def mapTerm(term: q.Term): Deferred[(q.Val, q.FnContext)] = term match {
      case q.Typed(x, _)                        => (c !! term).mapTerm(x)
      case q.Inlined(call, bindings, expansion) => (c !! term).mapTerm(expansion) // simple-inline
      case q.Literal(q.BooleanConstant(v))      => (p.Term.BoolConst(v), c !! term).pure
      case q.Literal(q.IntConstant(v))          => (p.Term.IntConst(v), c !! term).pure
      case q.Literal(q.FloatConstant(v))        => (p.Term.FloatConst(v), c !! term).pure
      case q.Literal(q.DoubleConstant(v))       => (p.Term.DoubleConst(v), c !! term).pure
      case q.Literal(q.LongConstant(v))         => (p.Term.LongConst(v), c !! term).pure
      case q.Literal(q.ShortConstant(v))        => (p.Term.ShortConst(v), c !! term).pure
      case q.Literal(q.ByteConstant(v))         => (p.Term.ByteConst(v), c !! term).pure
      case q.Literal(q.CharConstant(v))         => (p.Term.CharConst(v), c !! term).pure
      case q.Literal(q.UnitConstant())          => (p.Term.UnitConst, c !! term).pure
      case r: q.Ref =>
        (c.refs.get(r.symbol), r) match {
          case (Some(q.Reference(value, tpe)), _) =>
            val term = (value, tpe) match {
              case (name: String, tpe: p.Type) => p.Term.Select(Nil, p.Named(name, tpe))
              case (term: p.Term, _)           => term
              case _                           => ???
            }
            (term, c !! r).pure
          case (None, i @ q.Ident(s)) =>
            val name = i.tpe match {
              // we've encountered a case where the ident's name is different from the TermRef's name
              // this is likely a result of inline where we end up with synthetic names
              // we use the TermRefs name in this case
              case q.TermRef(_, name) if name != s => name
              case _                               => s
            }
            println(s"$i = $s")
            for {
              (_, tpe, c) <- c.typer(i.tpe)
              // Ident is local, so if it is an erased closure type, we check the context first to see if we have a defined suspension
              // for types that are already part of PolyAst, we just use it as is
              term <- tpe match {
                case tpe: p.Type => p.Term.Select(Nil, p.Named(name, tpe)).success.deferred
                case ect: q.ErasedClosureTpe =>
                  c.suspended.get(name -> ect) match {
                    case Some(x) => x.success.deferred
                    case None    => c.fail[q.Val](s"Can't find a previous definition of ${i}")
                  }
                case q.ErasedTpe(sym, true, Nil) => q.ErasedModuleSelect(sym).success.deferred
                case et: q.ErasedTpe             => c.fail[q.Val](s"Saw ${et}")
              }
            } yield (term, c !! r)
          case (None, s @ q.Select(root, name)) =>
            println(s"sel $s")
            for {
              (_, tpe, c)  <- c.typer(s.tpe)
              (rootVal, c) <- (c !! term).mapTerm(root)
              _ = println(s"Clear sel $s, tpe=${tpe}")

              // for selects, we got two cases
              // (a:p.Type).(b:p.Type).(c:p.Type)
              //   - This is the simple case where we just lift it to a PolyAst Select
              // (a:p.Type).(b:p.Type).(f:ErasedClosureTpe(X..., Y))
              //   - This gets lifted into an ErasedClosureVal using the correct name for immediate return as it's likely a foreign module call.
              //     This sort of select could be
              // it is an error if the select chain ends up with a Erased* type in the middle, we only expect it to appear last
              (term, c) <- tpe match {
                case tpe: p.Type =>
                  rootVal match {
                    case p.Term.Select(xs, x) => (p.Term.Select(xs :+ x, p.Named(name, tpe)), c).success.deferred
                    case q.ErasedModuleSelect(module) =>
                      s.symbol.tree match {
                        case dd: q.DefDef
                            if dd.paramss.isEmpty => // no-arg module def call (i.e. `object X{ def y :Int = ??? }` )
                          val fnSym = module :+ name
                          val named = c.named(tpe)
                          (
                            p.Term.Select(Nil, named),
                            c.mark(fnSym, dd) ::= p.Stmt.Var(named, Some(p.Expr.Invoke(fnSym, None, Nil, tpe)))
                          ).success.deferred
                        case dd: q.DefDef =>
                          c.fail(s"Unexpected arg list ${dd.paramss} for a 0-arg def via module ref ${module.repr}")
                        case vd: q.ValDef =>
                          c.fail(s"$vd via module ref ${module.repr} was not intercepted by the outliner!?")
                        case bad =>
                          c.fail(s"Unsupported construct $bad via module ref ${module.repr}")
                      }
                    case bad => c.fail(s"Unexpected root of a select that leads to a non-erased ${tpe} type: ${bad}")
                  }

                case ect: q.ErasedClosureTpe =>
                  println(">>>>" + (c !! term).mapTerm(root).resolve.map(_._1))

                  println(s"$ect $name")

                  // a.f(_)(_)
                  // ect.rtn : Term | ECV
                  // case Term => Invoke
                  // case ECV  => ???


                  rootVal match {
                    case receiver: p.Term =>
                      q.ErasedClosureVal(
                        ect.args.zipWithIndex.map((x, i) => (s"_$i", x)),
                        ect.rtn,
                        List(p.Stmt.Return(p.Expr.Invoke(p.Sym(name), Some(receiver), ???, ???)))
                      )
                    case q.ErasedModuleSelect(mod) =>
                      q.ErasedClosureVal(
                        ect.args.zipWithIndex.map((x, i) => (s"_$i", x)),
                        ect.rtn,
                        List(p.Stmt.Return(p.Expr.Invoke(mod :+ name, None, ???, ???)))
                      )
                    case bad => c.fail(s"Unexpected root of a select that leads to an erased ${ect} type: ${bad}")
                  }

                  ???

                // rootVal
                // case q.ErasedTpe(sym, true, Nil) => (q.ErasedModuleSelect(sym), c).success.deferred
                case et: q.ErasedTpe => c.fail[(q.Val, q.FnContext)](s"Saw ${et}")
              }

            } yield (term, c)

          case (None, x) => c.fail(s"Ref ${x} with tpe=${x.tpe} was not identified at closure args stage")

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
          (_, rtnTpe, c) <- c.typer(ap.tpe)
          (_, funTpe, c) <- c.typer(ap.fun.tpe)
          _ = println(s"B=$funTpe")
          (funVal, c)  <- c.mapTerm(ap.fun)
          (argVals, c) <- c.down(ap).mapTerms(ap.args)

          argTerms <- argVals.traverse {
            case t: p.Term => t.success.deferred
            case bad       => c.fail(s"Illegal ${bad}")
          }

          // _ = (funVal, tpe) match {
          //   case (term: p.Term, tpe: p.Type)                                   =>
          //   case (closure: q.ErasedClosureVal, closureTpe: q.ErasedClosureTpe) =>
          //   case (_, _)                                                        => ???
          // }

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
            val name = c.named(???)
            (p.Term.Select(Nil, name), c ::= p.Stmt.Var(name, Some(expr)))
          }

          _ = println(s"saw apply -> ${ap.symbol.tree.show}")
          _ = println(s"inner is  -> ${defdef.show}")

          withoutErasedTerms = argVals
          // .filter(_.tpe match {
          //   case p.Type.Erased(_, _) => false
          //   case _                   => true
          // })

          (ref, c) <-
            if (receiverOwnerFlags.is(q.Flags.Module)) // Object.x(ys)(erased?)
              mkReturn(
                p.Expr.Invoke(receiverSym, None, ???, ???),
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
                        sdef <- (rtnTpe match {
                          case p.Type.Struct(s) => c.clss.get(s)
                          case _                => None
                        }).failIfEmpty(s"No StructDef found for type $rtnTpe")

                        structTpes = sdef.members.map(_.tpe)
                        argTpes    = withoutErasedTerms.map(x => ???)

                        name = p.Named(s"v${c.depth}_new", ???)
                        expr = p.Stmt.Var(name, None)

                        _ <-
                          if (structTpes == argTpes) ().success
                          else s"Ctor args mismatch, class expects ${sdef.members} but was given ${argTpes}".fail

                        setMemberExprs = sdef.members.zip(withoutErasedTerms).map { (member, value) =>
                          p.Stmt.Mut(p.Term.Select(name :: Nil, member), p.Expr.Alias(???), copy = false)
                        }
                      } yield ((p.Term.Select(Nil, name)), c.::=(expr +: setMemberExprs*))).deferred
                    case x => s"Found ctor signature, expecting def with no rhs but got: $x".fail.deferred
                  }
                case s @ q.Select(q, n) =>
                  // up

                  // s.y(zs)
                  (c !! s)
                    .mark(receiverSym, defdef)
                    .mapTerm(q)
                    .map((receiverRef, c) =>
                      mkReturn(p.Expr.Invoke(receiverSym, Some(receiverRef.asInstanceOf[p.Term]), argTerms, ???), c)
                    )
                case _ => ??? // (ctx.depth, None, Nil).success.deferred
              }
        } yield (ref, c)
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
            case (lhsClosure: q.ErasedClosureVal, rhsClosure: q.ErasedClosureVal) =>
              if (lhsClosure.tpe != rhsClosure.tpe) c.fail(s"Assignment of incompatible type ${lhsRef} := ${rhsRef}")
              else ???

            case (s @ p.Term.Select(Nil, _), rhs) =>
              (
                p.Term.UnitConst,
                rhs match {
                  case t: p.Term             => c ::= p.Stmt.Mut(s, p.Expr.Alias(t), copy = false)
                  case p: q.ErasedClosureVal => ??? // c.suspend(s)(p)
                }
              ).pure
            case bad => c.fail(s"Illegal assign LHS,RHS: ${bad}")
          }
        } yield r
      case q.If(cond, thenTerm, elseTerm) =>
        for {
          (_, tpe, c)        <- c.typer(term.tpe) // TODO just return the value if result is known at type level
          (condVal, ifCtx)   <- c.down(term).mapTerm(cond)
          (thenVal, thenCtx) <- ifCtx.noStmts.mapTerm(thenTerm)
          (elseVal, elseCtx) <- thenCtx.noStmts.mapTerm(elseTerm)

          // cond, then, else, must be fully applied here
          failIfPartial = (name: String, v: q.Val) =>
            v match {
              case t: p.Term => t.success.deferred
              case partial: q.ErasedClosureVal =>
                c.fail(s"$name term of an if-then-else statement (${term.show}) is partial ($partial)")
            }

          condRef <- failIfPartial("condition", condVal)
          thenRef <- failIfPartial("then", thenVal)
          elseRef <- failIfPartial("else", elseVal)
          tpe <- tpe match {
            case tpe: p.Type => tpe.success.deferred
            case bad         => c.fail(s"illegal erased type ${bad}")
          }

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
              (p.Term.Select(Nil, name), elseCtx.replaceStmts(ifCtx.stmts :+ result :+ cond)).success.deferred
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
            case partial: q.ErasedClosureVal =>
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
          (p.Term.UnitConst, bodyCtx.replaceStmts(c.stmts :+ block))
        }
      case _ =>
        s"[depth=${c.depth}] Unhandled: $term\nSymbol:\n${term.symbol}\nTrace was:\n${(term :: c.traces)
          .map(x => "\t" + x.show + "\n\t" + x)
          .mkString("\n---\n")}".fail.deferred
    }

  }

}
