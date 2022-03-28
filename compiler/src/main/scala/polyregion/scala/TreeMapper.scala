package polyregion.scala

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object TreeMapper {

  import Retyper.*
  extension (using q: Quoted)(c: q.FnContext) {

    def mapTrees(args: List[q.Tree]): Deferred[(q.Val, q.FnContext)] = args match {
      case Nil => (p.Term.UnitConst, c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show.replace("\n", " ; ")))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show.replace("\n", " ; "))).mapTree(term)
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
            case (v: q.ErasedMethodVal, tpe: q.ErasedFnTpe) => c.suspend(name -> tpe)(v)
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

    def mapRef(r: q.Ref): Deferred[(q.Val, q.FnContext)] = (c.refs.get(r.symbol), r) match {
      case (Some(q.Reference(value, tpe)), _) =>
        val term = (value, tpe) match {
          case (name: String, tpe: p.Type) => p.Term.Select(Nil, p.Named(name, tpe))
          case (term: p.Term, _)           => term
          case _                           => ???
        }
        (term, c).pure
      case (None, i @ q.Ident(s)) =>
//        println(s"S=${i.symbol} name=$s")
        val name = i.tpe match {
          // we've encountered a case where the ident's name is different from the TermRef's name
          // this is likely a result of inline where we end up with synthetic names
          // we use the TermRef's name in this case
          case q.TermRef(_, name) if name != s => name
          case _                               => s
        }
        // println(s"$i = $s")
        for {
          (_, tpe, c) <- c.typer(i.tpe)
          // Ident is local, so if it is an erased closure type, we check the context first to see if we have a defined suspension
          // for types that are already part of PolyAst, we just use it as is

          // _ = println(  q.Printer.TreeCode)
//          _ = println(s"sel ident $s = ${tpe} = ${i.show(using q.Printer.TreeCode)} ${i.symbol.owner}")

          (term, c) <- tpe match {
            case tpe: p.Type =>
              // if the owner is a class, then prefix select starting with `this`
              (if (i.symbol.owner.isClassDef) {
                 c.clsSymTyper(i.symbol.owner).map {
                   case (cls: p.Type, c) => p.Named("this", cls) :: Nil
                   case (cls : q.ErasedClsTpe, c)  if cls.kind == q.ClassKind.Object => Nil

                   case (bad, c) =>
                     println(bad)
                     ???
                 }
               } else Nil.success).map { parent =>
                (p.Term.Select(parent, p.Named(name, tpe)), c)
              }.deferred
            case ect: q.ErasedFnTpe =>
              // we may end up here and not Select for functions inside modules, not entirely sure why that is
              // here ident's name will be the function name
              i.symbol.tree match {
                case f: q.DefDef => // (Symbol...).(x: X=>Y)
                  val sym      = p.Sym(i.symbol.name)
                  val receiver = p.Sym(i.symbol.maybeOwner.fullName)
                  (q.ErasedMethodVal(receiver, sym, ect, f), c /*.mark(receiver ~ sym, f)*/ ).success.deferred
                case _ =>
                  c.suspended.get(name -> ect) match {
                    case Some(x) => (x, c).success.deferred
                    case None    => c.fail[(q.Val, q.FnContext)](s"Can't find a previous definition of ${i}")
                  }
              }
            case q.ErasedClsTpe(sym, q.ClassKind.Object, Nil) => (q.ErasedModuleSelect(sym), c).success.deferred
            case et: q.ErasedClsTpe                           => c.fail[(q.Val, q.FnContext)](s"Saw ${et}")
          }
        } yield (term, c)
      case (None, s @ q.Select(root, name)) =>
//        println(s"S=${s.symbol}, root=${root} name=$name")
        // we must stop at the PolyType boundary as we discard any unapplied type trees

        for {
          (_, tpe, c) <- c.typer(s.tpe)
//          _ = println(s"sel $s = ${tpe}")
          // don't resolve root here yet
          (term, c) <- tpe match {
            case q.ErasedClsTpe(sym, q.ClassKind.Object, Nil) => // <module>.(...)
              (q.ErasedModuleSelect(sym), c).success.deferred
            case tpe: p.Type => // (selector...).(x:Term)
//              println(s"X=$tpe")

              c.mapTerm(root).flatMap {
                case (select @ p.Term.Select(xs, x), c) =>
                  s.symbol.tree match {
                    case dd: q.DefDef if dd.paramss.isEmpty =>
                      // no-arg instance def call (i.e. `x.toDouble` )
                      val fnSym              = p.Sym(s.symbol.name)
                      val named              = c.named(tpe)
                      val ivk: p.Expr.Invoke = p.Expr.Invoke(fnSym, Some(select), Nil, tpe)
                      val c1 = c
                        .down(dd)
                        .mark(ivk.signature, dd) ::= p.Stmt.Var(named, Some(ivk))
                      (p.Term.Select(Nil, named), c1).success.deferred
                    case dd: q.DefDef =>
                      c.fail(s"Unexpected arg list ${dd.paramss} for a 0-arg def via instance ref ${select.repr}")
                    case vd: q.ValDef =>
                      (p.Term.Select(xs :+ x, p.Named(name, tpe)), c).success.deferred

//                      c.fail(s"$vd via instance ref ${select.repr} was not intercepted by the outliner!?")
                    case bad => c.fail(s"Unsupported construct $bad via instance ref ${select.repr}")
                  }

//                  println(s"SEL s = ${s.symbol.tree}")

                case (q.ErasedModuleSelect(module), c) =>
                  s.symbol.tree match {
                    case dd: q.DefDef if dd.paramss.isEmpty =>
                      // no-arg module def call (i.e. `object X{ def y :Int = ??? }` )
                      val fnSym              = module :+ name
                      val named              = c.named(tpe)
                      val ivk: p.Expr.Invoke = p.Expr.Invoke(fnSym, None, Nil, tpe)
                      val c1 = c
                        .down(dd)
                        .mark(ivk.signature, dd) ::= p.Stmt.Var(named, Some(ivk))
                      (p.Term.Select(Nil, named), c1).success.deferred
                    case dd: q.DefDef =>
                      c.fail(s"Unexpected arg list ${dd.paramss} for a 0-arg def via module ref ${module.repr}")
                    case vd: q.ValDef =>
                      c.fail(s"$vd via module ref ${module.repr} was not intercepted by the outliner!?")
                    case bad => c.fail(s"Unsupported construct $bad via module ref ${module.repr}")
                  }
                case (term: p.Term, c) =>
                  s.symbol.tree match {
                    case dd: q.DefDef if dd.paramss.isEmpty =>
                      // no-arg instance def call (i.e. `1.toDouble` )
                      val fnSym              = p.Sym(s.symbol.name)
                      val named              = c.named(tpe)
                      val ivk: p.Expr.Invoke = p.Expr.Invoke(fnSym, Some(term), Nil, tpe)
                      val c1                 = c.down(dd).mark(ivk.signature, dd) ::= p.Stmt.Var(named, Some(ivk))
                      (p.Term.Select(Nil, named), c1).success.deferred
                    case bad => c.fail(s"Unsupported construct $bad via instance ref ${term.repr}")
                  }
                case (bad, c) =>
                  c.fail(s"Unexpected root of a select that leads to a non-erased ${tpe} type: ${bad}")
              }
            case ect: q.ErasedFnTpe => // (selector...).(x:(X=>Y))
              val defdef = s.symbol.tree match {
                case dd: q.DefDef => dd
                case _            => ???
              }
              val sym = p.Sym(s.symbol.name)
              c.mapTerm(root).flatMap {
                case (receiver: p.Term, c) => // `val f : X => Y = ??? | def f(x: X): Y = ???`
                  (q.ErasedMethodVal(receiver, sym, ect, defdef), c /* .mark(sym, defdef) */ ).success.deferred
                case (q.ErasedModuleSelect(module), c) =>
                  (q.ErasedMethodVal(module, sym, ect, defdef), c /* .mark(module ~ sym, defdef)*/ ).success.deferred
                case (bad, c) => c.fail(s"Unexpected root of a select that leads to an erased ${tpe} type: ${bad}")
              }
            case et: q.ErasedClsTpe => c.fail[(q.Val, q.FnContext)](s"Saw ${et}")
          }

        } yield (term, c)
      case (None, x) => c.fail(s"Ref ${x} with tpe=${x.tpe} was not identified at closure args stage")
    }

    def mapTerm(term: q.Term): Deferred[(q.Val, q.FnContext)] = {

      // println(s">>${term.show} = ${c.stmts.size} ~ ${term}")

      term match {
        case q.NamedArg(name, x) => (c !! term).mapTerm(x).map((v, c) => (q.ErasedNamedArg(name, v), c))
        case q.This(_) =>
          c.typer(term.tpe).subflatMap {
            case (_, tpe: p.Type, c) => (p.Term.Select(Nil, p.Named("this", tpe)), c).success
            case (_, bad, c)         => s"Unsupported `this` type: ${bad}".fail
          }
        case a @ q.TypeApply(term, args) =>
          for {
            (_, tpe, c) <- c.typer(a.tpe)
            (v, c) <- tpe match {
              case ect: q.ErasedFnTpe =>
                a.symbol.tree match {
                  case f: q.DefDef => // (Symbol...).(x: X=>Y)
                    val sym      = p.Sym(a.symbol.name)
                    val receiver = p.Sym(a.symbol.maybeOwner.fullName)
                    println(s"[mapper] type apply of erased fn: ${receiver ~ sym}")
                    (
                      q.ErasedMethodVal(receiver, sym, ect, f),
                      c                /* .mark(receiver ~ sym, f)*/
                    ).success.deferred // ofDim
                  case _ => ???
                }
              case bad => c.fail[(q.Val, q.FnContext)](s"Saw ${bad}")
            }
          } yield (v, c)
        case q.Typed(x, _)                        => (c !! term).mapTerm(x)
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
        case r: q.Ref                        =>
          // println(s"[mapper] ref @ ${r}")
          (c !! r).mapRef(r)
        case q.New(tptTree) =>
          // println(s"New=${term.show}")

          (c !! term).typer(tptTree.tpe).map {
            case (_, tpe: p.Type, c) =>
              val name = c.named(tpe)
              (p.Term.Select(Nil, name), c ::= p.Stmt.Var(name, None))
            case (_, tpe, c) => ???

          }
        case ap @ q.Apply(_, _) =>
          val receiverSym        = p.Sym(ap.fun.symbol.fullName)
          val receiverOwner      = ap.fun.symbol.maybeOwner
          val receiverOwnerFlags = receiverOwner.flags

          //
          // println(s"A=${ap.args}")

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
            (_, funTpe, c) <- c.typer(ap.fun.tpe)
            (funVal, c)    <- (c !! ap).mapTerm(ap.fun)
            (argTpes, c)   <- c.typerN(ap.args.map(_.tpe))

            // discard application of erased types
            argsNoErasedTpe = argTpes.zip(ap.args).flatMap {
              case ((_, _: q.ErasedClsTpe), x) => Nil
              case ((_, _), x)                 => x :: Nil
            }
//            _ = println(s"M=${funVal} (...) ")

            (argVals, c) <- c.down(ap).mapTerms(argsNoErasedTpe)

            // XXX although this behaviour wasn't explicitly documented, it appears that with named args,
            // the compiler will order the args correctly beforehand so we don't have to deal with it here
            argTerms <- argVals.traverse {
              case q.ErasedNamedArg(_, t: p.Term) => t.success.deferred
              case t: p.Term                      => t.success.deferred
              case bad                            => c.fail(s"Illegal ${bad}")
            }
//            _ = println("=== " + argTpes)

            mkReturn = (expr: p.Expr, c: q.FnContext) => {

              val name = c.named(expr.tpe)
              (p.Term.Select(Nil, name), c.down(term) ::= p.Stmt.Var(name, Some(expr)))
            }

            // _ = println(s"saw apply -> ${ap.symbol.tree.show}")
            // _ = println(s"inner is  -> ${defdef.show}")
            // _ = println(s"inner is  -> ${funVal} AP ${argTerms}")

//            _ = println(s"[mapper] apply function value: ${funVal}")

            (ref, c) <- (argTerms, funVal) match {
              case (Nil, x) => (x, c).success.deferred
              case (_, q.ErasedMethodVal(receiver: p.Term.Select, p.Sym("<init>" :: Nil), fnTpe, _)) => // ctor call
                (for {
                  sdef <- (fnTpe.rtn match {
                    case p.Type.Struct(s) => c.clss.get(s)
                    case _                => None
                  }).failIfEmpty(s"No StructDef found for type ${fnTpe.rtn}")

                  // we need to also make sure the ctor has no impl here
                  // that would mean it's in the form of `class X(val field : Y)`
                  structTpes  = sdef.members.map(_.tpe)
                  ctorArgTpes = fnTpe.args

//                  a = ctorArgTpes.x
                  _ <-
                    if (structTpes == argTpes.map(_._2) && structTpes == ctorArgTpes.map(_._2)) ().success
                    else
                      s"Ctor args mismatch, class ${sdef.name} expects ${structTpes}, fn expects ${ctorArgTpes} and was applied with ${argTpes}".fail

                  setMemberExprs = sdef.members.zip(argTerms).map { (member, value) =>
                    p.Stmt.Mut(p.Term.Select(receiver.init :+ receiver.last, member), p.Expr.Alias(value), copy = false)
                  }
                } yield (receiver, c.::=(setMemberExprs*))).deferred
              case (_, m @ q.ErasedMethodVal(module: p.Sym, sym, tpe, underlying)) => // module call
                val t = tpe.rtn match {
                  case t: p.Type => t
                  case q.ErasedFnTpe(args, rtn: p.Type) if args.forall {
                        case (_, _: q.ErasedClsTpe) => true
                        case _                      => false
                      } =>
                    rtn

                  //
                  case x: q.ErasedClsTpe =>
//                    println(x)
                    // x
                    ???
                } // TODO handle multiple arg list methods
                val ivk: p.Expr.Invoke = p.Expr.Invoke(module ~ sym, None, argTerms, t)
                mkReturn(ivk, c.mark(ivk.signature, underlying)).success.deferred
              case (_, q.ErasedMethodVal(receiver: p.Term, sym, tpe, underlying)) => // instance call
                val t = tpe.rtn match {
                  case t: p.Type => t
                  case _         => ???
                }
                val ivk: p.Expr.Invoke = p.Expr.Invoke(sym, Some(receiver), argTerms, t)
                mkReturn(ivk, c.mark(ivk.signature, underlying)).success.deferred
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
              case (lhsClosure: q.ErasedMethodVal, rhsClosure: q.ErasedMethodVal) =>
                if (lhsClosure.tpe != rhsClosure.tpe) c.fail(s"Assignment of incompatible type ${lhsRef} := ${rhsRef}")
                else ???

              case (s @ p.Term.Select(Nil, _), rhs) =>
                (
                  p.Term.UnitConst,
                  rhs match {
                    case t: p.Term            => c ::= p.Stmt.Mut(s, p.Expr.Alias(t), copy = false)
                    case p: q.ErasedMethodVal => ??? // c.suspend(s)(p)
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
                case partial: q.ErasedMethodVal =>
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
              case partial: q.ErasedMethodVal =>
                s"condition term of a while expression (${term.show}) is partial ($partial)".fail.deferred
            }
          } yield (
            p.Term.UnitConst,
            bodyCtx.replaceStmts(c.stmts :+ p.Stmt.While(condCtx.stmts, condRef, bodyCtx.stmts))
          )
        case _ => c.fail(s"Unhandled: `$term`, show=`${term.show}`\nSymbol:\n${term.symbol}")
      }
    }
  }

}
