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

    def mapTrees(args: List[q.Tree]): Deferred[(p.Term, q.FnContext)] = args match {
      case Nil => (p.Term.UnitConst, c).pure
      case x :: xs =>
        (c ::= p.Stmt.Comment(x.show))
          .mapTree(x)
          .flatMap(xs.foldLeftM(_) { case ((_, c0), term) =>
            (c0 ::= p.Stmt.Comment(term.show)).mapTree(term)
          })
    }

    def mapTree(tree: q.Tree): Deferred[(p.Term, q.FnContext)] = tree match {
      case q.ValDef(name, tpe, Some(rhs)) =>
        // if tpe is singleton, substitute with constant directly
        for {
          (term, t, c) <- c.typer(tpe.tpe)
          (ref, c)     <- term.fold((c !! tree).mapTerm(rhs))((_, c).pure)
        } yield (p.Term.UnitConst, c ::= p.Stmt.Var(p.Named(name, t), Some(p.Expr.Alias(ref))))
      case q.ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
      case t: q.Term                 => (c !! tree).mapTerm(t)
    }

    def mapTerms(args: List[q.Term]) = args match {
      case Nil => (Nil, c).pure
      case x :: xs =>
        c.mapTerm(x)
          .map((ref, c) => (ref :: Nil, c))
          .flatMap(xs.foldLeftM(_) { case ((rs, c0), term) =>
            c0.mapTerm(term).map((r, c) => (rs :+ r, c))
          })
    }

    def mapTerm(term: q.Term): Deferred[(p.Term, q.FnContext)] = term match {
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
            val term = value match {
              case name: String => p.Term.Select(Nil, p.Named(name, tpe))
              case term: p.Term => term
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
            c.typer(i.tpe).map { (_, tpe, c) =>
              (p.Term.Select(Nil, p.Named(name, tpe)), c.!!(r))
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
            } yield (ref, c)

          case (None, x) =>
            s"[depth=${c.depth}] Ref ${x} with tpe=${x.tpe} was not identified at closure args stage, ref pool:\n->${c.refs
              .mkString("\n->")} ".fail.deferred

        }

      case ap @ q.Apply(_, _) =>
        val receiverSym        = p.Sym(ap.fun.symbol.fullName)
        val receiverOwner      = ap.fun.symbol.maybeOwner
        val receiverOwnerFlags = receiverOwner.flags
        for {
          (_, tpe, c)  <- c.typer(ap.tpe)
          (argRefs, c) <- c.down(ap).mapTerms(ap.args)
          c <- ap.fun.symbol.tree match {
            case d: q.DefDef => c.mark(d).success.deferred
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

          _ = println(s"->${ap.symbol.tree.show}")

          (ref, c) <-
            if (receiverOwnerFlags.is(q.Flags.Module)) // Object.x(ys)
              mkReturn(p.Expr.Invoke(receiverSym, None, argRefs, tpe), c).success.deferred
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
                        argTpes    = argRefs.map(_.tpe)

                        name = p.Named(s"v${c.depth}_new", tpe)
                        expr = p.Stmt.Var(name, None)

                        _ <-
                          if (structTpes == argTpes) ().success
                          else s"Ctor args mismatch, class expects ${sdef.members} but was given ${argTpes}".fail

                        setMemberExprs = sdef.members.zip(argRefs).map { (member, value) =>
                          p.Stmt.Mut(p.Term.Select(name :: Nil, member), p.Expr.Alias(value))
                        }

                      } yield ((p.Term.Select(Nil, name)), c.::=(expr +: setMemberExprs*))).deferred
                    case x => s"Found ctor signature, expecting def with no rhs but got: $x".fail.deferred
                  }
                case s @ q.Select(q, n) => // s.y(zs)
                  (c !! s)
                    .mapTerm(q)
                    .map((receiverRef, c) => mkReturn(p.Expr.Invoke(receiverSym, Some(receiverRef), argRefs, tpe), c))
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
            case (s @ p.Term.Select(Nil, _), rhs) =>
              (p.Term.UnitConst, c ::= p.Stmt.Mut(s, p.Expr.Alias(rhs))).pure
            case bad => s"Illegal assign LHS,RHS: ${bad}".fail.deferred
          }
        } yield r
      case q.If(cond, thenTerm, elseTerm) =>
        for {
          (_, tpe, c)        <- c.typer(term.tpe) // TODO just return the value if result is known at type level
          (condRef, ifCtx)   <- c.down(term).mapTerm(cond)
          (thenRef, thenCtx) <- ifCtx.noStmts.mapTerm(thenTerm)
          (elseRef, elseCtx) <- thenCtx.noStmts.mapTerm(elseTerm)
          _ <- (if (condRef.tpe != p.Type.Bool) s"Cond must be a Bool ref, got ${condRef}".fail
                else ().success).deferred
          cond <- (thenRef, elseRef) match {
            case ((thenRef), (elseRef)) if thenRef.tpe == tpe && elseRef.tpe == tpe =>
              val name   = ifCtx.named(tpe)
              val result = p.Stmt.Var(name, None)
              val cond = p.Stmt.Cond(
                p.Expr.Alias(condRef),
                thenCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(thenRef)),
                elseCtx.stmts :+ p.Stmt.Mut(p.Term.Select(Nil, name), p.Expr.Alias(elseRef))
              )
              (p.Term.Select(Nil, name), elseCtx.replaceStmts(ifCtx.stmts :+ result :+ cond)).success.deferred
            case _ =>
              s"condition unification failure, then=${thenRef} else=${elseRef}, expr tpe=${tpe}".fail.deferred
          }
        } yield cond
      case q.While(cond, body) =>
        for {
          (condRef, condCtx) <- c.noStmts.down(term).mapTerm(cond)
          (_, bodyCtx)       <- condCtx.noStmts.mapTerm(body)
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
