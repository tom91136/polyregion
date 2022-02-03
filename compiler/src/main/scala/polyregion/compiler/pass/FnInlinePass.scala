package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import scala.collection.immutable.VectorMap

object FnInlinePass {

  def inlineAll(fs: List[p.Function]): List[p.Function] = {
    val lut = fs.map(f => f.name -> f).to(VectorMap)
    fs.map { f =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            lut.get(name) match {
              case None => (ivk, Nil) // can't find fn, keep it for now
              case Some(f) =>

                // do substitution first so the incoming names are not mangled
                val argSubstituted =
                  f.args.map(p.Term.Select(Nil, _)).zip(args).foldLeft(f.body) { case (xs, (target, replacement)) =>
                    xs.flatMap(
                      _.mapTerm(
                        original => if (original == target) replacement.asInstanceOf[p.Term.Select] else original,
                        original => if (original == target) replacement else original
                      )
                    )
                  }

                def rename(n: p.Named) = p.Named(s"${n.symbol}~~", n.tpe)

                val avoid = args.toSet
                // then rename all var and selects to avoid collision, avoid substituted ones
                val renamed = for {
                  s <- argSubstituted
                  s <- s.mapTerm(
                    {
                      case x if avoid contains x    => x
                      case p.Term.Select(Nil, n)    => p.Term.Select(Nil, rename(n))
                      case p.Term.Select(n :: _, _) => p.Term.Select(Nil, rename(n))
                      case x                        => x
                    },
                    identity
                  )
                  s <- s.map {
                    case p.Stmt.Var(n, expr) => p.Stmt.Var(rename(n), expr) :: Nil
                    case x                   => x :: Nil
                  }
                } yield s

                val returnExprs = renamed.flatMap(_.acc {
                  case p.Stmt.Return(e) => e :: Nil
                  case x                => Nil
                })

                returnExprs match {
                  case Nil => throw new AssertionError("no return in function")
                  case expr :: Nil => // single return, just pass the expr to the call-site
                    val noReturnStmt = argSubstituted.flatMap(_.map {
                      case p.Stmt.Return(e) => Nil
                      case x                => x :: Nil
                    })
                    (expr, noReturnStmt)
                  case xs => // multiple returns, create intermediate return var
                    val returnName               = p.Named("phi", tpe)
                    val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)
                    val returnRebound = argSubstituted.flatMap(_.map {
                      case p.Stmt.Return(e) => p.Stmt.Mut(returnRef, e, copy = false) :: Nil
                      case x                => x :: Nil
                    })
                    (p.Expr.Alias(returnRef), p.Stmt.Var(returnName, None) :: returnRebound)
                }

            }
          case x => (x, Nil)
        }
      })
    }
  }

}
