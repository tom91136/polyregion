package polyregion.backend.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.backend.ast.PolyAst as p
import polyregion.backend.compiler.*

import scala.annotation.tailrec
import scala.quoted.*
import scala.collection.immutable.VectorMap

object FnInlinePass {

  // rename all var and selects to avoid collision
  private def renameAll(f: p.Function): p.Function = {

    def rename(n: p.Named) = p.Named(s"_inline_${f.mangledName}_${n.symbol}", n.tpe)

    val stmts = for {
      s <- f.body
      s <- s.mapTerm(
        {
          case p.Term.Select(Nil, n)     => p.Term.Select(Nil, rename(n))
          case p.Term.Select(n :: ns, x) => p.Term.Select(rename(n) :: ns, x)
          case x                         => x
        },
        {
          case p.Term.Select(Nil, n)     => p.Term.Select(Nil, rename(n))
          case p.Term.Select(n :: ns, x) => p.Term.Select(rename(n) :: ns, x)
          case x                         => x
        }
      )
      s <- s.map {
        case p.Stmt.Var(n, expr) => p.Stmt.Var(rename(n), expr) :: Nil
        case x                   => x :: Nil
      }
    } yield s

    p.Function(f.name, f.receiver.map(rename(_)), f.args.map(rename(_)), f.rtn, stmts)
  }

  def inlineAll(fs: List[p.Function]): List[p.Function] = doUntilNotEq(fs) { fs =>

    println(s"[inline-pass] fns:\n -> ${fs.map(f => s"${f.signatureRepr}").mkString("\n -> ")}")

    val lut = fs.map(f => f.signature -> f).to(VectorMap)

    fs.map { f =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            val sig = p.Signature(name, recv.map(_.tpe), args.map(_.tpe), tpe)
            lut.get(sig) match {
              case None =>
                println(s"none = ${sig}")
                (ivk, Nil) // can't find fn, keep it for now
              case Some(f) =>
                println(s"yes = ${sig}")
                // do substitution first so the incoming names are not mangled

                val renamed = renameAll(f)

                val substituted =
                  (renamed.receiver ++ renamed.args).zip(recv ++ args).foldLeft(renamed.body) {
                    case (xs, (target, replacement)) =>
                      xs.flatMap(
                        _.mapTerm(
                          original => {

                            println(s"substitute  ${original} contains ${target} => ${replacement}")

                            (original, replacement) match {
                              case (p.Term.Select(Nil, `target`), r @ p.Term.Select(_, _)) =>
                                r
                              case (p.Term.Select(`target` :: xs, x), p.Term.Select(ys, y)) =>
                                p.Term.Select(ys ::: y :: xs, x)
                              case _ => original
                            }

                            // if (original == target) replacement.asInstanceOf[p.Term.Select] else original
                          },
                          original => {
                            println(s"substitute  ${original} ??? ${target}")

                            (original, replacement) match {
                              case (p.Term.Select(Nil, `target`), r) =>
                                r
                              case (p.Term.Select(`target` :: xs, x), p.Term.Select(ys, y)) =>
                                p.Term.Select(ys ::: y :: xs, x)
                              case _ => original
                            }

                            // if (original == target) replacement else original
                          }
                        )
                      )
                  }

                val returnExprs = substituted.flatMap(_.acc {
                  case p.Stmt.Return(e) => e :: Nil
                  case x                => Nil
                })

                returnExprs match {
                  case Nil =>
                    throw new AssertionError(
                      s"no return in function ${f.signature}, substituted:\n${returnExprs.map(_.repr).mkString("\n")}"
                    )
                  case expr :: Nil => // single return, just pass the expr to the call-site
                    val noReturnStmt = substituted.flatMap(_.map {
                      case p.Stmt.Return(e) => Nil
                      case x                => x :: Nil
                    })
                    (expr, noReturnStmt)
                  case xs => // multiple returns, create intermediate return var
                    val returnName               = p.Named("phi", tpe)
                    val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)
                    val returnRebound = substituted.flatMap(_.map {
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
