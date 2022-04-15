package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.collection.immutable.VectorMap

// inline all calls originating from entry
object FnInlinePass {

  // rename all var and selects to avoid collision
  private def renameAll(f: p.Function): p.Function = {
    def rename(n: p.Named) = p.Named(s"_inline_${f.mangledName}_${n.symbol}", n.tpe)

    val captureNames = f.captures.toSet
    val stmts = for {
      s <- f.body
      s <- s.mapTerm(
        {
          case s @ p.Term.Select(Nil, n) if captureNames.contains(n)    => s
          case s @ p.Term.Select(n :: _, _) if captureNames.contains(n) => s
          case p.Term.Select(Nil, n)                                    => p.Term.Select(Nil, rename(n))
          case p.Term.Select(n :: ns, x)                                => p.Term.Select(rename(n) :: ns, x)
          case x                                                        => x
        },
        {
          case s @ p.Term.Select(Nil, n) if captureNames.contains(n)    => s
          case s @ p.Term.Select(n :: _, _) if captureNames.contains(n) => s
          case p.Term.Select(Nil, n)                                    => p.Term.Select(Nil, rename(n))
          case p.Term.Select(n :: ns, x)                                => p.Term.Select(rename(n) :: ns, x)
          case x                                                        => x
        }
      )
      s <- s.map {
        case p.Stmt.Var(n, expr) => p.Stmt.Var(rename(n), expr) :: Nil
        case x                   => x :: Nil
      }
    } yield s
    p.Function(f.name, f.receiver.map(rename(_)), f.args.map(rename(_)), f.captures, f.rtn, stmts)
  }

  def run(program: p.Program)(log: Log): (p.Program, Log) = {
    val lut = program.functions.map(f => f.signature -> f).toMap
    val f = doUntilNotEq(program.entry) { f =>

      val (stmts, captures) = f.body.foldMap { x =>
        x.mapAccExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            val sig = p.Signature(name, recv.map(_.tpe), args.map(_.tpe), tpe)
            lut.get(sig) match {
              case None =>
                (ivk, Nil, Nil) // can't find fn, keep it for now
              case Some(f) =>
                val renamed = renameAll(f)

                val substituted =
                  (renamed.receiver ++ renamed.args).zip(recv ++ args).foldLeft(renamed.body) {
                    case (xs, (target, replacement)) =>
                      xs.flatMap(
                        _.mapTerm(
                          original => {

                            // println(s"substitute  ${original} contains ${target} => ${replacement}")

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
                            // println(s"substitute  ${original} ??? ${target}")

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
                    (expr, noReturnStmt, renamed.captures)
                  case xs => // multiple returns, create intermediate return var
                    val returnName               = p.Named("phi", tpe)
                    val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)
                    val returnRebound = substituted.flatMap(_.map {
                      case p.Stmt.Return(e) => p.Stmt.Mut(returnRef, e, copy = false) :: Nil
                      case x                => x :: Nil
                    })
                    (p.Expr.Alias(returnRef), p.Stmt.Var(returnName, None) :: returnRebound, renamed.captures)
                }

            }
          case x => (x, Nil, Nil)
        }
      }

      f.copy(body = stmts ,captures = f.captures ++ captures)
    }

    (p.Program(f, Nil, program.defs), log)

  }

}
