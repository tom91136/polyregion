package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.collection.immutable.VectorMap

// inline all calls originating from entry
object FnInlinePass extends ProgramPass {

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
    p.Function(f.name, f.tpeVars, f.receiver.map(rename(_)), f.args.map(rename(_)), f.captures, f.rtn, stmts)
  }

  private def applyTpeVars(table: Map[String, p.Type], f: p.Function) = {

    def apTpe(t: p.Type) = t match {
      case p.Type.Var(name) if table.contains(name) => table(name)
      case x                                        => x
    }

    p.Function(
      f.name,
      f.tpeVars,
      f.receiver.map(_.mapType(apTpe(_))),
      f.args.map(_.mapType(apTpe(_))),
      f.captures.map(_.mapType(apTpe(_))),
      apTpe(f.rtn),
      f.body.flatMap(_.mapType(apTpe(_)))
    )

  }

  def inlineOne(ivk: p.Expr.Invoke, f: p.Function) = {

    val concreteTpeArgs = ivk.receiver
      .map(_.tpe match {
        case p.Type.Struct(_, _, tpeArgs) => tpeArgs
        case _                            => Nil
      })
      .getOrElse(Nil) ++ ivk.tpeArgs

    val table = f.tpeVars.zip(concreteTpeArgs).toMap

    val renamed = renameAll(applyTpeVars(table, f))

    val substituted =
      (renamed.receiver ++ renamed.args).zip(ivk.receiver ++ ivk.args).foldLeft(renamed.body) {
        case (xs, (target, replacement)) =>
          xs.flatMap(
            _.mapTerm(
              original =>
                // println(s"substitute  ${original} contains ${target} => ${replacement}")

                (original, replacement) match {
                  case (p.Term.Select(Nil, `target`), r @ p.Term.Select(_, _)) =>
                    r
                  case (p.Term.Select(`target` :: xs, x), p.Term.Select(ys, y)) =>
                    p.Term.Select(ys ::: y :: xs, x)
                  case _ => original
                }

              // if (original == target) replacement.asInstanceOf[p.Term.Select] else original
              ,
              original =>
                // println(s"substitute  ${original} ??? ${target}")

                (original, replacement) match {
                  case (p.Term.Select(Nil, `target`), r) =>
                    r
                  case (p.Term.Select(`target` :: xs, x), p.Term.Select(ys, y)) =>
                    p.Term.Select(ys ::: y :: xs, x)
                  case _ => original
                }

              // if (original == target) replacement else original
            )
          )
      }

    val returnExprs = substituted.flatMap(_.acc {
      case p.Stmt.Return(e) => e :: Nil
      case x                => Nil
    })

    println("## " + ivk.repr)

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
        val returnName               = p.Named("phi", ivk.tpe)
        val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)
        val returnRebound = substituted.flatMap(_.map {
          case p.Stmt.Return(e) => p.Stmt.Mut(returnRef, e, copy = false) :: Nil
          case x                => x :: Nil
        })
        (p.Expr.Alias(returnRef), p.Stmt.Var(returnName, None) :: returnRebound, renamed.captures)
    }
  }

  override def apply(program: p.Program, log: Log): (p.Program, Log) = {

    val f = doUntilNotEq(program.entry) { f =>

      val (stmts, captures) = f.body.foldMap { x =>
        x.mapAccExpr {
          case ivk @ p.Expr.Invoke(name, tpeArgs, recv, args, tpe) =>
            println(s"IVK = ${ivk}")
            // Find the function through the invoke signature, we search manually as type vars can appear in args and return types as well
            program.functions.distinct.filter(f =>
              f.name == name && f.tpeVars.size == tpeArgs.size + f.receiver
                .map(_.tpe match {
                  case p.Type.Struct(_, vars, _) => vars.size
                  case _                         => 0
                })
                .getOrElse(0)
            ) match {
              case Nil =>
                println(s"-> Keep ${ivk}")
                (ivk, Nil, Nil) // can't find fn, keep it for now
              case f :: Nil => inlineOne(ivk, f)
              case xs =>
                println(xs.mkString("\n"))
                ??? // more than one, ambiguous
            }

          case x => (x, Nil, Nil)
        }
      }
      f.copy(body = stmts, captures = (f.captures ++ captures).distinct)
    }

    (p.Program(f, Nil, program.defs), log)

  }

}
