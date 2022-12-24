package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, given, *}
import polyregion.ast.Traversal.*
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

  def inlineOne(ivk: p.Expr.Invoke, f: p.Function): (p.Expr, List[p.Stmt], List[p.Named]) = {

    val concreteTpeArgs = ivk.receiver
      .map(_.tpe match {
        case p.Type.Struct(_, _, tpeArgs) => tpeArgs
        case _                            => Nil
      })
      .getOrElse(Nil) ++ ivk.tpeArgs

    val table = f.tpeVars.zip(concreteTpeArgs).toMap

    val renamed = renameAll(applyTpeVars(table, f))

    println("Renamed = " + renamed.signatureRepr)
    println("Ivk     = " + ivk.repr)
    val substituted =
      (renamed.receiver ++ renamed.args).zip(ivk.receiver ++ ivk.args).foldLeft(renamed.body) {
        case (xs, (target, replacement)) =>
          xs.flatMap(
            _.mapTerm(
              { original =>
                println(s"substitute  ${original.repr} ??? ${target.repr}")

                (original, replacement) match {
                  case (p.Term.Select(Nil, `target`), r) =>
                    println("\tHit")
                    r
                  case (p.Term.Select(`target` :: xs, x), p.Term.Select(ys, y)) =>
                    println("\tHit")
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

    val (n, f) = doUntilNotEq(program.entry, limit = 10) { (i, f) =>
      println(s"[Inline ${i}]\n${f.repr}")

    val (stmts, captures) = f.body.foldMap { x =>
      x.mapAccExpr {
        case ivk @ p.Expr.Invoke(name, tpeArgs, recv, args, rtn) =>
          // Find all viable overloads (same name and same arg count) first.
          val overloads = program.functions.distinct.filter(f => f.name == name && f.args.size == args.size)

          overloads.filter { f =>
            // For each overload candidate, we substitute any type variables with the actual invocation.
            // As we're resolving overloads, failures are expected so unresolvable variables are kept as-is.

            // We can still inline if method name and argument type resolution succeed but types don't match for the receiver.
            // This is because receivers could have *different* types in the inheritance tree.
            // `scalac` would have rejected bad receivers before this so it should be relatively safe.

            val varToTpeLut = f.tpeVars.zip(tpeArgs).toMap
            val sig = f.signature.mapType {
              case v @ p.Type.Var(n) => varToTpeLut.getOrElse(n, v)
              case x                 => x
            }
            sig.receiver.size == recv.size && // make sure receivers are both Some or None
//              sig.receiver.zip(recv.map(_.tpe)).forall(_ =:= _) &&
            sig.args.zip(args.map(_.tpe)).forall(_ =:= _) &&
            sig.rtn =:= rtn
          } match {
            case Nil =>
              println(s"-> Keep ${ivk.repr}")
              println(s"Everything:\n${program.functions.map(_.repr).mkString("\n")}")
              println(s"Overloads:\n${overloads.map(_.repr).mkString("\n")}")

              ???

              (ivk, Nil, Nil) // can't find fn, keep it for now
            case f :: Nil =>
              println(s"Overloads:\n${f.repr}")
              val (expr, stmts, names) = inlineOne(ivk, f)
              println(s"Outcome:\n${stmts.map(_.repr).mkString("\n")}")
              (expr, stmts, names)
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
