package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.ast.Traversal.*
import scala.collection.immutable.VectorMap

// inline all calls originating from entry
object FnInlinePass extends ProgramPass {

  // rename all var and selects to avoid collision
  private def renameAll(f: p.Function): p.Function = {
    def rename(n: p.Named) = p.Named(s"_inline_${f.mangledName}_${n.symbol}", n.tpe)
    val captureNames       = f.moduleCaptures.map(_.named).toSet
    val body = f.body
      .modifyAll[p.Term] {
        case s @ p.Term.Select(Nil, n) if captureNames.contains(n)    => s
        case s @ p.Term.Select(n :: _, _) if captureNames.contains(n) => s
        case p.Term.Select(Nil, n)                                    => p.Term.Select(Nil, rename(n))
        case p.Term.Select(n :: ns, x)                                => p.Term.Select(rename(n) :: ns, x)
        case x                                                        => x
      }
      .modifyAll[p.Stmt] {
        case p.Stmt.Var(n, expr) => p.Stmt.Var(rename(n), expr)
        case x                   => x
      }
    p.Function(
      f.name,
      f.tpeVars,
      f.receiver.map(arg => arg.copy(rename(arg.named))),
      f.args.map(arg => arg.copy(rename(arg.named))),
      f.moduleCaptures,
      f.termCaptures.map(arg => arg.copy(rename(arg.named))),
      f.rtn,
      body
    )
  }

  private def inlineOne(ivk: p.Expr.Invoke, f: p.Function): (p.Expr, List[p.Stmt], List[p.Arg]) = {

    val concreteTpeArgs = ivk.receiver
      .map(_.tpe match {
        case p.Type.Struct(_,   _, tpeArgs, _) => tpeArgs
        case _                                  => Nil
      })
      .getOrElse(Nil) ++ ivk.tpeArgs

    val table = f.tpeVars.zip(concreteTpeArgs).toMap

    val renamed = renameAll(f.modifyAll[p.Type](_.mapLeaf {
      case p.Type.Var(name) if table.contains(name) => table(name)
      case x                                        => x
    }))

    println("Renamed = " + renamed.signatureRepr)
    println("Ivk     = " + ivk.repr)
    val substituted =
      (renamed.receiver.map(_.named) ++ renamed.args.map(_.named) ++ renamed.termCaptures.map(_.named))
        .zip(ivk.receiver ++ ivk.args ++ ivk.captures)
        .foldLeft(renamed.body) { case (xs, (target, replacement)) =>
          xs.modifyAll[p.Term] { original =>
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

        }

    val returnExprs = substituted.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e }

    returnExprs match {
      case Nil =>
        throw new AssertionError(
          s"no return in function ${f.signature}, substituted:\n${returnExprs.map(_.repr).mkString("\n")}"
        )
      case expr :: Nil => // single return, just pass the expr to the call-site
        val noReturnStmt = substituted.modifyAll[p.Stmt] {
          case p.Stmt.Return(e) => p.Stmt.Comment(s"inlined return ${e}")
          case x                => x
        }
        (expr, noReturnStmt, renamed.moduleCaptures)
      case xs => // multiple returns, create intermediate return var
        val returnName               = p.Named("phi", ivk.tpe)
        val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)
        val returnRebound = substituted.modifyAll[p.Stmt] {
          case p.Stmt.Return(e) => p.Stmt.Mut(returnRef, e, copy = false)
          case x                => x
        }
        (p.Expr.Alias(returnRef), p.Stmt.Var(returnName, None) :: returnRebound, renamed.moduleCaptures)
    }
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    println(">FnInlinePass")

    val (n, f) = doUntilNotEq(program.entry, limit = 10) { (i, f) =>
      println(s"[Inline ${i}]\n${f.repr}")

      val (stmts, moduleCaptures) = f.body.foldMap { x =>

        val (y, xs) = x.modifyCollect[p.Expr, (List[p.Stmt], List[p.Arg])] {
          case ivk @ p.Expr.Invoke(name, tpeArgs, recv, args, captures, rtn) =>
            // Find all viable overloads (same name and same arg count) first.
            val overloads = program.functions.distinct.filter(f => f.name == name && f.args.size == args.size)

            overloads.filter { f =>
              // For each overload candidate, we substitute any type variables with the actual invocation.
              // As we're resolving overloads, failures are expected so unresolvable variables are kept as-is.

              // We can still inline if method name and argument type resolution succeed but types don't match for the receiver.
              // This is because receivers could have *different* types in the inheritance tree.
              // `scalac` would have rejected bad receivers before this so it should be relatively safe.

              val varToTpeLut = f.tpeVars.zip(tpeArgs).toMap
              val sig = f.signature.modifyAll[p.Type](_.mapLeaf {
                case v @ p.Type.Var(n) => varToTpeLut.getOrElse(n, v)
                case x                 => x
              })
              sig.receiver.size == recv.size && // make sure receivers are both Some or None
              sig.args.zip(args.map(_.tpe)).forall(_ =:= _) &&
              sig.termCaptures.zip(captures.map(_.tpe)).forall(_ =:= _) &&
              sig.rtn =:= rtn
            } match {
              case Nil =>
                println(s"-> Keep ${ivk.repr}")
                println(s"Everything:\n${program.functions.map(_.repr).mkString("\n")}")
                println(s"Overloads:\n${overloads.map(_.repr).mkString("\n")}")

                ???

                (ivk, Nil -> Nil) // can't find fn, keep it for now
              case f :: Nil =>
                println(s"Overloads:\n${f.repr}")
                val (expr, stmts, names) = inlineOne(ivk, f)
                println(s"Outcome:\n${stmts.map(_.repr).mkString("\n")}")
                (expr, stmts -> names)
              case xs =>
                println(xs.mkString("\n"))
                ??? // more than one, ambiguous
            }

          case x => (x, Nil -> Nil)
        }

        val (stmts, moduleCaptures) = xs.combineAll //
        (stmts :+ y, moduleCaptures)
      }

      f.copy(body = stmts, moduleCaptures = (f.moduleCaptures ++ moduleCaptures).distinct)
    }

    println("Done")
    p.Program(f, Nil, program.defs)

  }

}
