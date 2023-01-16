package polyregion.ast.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.ast.Traversal.*

import scala.annotation.tailrec

object VerifyPass {

  def run(program: p.Program)(log: Log): List[(p.Function, List[String])] =
    (program.entry :: program.functions).map { f =>
      f -> (f.body match {
        case Nil =>
          List("Function does not contain any statement")
        case xs =>
          val sdefLUT = program.defs.map(x => x.name -> x).toMap

          type Ctx = (Set[p.Named], List[String])
          val EmptyCtx: Ctx = (Set(), List.empty)

          var alloc = 0

          extension (x: Ctx) {
            def |+|(y: Ctx): Ctx = (x._1 ++ y._1, x._2 ::: y._2)
            def +(n: p.Named): Ctx =
              (x._1 + n, x._2)
            def ~(error: String): Ctx = (x._1, error :: x._2)
            def !!(n: p.Named, tree: String): Ctx =
              if (x._1.contains(n)) x
              else
                (
                  x._1,
                  s"$tree uses the unseen variable ${n.repr}, variables defined up to point: \n\t${x._1.mkString("\n\t")}" :: x._2
                )
          }

          def validateTerm(c: Ctx, t: p.Term): Ctx = t match {
            case p.Term.Select(Nil, local) => c !! (local, t.repr)
            case p.Term.Select(local :: rest, last) =>
              (rest :+ last)
                .foldLeft((c !! (local, t.repr), local.tpe)) { case ((acc, tpe), n) =>
                  val c0 = tpe match {
                    case s @ p.Type.Struct(name, tpeVars, args) =>
                      sdefLUT.get(name) match {
                        case None =>
                          acc ~ s"Unknown struct type ${name} in `${t.repr}`, known structs: \n${sdefLUT.map(_._2).map(_.repr).mkString("\n")}"
                        case Some(sdef) =>
                          if (sdef.tpeVars != tpeVars) {
                            acc ~ s"Struct def ${sdef.repr} and struct type ${s.repr} has different type variables"
                          } else {

                            val apTable = tpeVars.zip(args).toMap

                            println(apTable.toString + " " + sdef.repr + " " + s.repr)

                            sdef.members
                              .map(x =>
                                x.modifyAll[p.Type](_.mapLeaf {
                                  case p.Type.Var(name) => apTable(name)
                                  case x                => x
                                })
                              )
                              .filter(_.named == n) match {
                              case _ :: Nil => acc
                              case Nil =>
                                acc ~ s"Struct type ${sdef.repr} does not contain member ${n.repr} in `${t.repr} r={${program.defs}}`"
                              case _ => acc ~ s"Struct type ${sdef.repr} contains multiple members of $n in `${t.repr}`"
                            }
                          }
                      }
                    case illegal => acc ~ s"Cannot select member ${n} from a non-struct type $illegal in `${t.repr}`"
                  }
                  (c0, n.tpe)
                }
                ._1
            case p.Term.Poison(_)      => c
            case p.Term.UnitConst      => c
            case p.Term.BoolConst(_)   => c
            case p.Term.ByteConst(_)   => c
            case p.Term.CharConst(_)   => c
            case p.Term.ShortConst(_)  => c
            case p.Term.IntConst(_)    => c
            case p.Term.LongConst(_)   => c
            case p.Term.FloatConst(_)  => c
            case p.Term.DoubleConst(_) => c
          }

          def validateExpr(c: Ctx, e: p.Expr): Ctx = e match {
            case p.Expr.NullaryIntrinsic(kind, rtn)    => c
            case p.Expr.UnaryIntrinsic(lhs, kind, rtn) => validateTerm(c, lhs)
            case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) =>
              (validateTerm(_: Ctx, lhs)).andThen(validateTerm(_, rhs))(c)
            case p.Expr.Cast(from, as) => validateTerm(c, from)
            case p.Expr.Alias(ref)     => validateTerm(c, ref)
            case p.Expr.Invoke(name, tpeArgs, receiver, args, captures, rtn) =>
              val c0 = receiver.map(validateTerm(c, _)).getOrElse(c)
              val c1 = args.foldLeft(c0)(validateTerm(_, _))
              val c2 = captures.foldLeft(c1)(validateTerm(_, _))
              c2
            case p.Expr.Index(lhs, idx, component) => (validateTerm(_: Ctx, lhs)).andThen(validateTerm(_, idx))(c)
            case p.Expr.Alloc(witness, size)       => validateTerm(c, size)
          }

          def validateStmt(c: Ctx, s: p.Stmt): Ctx = s match {
            case p.Stmt.Comment(_) => c
            case p.Stmt.Var(name, expr) =>
              alloc += 1
              expr.map(e => validateExpr(_: Ctx, e)).getOrElse(identity[Ctx])(c + name)
            case p.Stmt.Mut(name, expr, copy) => (validateTerm(_: Ctx, name)).andThen(validateExpr(_, expr))(c)
            case p.Stmt.Update(lhs, idx, value) =>
              (validateTerm(_, lhs)).andThen(validateTerm(_, idx)).andThen(validateTerm(_, value))(c)
            case p.Stmt.While(tests, cond, body) =>
              (tests
                .foldLeft(_: Ctx)(validateStmt(_, _)))
                .andThen(validateTerm(_, cond))
                .andThen(body.foldLeft(_)(validateStmt(_, _)))(c)
            case p.Stmt.Break => c
            case p.Stmt.Cont  => c
            case p.Stmt.Cond(cond, trueBr, falseBr) =>
              (validateExpr(_, cond))
                .andThen(trueBr.foldLeft(_)(validateStmt(_, _)))
                .andThen(falseBr.foldLeft(_)(validateStmt(_, _)))(c)
            case p.Stmt.Return(value) => validateExpr(c, value)
          }

          val names = f.receiver.toList ++ f.args ++ f.moduleCaptures ++ f.termCaptures

          // Check for var name collisions:
          val varCollisions =
            f.collectWhere[p.Stmt] { case s: p.Stmt.Var =>
              s
            }.groupMap(v => v.name.symbol)(m => m.name.tpe -> m.expr)
              .collect {
                case (name, xs) if xs.size > 1 =>
                  s"Variable $name was defined ${xs.size} times, RHSs=${xs.map((tpe, rhs) => s"${rhs.fold("_")(_.repr)} :$tpe").mkString(";")}"
              }
              .toList

          // Check for unused vars:
          val (varTable, errors) = xs.foldLeft((names.toSet, Nil): Ctx)(validateStmt(_, _))

          // println(s"[Verifier] $alloc ${f.signatureRepr} vars:\n\t${varTable.mkString("\n\t")}")
//          if (errors.nonEmpty) {
//            throw new RuntimeException(s"[Verifier] alloc=${alloc} vars:\n\t${varTable
//              .mkString("\n\t")} \n${f.repr}\n${errors.map(e => s"  -> $e").mkString("\n")}")
//          }

          varCollisions ++ errors ++ (xs.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e.tpe } match {
            case Nil => List("Function contains no return statements")
            case ts if ts.exists(_ != f.rtn) =>
              List(
                s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
              )
            case _ => Nil
          })
      })
    }
}
