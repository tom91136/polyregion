package polyregion.ast.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object VerifyPass {

  def transform(fs: List[p.Function], sdefs: List[StructDef]): List[(p.Function, List[String])] = fs.map { f =>
    f -> (f.body match {
      case Nil =>
        List("Function does not contain any statement")
      case xs =>
        val sdefLUT = sdefs.map(x => x.name -> x).toMap

        type Ctx = (Set[p.Named], List[String])
        val EmptyCtx: Ctx = (Set(), List.empty)


        var alloc = 0

        extension (x: Ctx) {
          def |+|(y: Ctx): Ctx      = (x._1 ++ y._1, x._2 ::: y._2)
          def +(n: p.Named): Ctx    = {
            println(s"LET $n")
            (x._1 + n, x._2)
          }
          def ~(error: String): Ctx = (x._1, error :: x._2)
          def !!(n: p.Named, tree: String): Ctx =
            if (x._1.contains(n)) x else (x._1, s"$tree uses the unseen variable ${n}, varaibles defined up to point: \n\t${x._1.mkString("\n\t") }" :: x._2)
        }

        def validateTerm(c: Ctx, t: p.Term): Ctx = t match {
          case Term.Select(Nil, local) => c !! (local, t.repr)
          case Term.Select(local :: rest, last) =>
            (rest :+ last)
              .foldLeft((c !! (local, t.repr), local.tpe)) { case ((acc, tpe), n) =>
                val c0 = tpe match {
                  case Type.Struct(name) =>
                    sdefLUT.get(name) match {
                      case None => acc ~ s"Unknown struct type ${name} in ${t.repr}"
                      case Some(sdef) =>
                        sdef.members.filter(_ == n) match {
                          case _ :: Nil => acc
                          case Nil      => acc ~ s"Struct type ${sdef.repr} does not contain member ${n} in ${t.repr}"
                          case _ => acc ~ s"Struct type ${sdef.repr} contains multiple members of $n in ${t.repr}"
                        }
                    }
                  case illegal => acc ~ s"Cannot select member ${n} from a non-struct type $illegal in ${t.repr}"
                }
                (c0, n.tpe)
              }
              ._1
          case Term.UnitConst      => c
          case Term.BoolConst(_)   => c
          case Term.ByteConst(_)   => c
          case Term.CharConst(_)   => c
          case Term.ShortConst(_)  => c
          case Term.IntConst(_)    => c
          case Term.LongConst(_)   => c
          case Term.FloatConst(_)  => c
          case Term.DoubleConst(_) => c
          case Term.StringConst(_) => c
        }

        def validateExpr(c: Ctx, e: p.Expr): Ctx = e match {
          case Expr.UnaryIntrinsic(lhs, kind, rtn)       => validateTerm(c, lhs)
          case Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) => (validateTerm(_:Ctx, lhs)).andThen(validateTerm(_ , rhs))(c)
          case Expr.UnaryLogicIntrinsic(lhs, kind)       => validateTerm(c, lhs)
          case Expr.BinaryLogicIntrinsic(lhs, rhs, kind) => (validateTerm(_:Ctx, lhs)).andThen(validateTerm(_, rhs))(c)
          case Expr.Cast(from, as)                       => validateTerm(c, from)
          case Expr.Alias(ref)                           => validateTerm(c, ref)
          case Expr.Invoke(name, receiver, args, rtn) =>
            args.foldLeft(receiver.map(validateTerm(c, _)).getOrElse(c))(validateTerm(_, _))
          case Expr.Index(lhs, idx, component) => (validateTerm(_: Ctx, lhs)).andThen(validateTerm(_, idx))(c)
          case Expr.Alloc(witness, size)       => validateTerm(c, size)
        }

        def validateStmt(c: Ctx, s: p.Stmt): Ctx = s match {
          case Stmt.Comment(_)            => c
          case Stmt.Var(name, expr)       =>
            alloc+=1
            expr.map(e => validateExpr(_: Ctx, e)).getOrElse(identity[Ctx])(c + name)
          case Stmt.Mut(name, expr, copy) => (validateTerm(_: Ctx, name)).andThen(validateExpr(_, expr))(c)
          case Stmt.Update(lhs, idx, value) =>
            (validateTerm(_, lhs)).andThen(validateTerm(_, idx)).andThen(validateTerm(_, value))(c)
          case Stmt.While(tests, cond, body) =>
            (tests
              .foldLeft(_: Ctx)(validateStmt(_, _)))
              .andThen(validateTerm(_, cond))
              .andThen(body.foldLeft(_)(validateStmt(_, _)))(c)
          case Stmt.Break => c
          case Stmt.Cont  => c
          case Stmt.Cond(cond, trueBr, falseBr) =>
            (validateExpr(_, cond))
              .andThen(trueBr.foldLeft(_)(validateStmt(_, _)))
              .andThen(falseBr.foldLeft(_)(validateStmt(_, _)))(c)
          case Stmt.Return(value) => validateExpr(c, value)
        }

        val names = f.receiver.toList ++ f.args

        val (varTable, errors) = xs.foldLeft((names.toSet, Nil): Ctx)(validateStmt(_, _))

        println(s"[Verifier] $alloc ${f.signatureRepr} vars:\n\t${varTable.mkString("\n\t")}")
        if (errors.nonEmpty) {
          throw new RuntimeException(errors.map(e => s"[Verifier] $e").mkString("\n"))
        }

        xs.flatMap(x =>
          x.acc[p.Type] {
            case p.Stmt.Return(e) => e.tpe :: Nil
            case x                => Nil
          }
        ) match {
          case Nil => List("Function contains no return statements")
          case ts if ts.exists(_ != f.rtn) =>
            List(
              s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
            )
          case _ => Nil
        }
    })
  }
}
