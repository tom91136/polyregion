package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import scala.collection.immutable.VectorMap

object FnAllocElisionPass {

  def transform(xs: List[p.Function]): List[p.Function] = {

    // def fn(...): S {
    //   var x: S = ...
    //   (x.a: S.x) := ...
    //   return x
    // }

    // def fn(x: S, ...): S {
    //   (x.a: S.x) := ...
    //   return x
    // }
    //

    // rewrite all functions where the body allocate a struct
    val functions = xs.map { f =>
      // delete vars
      val (stmts, allocNames) = f.body.foldMap(_.mapAcc[p.Named] {
        case p.Stmt.Var(n @ p.Named(_, tpe), None) if tpe.kind == p.TypeKind.Ref && tpe != p.Type.Unit =>
          (Nil, n :: Nil)
        case x => (x :: Nil, Nil)
      })
      f.name -> (allocNames, f.copy(rtn = p.Type.Unit, body = stmts, args = allocNames ::: f.args))
    }.to(VectorMap)

    // rewrite all call sites for all functions
    functions.values.map { (_, f) =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            functions.get(name) match {
              case None => (ivk, Nil)
              case Some((allocNames, _)) =>
                val vars = allocNames.map(p.Stmt.Var(_, None))
                (ivk.copy(args = allocNames.map(p.Term.Select(Nil, _)) ::: ivk.args), vars)
            }
          case x => (x, Nil)
        }
      })
    }.toList
  }
}
