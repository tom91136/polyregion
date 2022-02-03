package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import scala.collection.immutable.VectorMap
import cats.Foldable

object FnPtrReturnToOutParamPass {

  @tailrec def doUntilNotEq[A](x: A)(f: A => A): A = {
    val y = f(x)
    if (y == x) y
    else doUntilNotEq(y)(f)
  }

  def transform(xs: List[p.Function]): List[p.Function] = doUntilNotEq(xs)(rewriteOnce(_))

  private def rewriteOnce(fs: List[p.Function]) = {

    // def fn(...): S {
    //   var x: S = ...
    //   (x.a: S.x) := ...
    //   return x
    // }

    // def fn(x: S, ...): S {
    //   (x.a: S.x) := ...
    //   return x
    // }

    // rewrite all functions where the body allocate a struct
    val functions =
      fs.map { f =>
        val rewritten =
          if (f.rtn.kind != p.TypeKind.Ref) (None, f)
          else {
            val outParam = p.Named("return_out", f.rtn)
            val stmts = f.body.foldMap(_.map {
              case p.Stmt.Return(e) =>
                p.Stmt.Mut(p.Term.Select(Nil, outParam), e, copy = true) ::
                  p.Stmt.Return(p.Expr.Alias(p.Term.UnitConst)) :: Nil
              case x => (x :: Nil)
            })
            (Some(outParam), f.copy(args = outParam :: f.args, rtn = p.Type.Unit, body = stmts))
          }
      (f.name, rewritten)
      }.to(VectorMap) // make sure we keep the order

    // rewrite all call sites for all functions
    functions.values.map { (_, f) =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            functions.get(name) match {
              case Some((None, _)) | None => (ivk, Nil)
              case Some((Some(outParam), _)) =>
                val outVar = p.Stmt.Var(outParam, None)
                (ivk.copy(args = p.Term.Select(Nil, outParam) :: ivk.args), outVar :: Nil)
            }
          case x => (x, Nil)
        }
      })
    }.toList
  }

}
