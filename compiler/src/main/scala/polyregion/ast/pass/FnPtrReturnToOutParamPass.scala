package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.collection.immutable.VectorMap

object FnPtrReturnToOutParamPass {

  def transform(xs: List[p.Function]): List[p.Function] = xs // doUntilNotEq(xs)(rewriteOnce(_))

  private def rewriteOnce(fs: List[p.Function]) = {

    // P                   // stack alloc, copy return A

    // P[]                 // heap alloc, return direct ptr
    // Struct{}[]          // heap alloc, return direct ptr
    // Struct{ Struct{} }  // heap alloc, return direct ptr
    // Struct{ A[], B }    // heap alloc, return direct ptr
    // Struct{ A, B }      // heap alloc, return direct ptr

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
