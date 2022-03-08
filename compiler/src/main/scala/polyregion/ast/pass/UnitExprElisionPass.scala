package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

object UnitExprElisionPass {

  def eliminateUnitExpr(xs: List[p.Function]): List[p.Function] = xs.map(x =>
    x.copy(body =
      x.body.flatMap(s =>
        s.map(_.map {
          case p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(p.Term.UnitConst)) | None) => Nil
          case x                                                                                => x :: Nil
        })
      )
    )
  )

}
