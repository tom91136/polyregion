package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.compiler.*

import scala.annotation.tailrec
import scala.quoted.*
import polyregion.compiler.Quoted

object UnitExprElisionPass {

  def eliminateUnitExpr(using q: Quoted)(xs: q.FnContext): q.FnContext = xs.mapStmts(_.flatMap {
    _.map {
      case p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(p.Term.UnitConst)) | None) => Nil
      case x                                                                                => x :: Nil
    }
  })

}
