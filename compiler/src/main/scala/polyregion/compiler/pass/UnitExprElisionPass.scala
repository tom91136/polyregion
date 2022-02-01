package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*

object UnitExprElisionPass {

  def eliminateUnitExpr(xs: List[p.Stmt]): List[p.Stmt] = xs.flatMap {
    _.map {
      case p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(p.Term.UnitConst)) | None) => Nil
      case x                                                                                => x :: Nil
    }
  }

}
