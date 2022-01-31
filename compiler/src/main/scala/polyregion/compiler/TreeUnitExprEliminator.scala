package polyregion.compiler

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.internal.*

import scala.annotation.tailrec
import scala.quoted.*

object TreeUnitExprEliminator {

  import Retyper.*

  def eliminateUnitExpr(xs: List[p.Stmt]): List[p.Stmt] = xs.flatMap {
    case p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(p.Term.UnitConst)) | None) => Nil
    case x                                                                                => x :: Nil
  }

}
