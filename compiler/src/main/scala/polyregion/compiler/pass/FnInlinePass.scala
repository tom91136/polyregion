package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*

object FnInlinePass {

  def inlineSyntheticApply(xs: List[p.Stmt]): List[p.Stmt] = ???

}
