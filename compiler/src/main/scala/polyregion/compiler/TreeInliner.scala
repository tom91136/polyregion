package polyregion.compiler

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.internal.*

import scala.annotation.tailrec
import scala.quoted.*

object TreeInliner {

  import Retyper.*

  def inlineSyntheticApply(xs: List[p.Stmt]): List[p.Stmt] = ???

}
