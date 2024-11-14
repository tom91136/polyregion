package polyregion

import scala.annotation.targetName
import scala.quoted.Expr
import scala.quoted.Quotes
import scala.quoted.Type

package object prism {

  type TermPrism[A, B] = (
      (Quotes, Expr[A]) => Expr[B],
      (Quotes, Expr[B]) => Expr[A],
      (Quotes, Expr[A], Expr[B]) => Expr[Unit]
  )
  type Prism = (polyregion.ast.ScalaSRR.Mirror, TermPrism[Any, Any])

  class WitnessK[A <: Any, B <: Any](
      val f: Tuple.Elem[TermPrism[A, B], 0],
      val g: Tuple.Elem[TermPrism[A, B], 1],
      val h: Tuple.Elem[TermPrism[A, B], 2]
  ) {
    def unsafePrism: TermPrism[Any, Any] =
      (
        f.asInstanceOf[Tuple.Elem[TermPrism[Any, Any], 0]],
        g.asInstanceOf[Tuple.Elem[TermPrism[Any, Any], 1]],
        h.asInstanceOf[Tuple.Elem[TermPrism[Any, Any], 2]]
      )
    // def unsafePrism : TermPrism[A,B] = (f,g)
  }

  @targetName("witness0") inline def witness[A, B]( //
      inline f: Tuple.Elem[TermPrism[A, B], 0],
      inline g: Tuple.Elem[TermPrism[A, B], 1],
      inline h: Tuple.Elem[TermPrism[A, B], 2]
  ) = WitnessK[A, B](f, g, h)

}
