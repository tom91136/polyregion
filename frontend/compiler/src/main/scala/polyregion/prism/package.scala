package polyregion

import _root_.scala.annotation.targetName
import _root_.scala.quoted.Expr
import _root_.scala.quoted.Quotes
import _root_.scala.quoted.Type

package object prism {

  type TermPrism[A, B] = (
    
    (Quotes, Expr[A]) => Expr[B]
    , 
    
    (Quotes, Expr[A], Expr[B]) => Expr[A]
    )
  type Prism           = (polyregion.ast.PolyAst.Mirror, TermPrism[Any, Any])

  class WitnessK[A <: Any, B <: Any](val f: (Quotes, Expr[A]) => Expr[B], val g: (Quotes, Expr[A], Expr[B]) => Expr[A]) {
    // def unsafePrism: TermPrism[Any, Any] = (f.asInstanceOf[Expr[Any => Any]], g.asInstanceOf[Expr[(Any, Any) => Any]])

    def unsafePrism : TermPrism[A,B] = (f,g)
  }

  // A => B
  // (Buffer, Any |B ) => Unit
  // (A, Buffer)
  @targetName("witness0") inline def witness[A, B]( //
      inline f: (Quotes, Expr[A]) => Expr[B],
      inline g: (Quotes, Expr[A], Expr[B]) => Expr[A]
  ) = WitnessK[A, B](f, g)
  // @targetName("witness1") inline def witness[A[_], B[_]]( //
  //     inline f: [T0] => (Quotes, Expr[A[T0]]) => Expr[B[T0]],
  //     inline g: [T0] => (Quotes, Expr[A[T0]], Expr[B[T0]]) => Expr[A[T0]]
  // ) = WitnessK[A[Any], B[Any]](f, g)
//   @targetName("witness2") inline def witness[A[_, _], B[_, _]]( //
//       inline f: [T0, T1] => (Quotes, Expr[A[T0, T1]]) => Expr[B[T0, T1]],
//       inline g: [T0, T1] => (Quotes, Expr[(A[T0, T1], B[T0, T1])]) => Expr[A[T0, T1]]
//   ) = WitnessK[A[Any, Any], B[Any, Any]](f, g)
//   @targetName("witness3") inline def witness[A[_, _, _], B[_, _, _]]( //
//       inline f: [T0, T1, T2] => (Quotes, Expr[A[T0, T1, T2]]) => Expr[B[T0, T1, T2]],
//       inline g: [T0, T1, T2] => (Quotes, Expr[(A[T0, T1, T2], B[T0, T1, T2])]) => Expr[A[T0, T1, T2]]
//   ) = WitnessK[A[Any, Any, Any], B[Any, Any, Any]](f, g)
//   @targetName("witness4") inline def witness[A[_, _, _, _], B[_, _, _, _]]( //
//       inline f: [T0, T1, T2, T3] => (Quotes, Expr[A[T0, T1, T2, T3]]) => Expr[B[T0, T1, T2, T3]],
//       inline g: [T0, T1, T2, T3] => (Quotes, Expr[(A[T0, T1, T2, T3], B[T0, T1, T2, T3])]) => Expr[A[T0, T1, T2, T3]]
//   ) = WitnessK[A[Any, Any, Any, Any], B[Any, Any, Any, Any]](f, g)

}
