package polyregion

import _root_.scala.annotation.targetName

package object prism {

  type TermPrism[A, B] = (A => B, (A, B) => A)
  type Prism           = (polyregion.ast.PolyAst.Mirror, TermPrism[Any, Any])

  class WitnessK[A <: Any, B <: Any](val f: Any, val g: Any) {
    def unsafePrism: TermPrism[Any, Any] = (f.asInstanceOf[Any => Any], g.asInstanceOf[(Any, Any) => Any])
  }

  @targetName("witness0") inline def witness[A, B]( //
      f: A => B,
      g: (A, B) => A
  ) = WitnessK[A, B](f, g)
  @targetName("witness1") inline def witness[A[_], B[_]]( //
      f: [T0] => A[T0] => B[T0],
      g: [T0] => (A[T0], B[T0]) => A[T0]
  ) = WitnessK[A[Any], B[Any]](f, g)
  @targetName("witness2") inline def witness[A[_, _], B[_, _]]( //
      f: [T0, T1] => A[T0, T1] => B[T0, T1],
      g: [T0, T1] => (A[T0, T1], B[T0, T1]) => A[T0, T1]
  ) = WitnessK[A[Any, Any], B[Any, Any]](f, g)
  @targetName("witness3") inline def witness[A[_, _, _], B[_, _, _]]( //
      f: [T0, T1, T2] => A[T0, T1, T2] => B[T0, T1, T2],
      g: [T0, T1, T2] => (A[T0, T1, T2], B[T0, T1, T2]) => A[T0, T1, T2]
  ) = WitnessK[A[Any, Any, Any], B[Any, Any, Any]](f, g)
  @targetName("witness4") inline def witness[A[_, _, _, _], B[_, _, _, _]]( //
      f: [T0, T1, T2, T3] => A[T0, T1, T2, T3] => B[T0, T1, T2, T3],
      g: [T0, T1, T2, T3] => (A[T0, T1, T2, T3], B[T0, T1, T2, T3]) => A[T0, T1, T2, T3]
  ) = WitnessK[A[Any, Any, Any, Any], B[Any, Any, Any, Any]](f, g)

}
