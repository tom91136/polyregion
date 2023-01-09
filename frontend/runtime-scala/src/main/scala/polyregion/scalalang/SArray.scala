package polyregion.scalalang

import scala.compiletime.ops.int.*
import scala.compiletime.ops.boolean.*
import scala.compiletime.ops.any.*
import scala.compiletime.*
import scala.reflect.ClassTag
import scala.reflect.ManifestFactory.NothingManifest

final class SArray[A: ClassTag, N <: Int] private (val actual: Array[A]) {
  type Length = N
  type Empty  = N == 0
  export actual.{apply, length, update}
  def ++[N2 <: Int](xs: SArray[A, N2]): SArray[A, N + N2] = new SArray(actual ++ xs.actual)
  def :+(x: A): SArray[A, N + 1]                          = new SArray(actual :+ x)
  def +:(x: A): SArray[A, N + 1]                          = new SArray(x +: actual)
  def reverse(x: A): SArray[A, N]                         = new SArray(actual.reverse)

  def head: N > 0 match { case true => A } = actual(0).asInstanceOf

  type ClampZero <: Int = N match {
    case 0   => 0
    case Int => (N - 1)
  }
  def tail: SArray[A, ClampZero] = new SArray(actual.tail)

  def headOption: Option[A] = actual.headOption

}
object SArray {

  def empty[A: ClassTag]: SArray[A, 0]       = new SArray(Array.empty[A])
  def apply[A: ClassTag](x: A): SArray[A, 1] = new SArray(Array(x))
//   def apply[A: ClassTag](xs: A*): SArray[A, xs.size] = new SArray(Array(x))

  def foo = {

    val u: Nothing = ???
    val v: Int     = u

    val xs: SArray[Int, 1] = ???
    val m: Int             = xs.head

    val n            = xs.tail
    val xx: n.Length = 0

    ???

  }

}
