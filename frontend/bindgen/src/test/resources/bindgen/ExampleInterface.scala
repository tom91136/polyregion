package example.bindings

import scala.annotation.compileTimeOnly

object ExampleInterface {

  @compileTimeOnly("polyregion_interface:example:example.count")
  def count[T](in: Array[T], n: Int): Int =
    throw UnsupportedOperationException("compiler did not replace example.count")

  @compileTimeOnly("polyregion_interface:example:example.transform")
  def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit =
    throw UnsupportedOperationException("compiler did not replace example.transform")
}
