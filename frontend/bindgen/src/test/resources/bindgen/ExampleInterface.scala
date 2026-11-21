package example.bindings

import scala.annotation.compileTimeOnly

object ExampleInterface {

  @compileTimeOnly("polyregion_interface:example:example.count")
  inline def count[T](in: Array[T], n: Int): Int =
    polyregion.scalalang.launch.call[Int]("example:example.count", polyregion.jvm.runtime.Context.current(), in, n)

  @compileTimeOnly("polyregion_interface:example:example.transform")
  inline def transform[T, U](in: Array[T], out: Array[U], n: Int, inline op: T => U): Unit =
    polyregion.scalalang.launch.call[Unit](
      "example:example.transform",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      op
    )
}
