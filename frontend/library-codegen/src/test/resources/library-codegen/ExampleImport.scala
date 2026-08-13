package example.bindings

import scala.annotation.StaticAnnotation

object ExampleImport {
  final class PolyregionImport(val library: String, val declaration: String) extends StaticAnnotation

  final class PolyregionImportFailure(message: String) extends RuntimeException(message)
}

trait ExampleImport {

  @ExampleImport.PolyregionImport("example", "example.count")
  def count[T](in: Array[T], n: Int): Int =
    throw ExampleImport.PolyregionImportFailure("compiler did not replace example.count")

  @ExampleImport.PolyregionImport("example", "example.transform")
  def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit =
    throw ExampleImport.PolyregionImportFailure("compiler did not replace example.transform")
}
