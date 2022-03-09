package polyregion.prism

import polyregion.ast.repr
import polyregion.prism.compiletime.reflectAndDerivePackedMirrors

object StdLib {

  class Range(start: Int, end: Int, step: Int) {
    def by(step: Int): Range = new Range(start, end, step)
  }

  class RichInt(private val x: Int) {
    def min(y: Int)          = math.min(x, y)
    def max(y: Int)          = math.max(x, y)
    def until(y: Int): Range = new Range(x, y, 1)
  }

  type ->[A, B] = (A, B)

  import _root_.scala as S

  final val Mirrors = reflectAndDerivePackedMirrors[
    (
        S.collection.immutable.Range -> Range,
        S.runtime.RichInt -> RichInt
    )
  ]

  @main def main(): Unit = {

    Mirrors.foreach { case (s, fn) =>
      println(s"$s => \n${fn.functions.map(_.repr).mkString("\n")}")
    }

    ()
  }

}
