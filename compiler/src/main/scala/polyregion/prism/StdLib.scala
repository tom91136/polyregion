package polyregion.prism

import polyregion.ast.{PolyAst as p, repr}
import polyregion.prism.compiletime.derivePackedMirrors1

object StdLib {

  class Range(start: Int, end: Int, step: Int) {
    def by(step: Int): Range            = mkDef(step) // new Range(start, end, step)
    private def mkDef(step: Int): Range = new Range(start, end, step)
  }

  class RichInt(private val self: Int) {
    def min(y: Int)          = math.min(self, y)
    def max(y: Int)          = math.max(self, y)
    def until(y: Int): Range = new Range(self, y, 1)
  }

  class Predef {
    def intWrapper(i: Int): RichInt = new RichInt(i)
  }

  class AAA {
    def bbb(): BBB = ???
  }

  class BBB(i: Int) {
    def aaa(): AAA = ???
  }

  class B_(i: Int) {
    def aaa(): A_ = new A_(1)
  }

  class A_(i: Int) {
    def bbb(): B_ = new B_(2)
  }

  type ->[A, B] = (A, B)

  import _root_.scala as S

  final val Mirrors: Map[p.Sym, p.Mirror] = derivePackedMirrors1[
    (
        S.collection.immutable.Range -> Range,
        S.runtime.RichInt -> RichInt,
        S.Predef.type -> Predef
//        AAA -> A_  ,
//        BBB-> B_  ,
    )
  ]

  final val Functions: List[p.Function]   = Mirrors.values.toList.flatMap(_.functions)
  final val StructDefs: List[p.StructDef] = Mirrors.values.toList.map(_.struct)

  @main def main(): Unit = {

    Functions.foreach { case (fn) =>
      println(s"${fn.repr.linesIterator.map("\t" + _).mkString("\n")}")
    }

//    derivePackedMirrors1[ ((1, 2 ) ,(3,4)) ]
//    derivePackedMirrors1[M]

    ()
  }

}
