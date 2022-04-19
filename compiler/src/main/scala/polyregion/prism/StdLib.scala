package polyregion.prism

import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.compiletime.derivePackedMirrors1

object StdLib {

  case class Tuple2[T1, T2](_1: T1, _2: T2)

  class Range(start: Int, end: Int, step: Int) {
    def by(step: Int): Range            = mkDef(step) // new Range(start, end, step)
    private def mkDef(step: Int): Range = new Range(start, end, step)
  }

  class RichInt(private val x: Int) {
    def min(that: Int)         = math.min(x, that)
    def max(that: Int)         = math.max(x, that)
    def until(end: Int): Range = new Range(x, end, 1)
  }

  class Predef {
    def intWrapper(x: Int): RichInt = new RichInt(x)
  }

  private type ->[A, B] = (A, B)

  import _root_.scala as S

  final val Mirrors: Map[p.Sym, p.Mirror] = derivePackedMirrors1[
    (
        S.collection.immutable.Range -> Range,
//        S.runtime.RichInt -> RichInt,
//        S.Predef.type -> Predef,
        S.Tuple2[_, _] -> Tuple2[_, _]
    )
  ]

  final val Functions: Map[p.Signature, p.Function] =
    Mirrors.values.flatMap(_.functions).map(f => f.signature -> f).toMap
  final val StructDefs: Map[p.Sym, p.StructDef] =
    Mirrors.values.map(x => x.source -> x.struct.copy(name = x.source)).toMap

  @main def main(): Unit = {

    Functions.values.foreach { fn =>
      println(s"${fn.repr.linesIterator.map("\t" + _).mkString("\n")}")
    }
    StructDefs.values.foreach(f => println(s"-> $f"))

//    derivePackedMirrors1[ ((1, 2 ) ,(3,4)) ]
//    derivePackedMirrors1[M]

    ()
  }

}
