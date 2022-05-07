package polyregion.prism

import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.compiletime.derivePackedMirrors1

object StdLib {

  case class Tuple2[T1, T2](_1: T1, _2: T2)

  class Range(start: Int, end: Int, step: Int) {
    // def by(step: Int): Range            = mkDef(step) // new Range(start, end, step)
    // private inline def mkDef(step: Int): Range = new Range(start, end, step)
  }

  class RichInt(private val x: Int) {
    // def min(that: Int)         = math.min(x, that)
    // def max(that: Int)         = math.a.asInstanceOf[Int]
    // def until(end: Int): Range = new Range(x, end, 1)
  }

  class Predef {
    def intWrapper(x: Int): RichInt = new RichInt(x)
  }

  object math {
    inline def max(x: Int, y: Int): Int          = x
    inline def max(x: Long, y: Long): Long       = x
    inline def max(x: Float, y: Float): Float    = x
    inline def max(x: Double, y: Double): Double = x

    inline def min(x: Int, y: Int): Int          = x
    inline def min(x: Long, y: Long): Long       = x
    inline def min(x: Float, y: Float): Float    = x
    inline def min(x: Double, y: Double): Double = x
     def cos(x : Double) : Double = polyregion.scala.compiletime.intrinsics.add(x, 2)
  }

  private type ->[A, B] = (A, B)

  import _root_.scala as S

  final val Mirrors: Map[p.Sym, p.Mirror] = derivePackedMirrors1[
    (
        S.collection.immutable.Range -> Range,
        S.runtime.RichInt -> RichInt,
        S.Predef.type -> Predef,
        S.Tuple2[_, _] -> Tuple2[_, _],
        S.math.package$ -> math.type,
    )
  ]

  // final val Mirrors: Map[p.Sym, p.Mirror] = Map.empty

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
