package polyregion.prism

import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.compiletime.derivePackedMirrors1
import polyregion.scala.intrinsics

object StdLib {

  class Tuple2[T1, T2](_1: T1, _2: T2)

  class Range(start: Int, end: Int, step: Int) {
    // def by(step: Int): Range            = mkDef(step) // new Range(start, end, step)
    // private inline def mkDef(step: Int): Range = new Range(start, end, step)
  }

  class ClassTag[T] {
    // TODO
  }
  object ClassTag {
    def apply[T](runtimeClass1: java.lang.Class[_]): ClassTag[T] = new ClassTag[T]
  }

  class RichInt(private val x: Int) {
    def min(that: Int)         = intrinsics.min(x, that)
    def max(that: Int)         = intrinsics.max(x, that)
    def until(end: Int): Range = new Range(x, end, 1)
  }

  class Predef {
    def intWrapper(x: Int): RichInt = new RichInt(x)
  }

  object math {

    val Pi = 3.14

    def sin(x: Double): Double  = intrinsics.cos(x)
    def cos(x: Double): Double  = intrinsics.cos(x)
    def tan(x: Double): Double  = intrinsics.tan(x)
    def asin(x: Double): Double = intrinsics.asin(x)
    def acos(x: Double): Double = intrinsics.acos(x)
    def atan(x: Double): Double = intrinsics.atan(x)

    def toRadians(x: Double): Double = x * 0.017453292519943295
    def toDegrees(x: Double): Double = x * 57.29577951308232

    def atan2(y: Double, x: Double): Double = intrinsics.atan2(y, x)
    def hypot(x: Double, y: Double): Double = intrinsics.hypot(x, y)
    def ceil(x: Double): Double             = intrinsics.ceil(x)
    def floor(x: Double): Double            = intrinsics.floor(x)
    def rint(x: Double): Double             = intrinsics.rint(x)

    def round(x: Long): Long   = x
    def round(x: Float): Int   = intrinsics.round[Float, Int](x)
    def round(x: Double): Long = intrinsics.round[Double, Long](x)

    def abs(x: Int): Int       = intrinsics.abs(x)
    def abs(x: Long): Long     = intrinsics.abs(x)
    def abs(x: Float): Float   = intrinsics.abs(x)
    def abs(x: Double): Double = intrinsics.abs(x)

    def max(x: Int, y: Int): Int          = intrinsics.max(x, y)
    def max(x: Long, y: Long): Long       = intrinsics.max(x, y)
    def max(x: Float, y: Float): Float    = intrinsics.max(x, y)
    def max(x: Double, y: Double): Double = intrinsics.max(x, y)

    def min(x: Int, y: Int): Int          = intrinsics.min(x, y)
    def min(x: Long, y: Long): Long       = intrinsics.min(x, y)
    def min(x: Float, y: Float): Float    = intrinsics.min(x, y)
    def min(x: Double, y: Double): Double = intrinsics.min(x, y)

    def signum(x: Int): Int       = intrinsics.signum(x)
    def signum(x: Long): Long     = intrinsics.signum(x)
    def signum(x: Float): Float   = intrinsics.signum(x)
    def signum(x: Double): Double = intrinsics.signum(x)

    // // def floorDiv(x: Int, y: Int): Int                       = intrinsics.floorDiv(x, y)
    // // def floorDiv(x: Long, y: Long): Long                    = intrinsics.floorDiv(x, y)
    // // def floorMod(x: Int, y: Int): Int                       = intrinsics.floorMod(x, y)
    // // def floorMod(x: Long, y: Long): Long                    = intrinsics.floorMod(x, y)
    // // def copySign(magnitude: Double, sign: Double): Double   = intrinsics.copySign(magnitude, sign)
    // // def copySign(magnitude: Float, sign: Float): Float      = intrinsics.copySign(magnitude, sign)
    // // def nextAfter(start: Double, direction: Double): Double = intrinsics.nextAfter(start, direction)
    // // def nextAfter(start: Float, direction: Double): Float   = intrinsics.nextAfter(start, direction)
    // // def nextUp(d: Double): Double                           = intrinsics.nextUp(d)
    // // def nextUp(f: Float): Float                             = intrinsics.nextUp(f)
    // // def nextDown(d: Double): Double                         = intrinsics.nextDown(d)
    // // def nextDown(f: Float): Float                           = intrinsics.nextDown(f)
    // // def scalb(d: Double, scaleFactor: Int): Double          = intrinsics.scalb(d, scaleFactor)
    // // def scalb(f: Float, scaleFactor: Int): Float            = intrinsics.scalb(f, scaleFactor)

    def sqrt(x: Double): Double           = intrinsics.sqrt(x)
    def cbrt(x: Double): Double           = intrinsics.cbrt(x)
    def pow(x: Double, y: Double): Double = intrinsics.pow(x, y)
    def exp(x: Double): Double            = intrinsics.exp(x)
    def expm1(x: Double): Double          = intrinsics.expm1(x)

    // // def getExponent(f: Float): Int                          = intrinsics.getExponent(f)
    // // def getExponent(d: Double): Int                         = intrinsics.getExponent(d)

    def log(x: Double): Double   = intrinsics.log(x)
    def log1p(x: Double): Double = intrinsics.log1p(x)
    def log10(x: Double): Double = intrinsics.log10(x)
    def sinh(x: Double): Double  = intrinsics.sinh(x)
    def cosh(x: Double): Double  = intrinsics.cosh(x)
    def tanh(x: Double): Double  = intrinsics.tanh(x)

    // def ulp(x: Double): Double                              = intrinsics.ulp(x)
    // def ulp(x: Float): Float                                = intrinsics.ulp(x)
    // def IEEEremainder(x: Double, y: Double): Double         = intrinsics.IEEEremainder(x, y)
    // def addExact(x: Int, y: Int): Int                       = intrinsics.addExact(x, y)
    // def addExact(x: Long, y: Long): Long                    = intrinsics.addExact(x, y)
    // def subtractExact(x: Int, y: Int): Int                  = intrinsics.subtractExact(x, y)
    // def subtractExact(x: Long, y: Long): Long               = intrinsics.subtractExact(x, y)
    // def multiplyExact(x: Int, y: Int): Int                  = intrinsics.multiplyExact(x, y)
    // def multiplyExact(x: Long, y: Long): Long               = intrinsics.multiplyExact(x, y)
    // def incrementExact(x: Int): Int                         = intrinsics.incrementExact(x)
    // def incrementExact(x: Long)                             = intrinsics.incrementExact(x)
    // def decrementExact(x: Int)                              = intrinsics.decrementExact(x)
    // def decrementExact(x: Long)                             = intrinsics.decrementExact(x)
    // def negateExact(x: Int)                                 = intrinsics.negateExact(x)
    // def negateExact(x: Long)                                = intrinsics.negateExact(x)
    // def toIntExact(x: Long): Int                            = intrinsics.toIntExact(x)

  }

  object Array {
    def ofDim[T](n1: Int)(implicit ev: ClassTag[T]): scala.Array[T] = intrinsics.array[T](n1)
  }

  private type ->[A, B] = (A, B)

  import _root_.scala as S

  final def Mirrors: Map[p.Sym, p.Mirror] = derivePackedMirrors1[
    (
        S.collection.immutable.Range -> Range,
        S.Array.type -> Array.type,
        S.runtime.RichInt -> RichInt,
        S.Predef.type -> Predef,
        S.Tuple2[_, _] -> Tuple2[_, _],
        S.math.package$ -> math.type,
        S.reflect.ClassTag[_] -> ClassTag[_],
        S.reflect.ClassTag.type -> ClassTag.type
    )
  ]

  // final val Mirrors: Map[p.Sym, p.Mirror] = Map.empty

  final def Functions: Map[p.Signature, p.Function] =
    Mirrors.values.flatMap(_.functions).map(f => f.signature -> f).toMap
  final def StructDefs: Map[p.Sym, p.StructDef] =
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
