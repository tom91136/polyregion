package polyregion.prism

import polyregion.ast.{PolyAst as p, *}
import polyregion.prism.compiletime.*
import polyregion.scala.intrinsics
import polyregion.scala.intrinsics.TypedBuffer

import scala.Predef as _
import scala.quoted.Quotes
import scala.quoted.Expr
import scala.quoted.Type

object StdLib {

  import _root_.scala as S
  import _root_.java as J

  class Tuple2[T1, T2](_1: T1, _2: T2)

  class Range(val start: Int, val end: Int, val step: Int) {
    // def by(step: Int): Range            = mkDef(step) // new Range(start, end, step)
    // private inline def mkDef(step: Int): Range = new Range(start, end, step)
  }

  class Class[T] {}

  class ClassTag[T] {
    // TODO
  }
  object ClassTag {
    def apply[T](runtimeClass1: java.lang.Class[_]): ClassTag[T] = new ClassTag[T]
  }

  class RichInt(val x: Int) {
    def min(that: Int)         = intrinsics.min(x, that)
    def max(that: Int)         = intrinsics.max(x, that)
    def until(end: Int): Range = new Range(x, end, 1)
  }

  class ArrayOps[A](val xs: S.Array[A]) {
    def size: Int = xs.length
  }

  class Predef {
    def intWrapper(x: Int): RichInt                  = new RichInt(x)
    def intArrayOps(xs: S.Array[Int]): ArrayOps[Int] = new ArrayOps[Int](xs)
  }

  scala.math.Pi

  object math {

    // TODO mirror fields as well
    val E: Double  = 2.7182818284590452354
    val Pi: Double = 3.14159265358979323846

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

//   object Array {
//     def ofDim[T](n1: Int)(implicit ev: ClassTag[T]): Array[T] = new Array[T](n1, intrinsics.array[T](n1))
//   }
//   class Array[T](val length: Int, val data: intrinsics.Arr[T]) extends SizedArr[T] {
// //    def length: Int                = data.length
//     def apply(i: Int): T           = data.apply(i)
//     def update(i: Int, x: T): Unit = data.update(i, x)
//   }

  object MutableSeq {
    def onDim[T](N: Int): MutableSeq[T] = new MutableSeq[T](N, intrinsics.array[T](N))
  }
  class MutableSeq[A](val length_ : Int, val data: intrinsics.TypedBuffer[A]) {
    def length: Int                = length_
    def apply(i: Int): A           = data.apply(i)
    def update(i: Int, x: A): Unit = data.update(i, x)
  }

  final def Mirrors: List[Prism] = derivePackedMirrors(
    (
      // Unsupported

      witness[S.collection.ArrayOps[_], ArrayOps[_]]( //
        { case (+ @ given Quotes, _) => throw J.lang.AssertionError("No") },
        { case (+ @ given Quotes, ys, _) => throw J.lang.AssertionError("No") }
      ),
//       // Mutable collections
//       witness[S.Array.type, Array.type](_ => Array, (_, _) => S.Array),
//       witness[S.Array, Array](
//         [X] =>
//           (xs: S.Array[X]) =>
//             Array[X](
//               xs.length,
//               new intrinsics.Arr[X] {
// //              override def length: Int                    = xs.length
//                 override def apply(i: S.Int): X             = xs(i)
//                 override def update(i: S.Int, x: X): S.Unit = xs(i) = x
//               }
//           ),
//         [X] =>
//           (ys: S.Array[X], xs: Array[X]) => {
//             var i = 0; while (i < xs.length) ys(i) = xs(i); ys
//         }
//       ),
      // witness[S.collection.mutable.Seq.type, MutableSeq.type](_ => MutableSeq, (_, _) => S.collection.mutable.Seq),
      witness[S.collection.mutable.Seq[_], MutableSeq[_]](
        { case (q @ given Quotes, xs) =>
          xs match {
            case '{ $xs: S.collection.mutable.Seq[a] } =>
              '{
                new MutableSeq[a](
                  $xs.length,
                  new intrinsics.TypedBuffer[a] {
                    override def apply(i: S.Int): a             = $xs(i)
                    override def update(i: S.Int, x: a): S.Unit = $xs(i) = x
                  }
                )
              }
          }
        },
        { case (q @ given Quotes, ys, xs) =>
          (ys, xs) match {
            case ('{ $ys: S.collection.mutable.Seq[Any] }, '{ $xs: MutableSeq[Any] }) =>
              '{ var i = 0; while (i < $xs.length) $ys(i) = $xs(i); $ys }
          }
        }
      ),

      // Immutable types, restore simply uses the original instance
      witness[S.reflect.ClassTag[_], ClassTag[_]]( //
        { case (+ @ given Quotes, _) => '{ new ClassTag[Any]() } },
        { case (+ @ given Quotes, ys, _) => ys }
      ),
      witness[J.lang.Class[_], Class[_]]( //
        { case (+ @ given Quotes, _) => '{ new Class[Any]() } },
        { case (+ @ given Quotes, ys, _) => ys }
      ),
      witness[S.collection.immutable.Range, Range](
        { case (_ @ given Quotes, r) => '{ new Range($r.start, $r.end, $r.step) } },
        { case (_ @ given Quotes, x, _) => x }
      ),
      witness[S.reflect.ClassTag.type, ClassTag.type](
        { case (_ @ given Quotes, _) => '{ ClassTag } },
        { case (_ @ given Quotes, _, _) => '{ S.reflect.ClassTag } }
      ),
      witness[S.runtime.RichInt, RichInt](
        { case (_ @ given Quotes, x) => '{ new RichInt($x.self) } },
        { case (_ @ given Quotes, x, _) => x }
      ),
      witness[S.Predef.type, Predef](
        { case (_ @ given Quotes, _) => '{ new Predef() } },
        { case (_ @ given Quotes, x, _) => x }
      ),
      witness[S.math.package$, math.type](
        { case (_ @ given Quotes, _) => '{ math } },
        { case (_ @ given Quotes, x, _) => x }
      )
    )
  )

  final def Functions: Map[p.Signature, (p.Function, Set[p.StructDef])] =
    Mirrors
      .map(_._1)
      .flatMap(m => m.functions.map(f => f -> Set(m.struct.copy(name = m.source))))
      .map { case (f, clsDeps) => f.signature -> (f, clsDeps) }
      .toMap
  final def StructDefs: Map[p.Sym, p.StructDef] =
    Mirrors
      .map(_._1)
      .map { x =>
        x.source -> x.struct.copy(name = x.source)
      }
      .toMap

  final def StructDefs2: Map[p.Sym, (p.StructDef, List[p.Sym])] =
    Mirrors
      .map(_._1)
      .map { x =>
        x.source -> (x.struct.copy(name = x.source), x.sourceParents)
      }
      .toMap

  @main def main(): Unit = {

    Functions.values.foreach { case (fn, deps) =>
      println(s"${fn.repr.linesIterator.map("\t" + _).mkString("\n")}")
    }
    StructDefs2.values.toList
      .map { case (d, xs) =>
        s"-> ${d.repr}\n${xs.map(x => s"\t${x.repr}").mkString("\n")}"
      }
      .sorted
      .foreach(println(_))

    ()
  }

}
