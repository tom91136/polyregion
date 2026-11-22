package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.compiletime.*

class CudaIntrinsicSuite extends CudaIntrinsicBase {

  {
    val d   = 2.25d
    val e   = -2.5d
    val h   = 0.49999999999999994d
    val f   = 2.25f
    val g   = -2.5f
    val nd  = Double.NaN
    val nf  = Float.NaN
    val pz  = 0.0d
    val nz  = -0.0d
    val pzf = 0.0f
    val nzf = -0.0f

    testExpr[Double](java.lang.Math.sqrt(d))
    testExpr[Double](java.lang.StrictMath.sqrt(d))
    testExpr[Double](scala.math.sqrt(d))

    testExpr[Int](java.lang.Math.abs(-7))
    testExpr[Long](java.lang.Math.abs(-7L))
    testExpr[Float](java.lang.Math.abs(g))
    testExpr[Double](java.lang.Math.abs(e))

    testExpr[Int](java.lang.Math.min(3, -4))
    testExpr[Int](java.lang.Math.max(3, -4))
    testExpr[Long](java.lang.Math.min(3L, -4L))
    testExpr[Long](java.lang.Math.max(3L, -4L))
    testExpr[Float](java.lang.Math.min(f, g))
    testExpr[Float](java.lang.Math.max(f, g))
    testExpr[Double](java.lang.Math.min(d, e))
    testExpr[Double](java.lang.Math.max(d, e))
    testExpr[Float](java.lang.Math.min(nf, f))
    testExpr[Float](java.lang.Math.max(f, nf))
    testExpr[Double](java.lang.Math.min(nd, d))
    testExpr[Double](java.lang.Math.max(d, nd))
    testExpr[Double](1.0d / java.lang.Math.min(pz, nz))
    testExpr[Double](1.0d / java.lang.Math.max(pz, nz))
    testExpr[Float](1.0f / java.lang.Math.min(pzf, nzf))
    testExpr[Float](1.0f / java.lang.Math.max(pzf, nzf))
    testExpr[Int](scala.math.max(3, -4))
    testExpr[Double](scala.math.min(d, e))

    testExpr[Int](java.lang.Math.floorMod(-7, 3))
    testExpr[Int](java.lang.Math.floorMod(7, -3))
    testExpr[Int](java.lang.Math.floorMod(7, 3))
    testExpr[Long](java.lang.Math.floorMod(-7L, 3L))
    testExpr[Int](java.lang.Math.floorMod(-7L, 3))
    testExpr[Int](java.lang.Math.floorDiv(-7, 3))
    testExpr[Long](java.lang.Math.floorDiv(7L, -3L))

    val zero  = 0
    val zeroL = 0L
    testFails("floorDiv by zero")(java.lang.Math.floorDiv(7, zero))
    testFails("floorMod by zero")(java.lang.Math.floorMod(7, zero))
    testFails("long floorDiv by zero")(java.lang.Math.floorDiv(7L, zeroL))
    testFails("long floorMod by zero")(java.lang.Math.floorMod(7L, zeroL))
    testFails("division by zero")(7 / zero)
    testFails("remainder by zero")(7 % zero)
    testFails("long division by zero")(7L / zeroL)
    testFails("long remainder by zero")(7L % zeroL)
    testFails("unused division by zero") { 7 / zero; () }
    testFails("unused remainder by zero") { 7 % zero; () }
    testFails("unused long division by zero") { 7L / zeroL; () }
    testFails("unused long remainder by zero") { 7L % zeroL; () }

    testExpr[Double](java.lang.Math.signum(e))
    testExpr[Float](java.lang.Math.signum(g))
    testExpr[Int](java.lang.Integer.signum(-12))
    testExpr[Int](java.lang.Long.signum(-12L))

    testExpr[Long](java.lang.Math.round(d))
    testExpr[Long](java.lang.Math.round(e))
    testExpr[Long](java.lang.Math.round(h))
    testExpr[Int](java.lang.Math.round(f))
    testExpr[Int](java.lang.Math.round(g))
    testExpr[Long](scala.math.round(e))

    testExpr[Double](java.lang.Math.copySign(d, e))
    testExpr[Double](java.lang.Math.toRadians(d))
    testExpr[Double](java.lang.Math.toDegrees(d))

    testExpr[Int](java.lang.Integer.bitCount(0x55aa55aa))
    testExpr[Int](java.lang.Long.bitCount(0x55aa55aa55aa55aaL))
    testExpr[Int](java.lang.Integer.numberOfLeadingZeros(0x00ff0000))
    testExpr[Int](java.lang.Integer.numberOfLeadingZeros(0))
    testExpr[Int](java.lang.Long.numberOfLeadingZeros(1L << 40))
    testExpr[Int](java.lang.Long.numberOfLeadingZeros(0L))
    testExpr[Int](java.lang.Integer.numberOfTrailingZeros(0x00ff0000))
    testExpr[Int](java.lang.Long.numberOfTrailingZeros(1L << 40))
    testExpr[Int](java.lang.Integer.rotateLeft(0x80000001, 5))
    testExpr[Int](java.lang.Integer.compare(3, 9))
    testExpr[Int](java.lang.Float.floatToRawIntBits(f))
    testExpr[Boolean](java.lang.Double.isNaN(nd))
    testExpr[Boolean](java.lang.Double.isInfinite(d))

    val iMin = Int.MinValue
    val lMin = Long.MinValue
    val m1   = -1
    val m1l  = -1L
    val nanA = java.lang.Float.intBitsToFloat(0x7fc00001)
    val nanB = java.lang.Float.intBitsToFloat(0x7fc00002)
    val nanC = java.lang.Double.longBitsToDouble(0x7ff8000000000001L)
    val nanD = java.lang.Double.longBitsToDouble(0x7ff8000000000002L)
    val neg3 = -3
    testExpr[Int](iMin / m1)
    testExpr[Int](iMin % m1)
    testExpr[Long](lMin / m1l)
    testExpr[Long](lMin % m1l)

    testExpr[Int](java.lang.Math.floorDiv(iMin, m1))
    testExpr[Int](java.lang.Math.floorMod(iMin, m1))
    testExpr[Long](java.lang.Math.floorDiv(lMin, m1l))
    testExpr[Long](java.lang.Math.floorMod(lMin, m1l))
    testExpr[Long](java.lang.Math.floorDiv(lMin, m1))
    testExpr[Int](java.lang.Math.floorMod(lMin, m1))
    testExpr[Int](java.lang.Math.abs(iMin))
    testExpr[Long](java.lang.Math.abs(lMin))
    testExpr[Int](java.lang.Integer.lowestOneBit(iMin))
    testExpr[Long](java.lang.Long.lowestOneBit(lMin))

    testExpr[Int](java.lang.Integer.bitCount(m1))
    testExpr[Int](java.lang.Long.bitCount(m1l))
    testExpr[Int](java.lang.Integer.numberOfLeadingZeros(m1))
    testExpr[Int](java.lang.Integer.numberOfLeadingZeros(iMin))
    testExpr[Int](java.lang.Integer.numberOfTrailingZeros(iMin))
    testExpr[Int](java.lang.Long.numberOfTrailingZeros(lMin))
    testExpr[Int](java.lang.Long.numberOfLeadingZeros(m1l))
    testExpr[Int](java.lang.Integer.rotateLeft(0x80000001, neg3))
    testExpr[Int](java.lang.Integer.rotateRight(0x80000001, neg3))
    testExpr[Long](java.lang.Long.rotateLeft(lMin, neg3))

    testExpr[Int](java.lang.Float.floatToRawIntBits(java.lang.Math.min(nanA, nanB)))
    testExpr[Int](java.lang.Float.floatToRawIntBits(java.lang.Math.max(nanA, nanB)))
    testExpr[Long](java.lang.Double.doubleToRawLongBits(java.lang.Math.min(nanC, nanD)))
    testExpr[Long](java.lang.Double.doubleToRawLongBits(java.lang.Math.max(nanC, nanD)))
    testExpr[Boolean](java.lang.Float.isNaN(nanA))
    testExpr[Boolean](java.lang.Double.isInfinite(Double.PositiveInfinity))
    testExpr[Boolean](java.lang.Double.isFinite(nd))
    testExpr[Int](java.lang.Math.round(nf))
    testExpr[Long](java.lang.Math.round(nd))
  }

}
