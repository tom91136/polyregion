package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.compiletime.*
import scala.reflect.ClassTag

class JvmIeeeIntrinsicSuite extends BaseSuite {

  private inline def testExpr[A <: AnyVal](inline r: A)(using C: ClassTag[A]) = if (Toggles.JvmIeeeIntrinsicSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  {
    val d   = 2.25d
    val e   = -2.5d
    val f   = 2.25f
    val g   = -2.5f
    val nd  = Double.NaN
    val nf  = Float.NaN
    val pz  = 0.0d
    val nz  = -0.0d
    val pzf = 0.0f

    val tiny   = java.lang.Double.longBitsToDouble(0x000fffffffffffffL)
    val tinyF  = java.lang.Float.intBitsToFloat(0x007fffff)
    val minD   = java.lang.Double.MIN_VALUE
    val minF   = java.lang.Float.MIN_VALUE
    val maxD   = java.lang.Double.MAX_VALUE
    val maxF   = java.lang.Float.MAX_VALUE
    val pInf   = Double.PositiveInfinity
    val nInf   = Double.NegativeInfinity
    val pInfF  = Float.PositiveInfinity
    val nInfF  = Float.NegativeInfinity
    val tenth  = 0.1d
    val tenthF = 0.1f

    testExpr[Double](java.lang.Math.fma(tenth, 10.0d, -1.0d))
    testExpr[Float](java.lang.Math.fma(tenthF, 10.0f, -1.0f))
    testExpr[Double](java.lang.StrictMath.fma(tenth, 10.0d, -1.0d))
    testExpr[Float](java.lang.StrictMath.fma(tenthF, 10.0f, -1.0f))
    testExpr[Double](java.lang.Math.fma(maxD, 2.0d, -maxD))
    testExpr[Double](java.lang.Math.fma(1.0d + 1.0d / (1L << 52), 1.0d - 1.0d / (1L << 52), -1.0d))
    testExpr[Double](java.lang.Math.fma(nd, d, e))
    testExpr[Double](java.lang.Math.fma(pInf, pz, e))
    testExpr[Double](java.lang.Math.fma(pInf, d, nInf))
    testExpr[Double](java.lang.Math.fma(d, e, pz))
    testExpr[Float](java.lang.Math.fma(f, g, 1.0f))
    testExpr[Double](1.0d / java.lang.Math.fma(pz, d, nz))

    testExpr[Double](java.lang.Math.ulp(pz))
    testExpr[Double](java.lang.Math.ulp(nz))
    testExpr[Double](java.lang.Math.ulp(1.0d))
    testExpr[Double](java.lang.Math.ulp(d))
    testExpr[Double](java.lang.Math.ulp(e))
    testExpr[Double](java.lang.Math.ulp(maxD))
    testExpr[Double](java.lang.Math.ulp(-maxD))
    testExpr[Double](java.lang.Math.ulp(minD))
    testExpr[Double](java.lang.Math.ulp(tiny))
    testExpr[Double](java.lang.Math.ulp(java.lang.Double.MIN_NORMAL))
    testExpr[Double](java.lang.Math.ulp(pInf))
    testExpr[Double](java.lang.Math.ulp(nInf))
    testExpr[Double](java.lang.Math.ulp(nd))
    testExpr[Double](java.lang.StrictMath.ulp(d))
    testExpr[Float](java.lang.Math.ulp(pzf))
    testExpr[Float](java.lang.Math.ulp(1.0f))
    testExpr[Float](java.lang.Math.ulp(g))
    testExpr[Float](java.lang.Math.ulp(maxF))
    testExpr[Float](java.lang.Math.ulp(minF))
    testExpr[Float](java.lang.Math.ulp(tinyF))
    testExpr[Float](java.lang.Math.ulp(java.lang.Float.MIN_NORMAL))
    testExpr[Float](java.lang.Math.ulp(pInfF))
    testExpr[Float](java.lang.Math.ulp(nf))

    testExpr[Int](java.lang.Math.getExponent(pz))
    testExpr[Int](java.lang.Math.getExponent(nz))
    testExpr[Int](java.lang.Math.getExponent(1.0d))
    testExpr[Int](java.lang.Math.getExponent(e))
    testExpr[Int](java.lang.Math.getExponent(maxD))
    testExpr[Int](java.lang.Math.getExponent(minD))
    testExpr[Int](java.lang.Math.getExponent(tiny))
    testExpr[Int](java.lang.Math.getExponent(java.lang.Double.MIN_NORMAL))
    testExpr[Int](java.lang.Math.getExponent(pInf))
    testExpr[Int](java.lang.Math.getExponent(nd))
    testExpr[Int](java.lang.Math.getExponent(pzf))
    testExpr[Int](java.lang.Math.getExponent(g))
    testExpr[Int](java.lang.Math.getExponent(maxF))
    testExpr[Int](java.lang.Math.getExponent(tinyF))
    testExpr[Int](java.lang.Math.getExponent(nInfF))
    testExpr[Int](java.lang.Math.getExponent(nf))
    testExpr[Int](java.lang.StrictMath.getExponent(d))

    testExpr[Double](java.lang.Math.nextUp(pz))
    testExpr[Double](java.lang.Math.nextUp(nz))
    testExpr[Double](java.lang.Math.nextUp(1.0d))
    testExpr[Double](java.lang.Math.nextUp(e))
    testExpr[Double](java.lang.Math.nextUp(-minD))
    testExpr[Double](java.lang.Math.nextUp(maxD))
    testExpr[Double](java.lang.Math.nextUp(pInf))
    testExpr[Double](java.lang.Math.nextUp(nInf))
    testExpr[Double](java.lang.Math.nextUp(nd))
    testExpr[Float](java.lang.Math.nextUp(pzf))
    testExpr[Float](java.lang.Math.nextUp(g))
    testExpr[Float](java.lang.Math.nextUp(maxF))
    testExpr[Float](java.lang.Math.nextUp(nInfF))
    testExpr[Double](java.lang.Math.nextDown(pz))
    testExpr[Double](java.lang.Math.nextDown(nz))
    testExpr[Double](java.lang.Math.nextDown(1.0d))
    testExpr[Double](java.lang.Math.nextDown(e))
    testExpr[Double](java.lang.Math.nextDown(minD))
    testExpr[Double](java.lang.Math.nextDown(-maxD))
    testExpr[Double](java.lang.Math.nextDown(nInf))
    testExpr[Double](java.lang.Math.nextDown(pInf))
    testExpr[Double](java.lang.Math.nextDown(nd))
    testExpr[Float](java.lang.Math.nextDown(pzf))
    testExpr[Float](java.lang.Math.nextDown(f))
    testExpr[Float](java.lang.Math.nextDown(-maxF))
    testExpr[Double](java.lang.Math.nextAfter(1.0d, 2.0d))
    testExpr[Double](java.lang.Math.nextAfter(1.0d, 0.0d))
    testExpr[Double](java.lang.Math.nextAfter(pz, 1.0d))
    testExpr[Double](java.lang.Math.nextAfter(nz, -1.0d))
    testExpr[Double](java.lang.Math.nextAfter(pz, nz))
    testExpr[Double](java.lang.Math.nextAfter(d, d))
    testExpr[Double](java.lang.Math.nextAfter(e, pInf))
    testExpr[Double](java.lang.Math.nextAfter(maxD, pInf))
    testExpr[Double](java.lang.Math.nextAfter(nd, 1.0d))
    testExpr[Double](java.lang.Math.nextAfter(1.0d, nd))
    testExpr[Float](java.lang.Math.nextAfter(1.0f, 2.0d))
    testExpr[Float](java.lang.Math.nextAfter(1.0f, 0.0d))
    testExpr[Float](java.lang.Math.nextAfter(pzf, -1.0d))
    testExpr[Float](java.lang.Math.nextAfter(g, g.toDouble))
    testExpr[Float](java.lang.Math.nextAfter(maxF, pInf))
    testExpr[Float](java.lang.Math.nextAfter(minF, -1.0d))
    testExpr[Float](java.lang.Math.nextAfter(nf, 1.0d))
  }

}
