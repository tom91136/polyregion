package polyregion

class CudaExactIntrinsicSuite extends CudaIntrinsicBase {

  {
    val e    = -2.5d
    val g    = -2.5f
    val nd   = Double.NaN
    val pz   = 0.0d
    val nz   = -0.0d
    val pzf  = 0.0f
    val iMin = Int.MinValue
    val lMin = Long.MinValue
    val m1   = -1
    val m1l  = -1L

    val iMax   = Int.MaxValue
    val lMax   = Long.MaxValue
    val one    = 1
    val oneL   = 1L
    val two31  = 1L << 31
    val tiny   = java.lang.Double.longBitsToDouble(0x000fffffffffffffL)
    val tinyF  = java.lang.Float.intBitsToFloat(0x007fffff)
    val maxD   = java.lang.Double.MAX_VALUE
    val pInf   = Double.PositiveInfinity
    val nInf   = Double.NegativeInfinity
    val tenth  = 0.1d
    val tenthF = 0.1f

    testExpr[Int](java.lang.Math.addExact(iMax, -1))
    testExpr[Long](java.lang.Math.addExact(lMin, lMax))
    testExpr[Int](java.lang.Math.subtractExact(iMin, -1))
    testExpr[Int](java.lang.Math.multiplyExact(46340, 46340))
    testExpr[Long](java.lang.Math.multiplyExact(3037000499L, 3037000499L))
    testExpr[Long](java.lang.Math.multiplyExact(lMax, -1L))
    testExpr[Int](java.lang.Math.incrementExact(iMax - 1))
    testExpr[Long](java.lang.Math.decrementExact(lMin + 1))
    testExpr[Int](java.lang.Math.negateExact(iMax))
    testExpr[Int](java.lang.Math.toIntExact(iMin.toLong))
    testExpr[Long](java.lang.Math.multiplyHigh(lMax, lMax))
    testExpr[Long](java.lang.Math.multiplyHigh(lMin, lMin))
    testExpr[Long](java.lang.Math.multiplyHigh(m1l, 5L))
    testExpr[Long](java.lang.Math.multiplyHigh(0x123456789abcdefL, -0x0fedcba987654321L))

    testFails("addExact overflow", "integer overflow")(java.lang.Math.addExact(iMax, one))
    testFails("long addExact overflow", "long overflow")(java.lang.Math.addExact(lMax, oneL))
    testFails("subtractExact overflow", "integer overflow")(java.lang.Math.subtractExact(iMin, one))
    testFails("multiplyExact overflow", "integer overflow")(java.lang.Math.multiplyExact(iMin, m1))
    testFails("long multiplyExact overflow", "long overflow")(java.lang.Math.multiplyExact(lMin, m1l))
    testFails("long multiplyExact large overflow", "long overflow")(
      java.lang.Math.multiplyExact(4294967296L, 2147483648L)
    )
    testFails("incrementExact overflow", "integer overflow")(java.lang.Math.incrementExact(iMax))
    testFails("decrementExact overflow", "long overflow")(java.lang.Math.decrementExact(lMin))
    testFails("negateExact overflow", "integer overflow")(java.lang.Math.negateExact(iMin))
    testFails("toIntExact overflow", "integer overflow")(java.lang.Math.toIntExact(two31))

    testExpr[Double](java.lang.Math.fma(tenth, 10.0d, -1.0d))
    testExpr[Float](java.lang.Math.fma(tenthF, 10.0f, -1.0f))
    testExpr[Double](java.lang.Math.fma(maxD, 2.0d, -maxD))
    testExpr[Double](java.lang.Math.fma(pInf, pz, e))

    testExpr[Double](java.lang.Math.ulp(pz))
    testExpr[Double](java.lang.Math.ulp(e))
    testExpr[Double](java.lang.Math.ulp(maxD))
    testExpr[Double](java.lang.Math.ulp(tiny))
    testExpr[Double](java.lang.Math.ulp(pInf))
    testExpr[Float](java.lang.Math.ulp(g))
    testExpr[Float](java.lang.Math.ulp(tinyF))
    testExpr[Int](java.lang.Math.getExponent(e))
    testExpr[Int](java.lang.Math.getExponent(tiny))
    testExpr[Int](java.lang.Math.getExponent(nd))
    testExpr[Int](java.lang.Math.getExponent(g))
    testExpr[Double](java.lang.Math.nextUp(pz))
    testExpr[Double](java.lang.Math.nextUp(e))
    testExpr[Double](java.lang.Math.nextDown(pz))
    testExpr[Double](java.lang.Math.nextDown(nInf))
    testExpr[Double](java.lang.Math.nextAfter(pz, nz))
    testExpr[Double](java.lang.Math.nextAfter(1.0d, 0.0d))
    testExpr[Float](java.lang.Math.nextAfter(pzf, -1.0d))
  }

}
