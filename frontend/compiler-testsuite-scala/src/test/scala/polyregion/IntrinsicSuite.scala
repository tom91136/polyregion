package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class IntrinsicSuite extends BaseSuite {

  private inline def testExpr[A <: AnyVal](inline r: A)(using C: ClassTag[A]) = if (Toggles.IntrinsicSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  {
    val a = 1.23d
    val b = -0.12d
    testExpr[Double](math.sin(a))  // D
    testExpr[Double](math.cos(a))  // D
    testExpr[Double](math.tan(a))  // D
    testExpr[Double](math.asin(a)) // D
    testExpr[Double](math.acos(a)) // D
    testExpr[Double](math.atan(a)) // D
    testExpr[Double](math.sinh(a)) // D
    testExpr[Double](math.cosh(a)) // D
    testExpr[Double](math.tanh(a)) // D

    testExpr[Double](math.ceil(a))  // D
    testExpr[Double](math.floor(a)) // D
    testExpr[Double](math.rint(a))  // D

    testExpr[Double](math.sqrt(a))  // D
    testExpr[Double](math.cbrt(a))  // D
    testExpr[Double](math.exp(a))   // D
    testExpr[Double](math.expm1(a)) // D
    testExpr[Double](math.log(a))   // D
    testExpr[Double](math.log1p(a)) // D
    testExpr[Double](math.log10(a)) // D

    testExpr[Double](math.atan2(a, b)) // D
    testExpr[Double](math.hypot(a, b)) // D
    testExpr[Double](math.pow(a, b))   // D
    //    testExpr[Double](math.toDegrees(a)) // D // XXX JDK 1.8 is off by one in the bit pattern
    testExpr[Double](math.toRadians(a)) // D

    // TODO parameterise these

    testExpr[Double](math.abs(b)) // I, L, F, D
    //  testExpr[Double](math.min(a, b)) // I, L, F, D
    //  testExpr[Double](math.max(a, b)) // I, L, F, D
    testExpr[Double](math.signum(b)) // I, L, F, D

    //  testExpr[Double](math.copySign(a, b)) //  F, D

    //  testExpr[Double](math.scalb(a, 1)) // F(F, I), D(D, I)
    //  testExpr[Long](math.round(a)) // L(L) = id, I(F), L(D)

    {
      import _root_.scala.math.*
      testExpr[Double](sin(a))
      testExpr[Double](cos(a))
      testExpr[Double](tan(a))
    } //
    {
      import _root_.java.lang.Math.*
      testExpr[Double](sin(a))
      testExpr[Double](cos(a))
      testExpr[Double](tan(a))
    } //
    {
      import _root_.scala.math as mymath
      testExpr[Double](mymath.sin(a))
      testExpr[Double](mymath.cos(a))
      testExpr[Double](mymath.tan(a))
    } //
    {
      import _root_.java.lang.Math as mymath
      testExpr[Double](mymath.sin(a))
      testExpr[Double](mymath.cos(a))
      testExpr[Double](mymath.tan(a))
    } //
  }

}
