package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

//noinspection DoubleNegationScala,SimplifyBoolean
class LogicSuite extends BaseSuite {

  private inline def testExpr[A](inline r: => A) = if (Toggles.LogicSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  // Test both normal and inverse of any expr to weed out any false negatives
  inline def testExprNormalAndInv(inline r: => Boolean) = {
    testExpr(r)
    testExpr(!r)
  }

  {

    val a = true
    val b = false

    testExprNormalAndInv(a)
    testExprNormalAndInv(b)
    testExprNormalAndInv(a && b)
    testExprNormalAndInv(a || b)
    testExprNormalAndInv(a == b)
    testExprNormalAndInv(a != b)
    testExprNormalAndInv(!b)
    testExprNormalAndInv(a && !b)
    testExprNormalAndInv(a || !b)
    testExprNormalAndInv(!(!(!b)))
  }

  // delay constant folding
  inline def repr(inline x: Boolean): Boolean = x

  testExprNormalAndInv(repr(true))
  testExprNormalAndInv(repr(false))
  testExprNormalAndInv(repr(true) && repr(false))
  testExprNormalAndInv(repr(true) || repr(false))
  testExprNormalAndInv(repr(true) == repr(false))
  testExprNormalAndInv(repr(true) != repr(false))
  testExprNormalAndInv(!repr(false))
  testExprNormalAndInv(repr(true) && !repr(false))
  testExprNormalAndInv(repr(true) || !repr(false))
  testExprNormalAndInv(!(!(!repr(false))))

  {

    val a = 1
    val b = 2
    testExprNormalAndInv(a > b)
    testExprNormalAndInv(a >= b)
    testExprNormalAndInv(a <= b)
    testExprNormalAndInv(a < b)
    testExprNormalAndInv(a == b)
    testExprNormalAndInv(a != b)
  }

}
