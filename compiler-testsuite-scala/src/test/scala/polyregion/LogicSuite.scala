package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

//noinspection DoubleNegationScala,SimplifyBoolean
class LogicSuite extends BaseSuite {

  inline def testExpr[A](inline r: => A) = if (Toggles.LogicSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffload[A](r))
  }
  {

    val a = true
    val b = false

    testExpr(a)
    testExpr(b)
    testExpr(a && b)
    testExpr(a || b)
    testExpr(a == b)
    testExpr(a != b)
    testExpr(!b)
    testExpr(a && !b)
    testExpr(a || !b)
    testExpr(!(!(!b)))
  }

  // delay constant folding
  inline def repr(inline x: Boolean): Boolean = x

  testExpr(repr(true))
  testExpr(repr(false))
  testExpr(repr(true) && repr(false))
  testExpr(repr(true) || repr(false))
  testExpr(repr(true) == repr(false))
  testExpr(repr(true) != repr(false))
  testExpr(!repr(false))
  testExpr(repr(true) && !repr(false))
  testExpr(repr(true) || !repr(false))
  testExpr(!(!(!repr(false))))

  {

    val a = 1
    val b = 2
    testExpr(a > b)
    testExpr(a >= b)
    testExpr(a <= b)
    testExpr(a < b)
    testExpr(a == b)
    testExpr(a != b)
  }

}
