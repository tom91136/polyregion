package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag

class MathSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.MathSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  {
    type V = Int
    val a: V = 1
    val b: V = 2
    val c: V = 3
    val d: V = 4
    testExpr[V](a + b)
    testExpr[V](a - b)
    testExpr[V](a * b)
    testExpr[V](a / b)

    testExpr[V](a + b + c + d)
    testExpr[V](a - b - c - d)
    testExpr[V](a * b * c * d)
    testExpr[V](a / b / c / d)

    testExpr[V](a + b - c * d / a)
  }

  {
    type V = Long
    val a: V = 1
    val b: V = 2
    val c: V = 3
    val d: V = 4
    testExpr[V](a + b)
    testExpr[V](a - b)
    testExpr[V](a * b)
    testExpr[V](a / b)

    testExpr[V](a + b + c + d)
    testExpr[V](a - b - c - d)
    testExpr[V](a * b * c * d)
    testExpr[V](a / b / c / d)

    testExpr[V](a + b - c * d / a)
  }

  {
    type V = Float
    val a: V = 1.1
    val b: V = 2.2
    val c: V = 3.3
    val d: V = 4.4
    testExpr[V](a + b)
    testExpr[V](a - b)
    testExpr[V](a * b)
    testExpr[V](a / b)

    testExpr[V](a + b + c + d)
    testExpr[V](a - b - c - d)
    testExpr[V](a * b * c * d)
    testExpr[V](a / b / c / d)

    testExpr[V](a + b - c * d / a)
  }

  {
    type V = Double
    val a: V = 1.1
    val b: V = 2.2
    val c: V = 3.3
    val d: V = 4.4
    testExpr[V](a + b)
    testExpr[V](a - b)
    testExpr[V](a * b)
    testExpr[V](a / b)

    testExpr[V](a + b + c + d)
    testExpr[V](a - b - c - d)
    testExpr[V](a * b * c * d)
    testExpr[V](a / b / c / d)

    testExpr[V](a + b - c * d / a)
  }

}
