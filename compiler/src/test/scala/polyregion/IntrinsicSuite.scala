package polyregion

import polyregion.compileTime._
import scala.compiletime._
import scala.reflect.ClassTag

class IntrinsicSuite extends BaseSuite {


  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.IntrinsicSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }
 

  import scala.math._
  val a = 1d
  testExpr[Double](cos(a))

  import scala.{math => mymath}
  testExpr[Double](mymath.cos(a))

  testExpr[Double](math.cos(a))



}
