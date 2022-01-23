package polyregion

import polyregion.compileTime._
import scala.compiletime._
import scala.reflect.ClassTag

class IntrinsicSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.IntrinsicSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  {
    val a = 1d
    testExpr[Double](math.sin(a))           //
    testExpr[Double](java.lang.Math.sin(a)) //
    {
      import scala.math._
      testExpr[Double](sin(a))
    } //
    {
      import java.lang.Math._
      testExpr[Double](sin(a))
    } //
    {
      import scala.{math => mymath}
      testExpr[Double](mymath.sin(a))
    } //
    {
      import java.lang.{Math => mymath}
      testExpr[Double](mymath.sin(a))
    } //
  }

  {
    val a = 1d
    testExpr[Double](math.cos(a))           //
    testExpr[Double](java.lang.Math.cos(a)) //
    {
      import scala.math._
      testExpr[Double](cos(a))
    } //
    {
      import java.lang.Math._
      testExpr[Double](cos(a))
    } //
    {
      import scala.{math => mymath}
      testExpr[Double](mymath.cos(a))
    } //
    {
      import java.lang.{Math => mymath}
      testExpr[Double](mymath.cos(a))
    } //
  }

  {
    val a = 1d
    testExpr[Double](math.tan(a))           //
    testExpr[Double](java.lang.Math.tan(a)) //
    {
      import scala.math._
      testExpr[Double](tan(a))
    } //
    {
      import java.lang.Math._
      testExpr[Double](tan(a))
    } //
    {
      import scala.{math => mymath}
      testExpr[Double](mymath.tan(a))
    } //
    {
      import java.lang.{Math => mymath}
      testExpr[Double](mymath.tan(a))
    } //
  }

}
