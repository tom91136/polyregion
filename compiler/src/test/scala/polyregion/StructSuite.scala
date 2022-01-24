package polyregion

import polyregion.compileTime._
import scala.compiletime._
import scala.reflect.ClassTag
import polyregion.Runtime.NativeStruct

class StructSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.MathSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  // case class Foo(a : Int, b : Short, c : Float) derives NativeStruct


  showOffsets[Int]

 inline def testValueReturn[A](inline r: A) = if (Toggles.StructSuite) {
    test(s"${r.getClass}-const=$r")(assertOffload[A](r))
    val x: A = r
    test(s"${r.getClass}-ref1=$r")(assertOffload[A](x))
    val y: A = x
    test(s"${r.getClass}-ref2=$r")(assertOffload[A](y))
    val z: A = y
    test(s"${r.getClass}-ref3=$r")(assertOffload[A](z))
  }

  // testValueReturn[Foo](Foo(1, 2, 3f))

 

}
