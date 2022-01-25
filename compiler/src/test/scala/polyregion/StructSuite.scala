package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag
import polyregion.NativeStruct

class StructSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.MathSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  case class Foo(a: Float, b: Float, c: Float)

//  test("a") {
//
//    given NativeStruct[Foo] = nativeStructOf
//    val xs                  = Buffer(Foo(1, 2, 3))
//
//
//
//    val ys = Buffer[Float](1,2)
//
//    val x = offload[Float] {
////      ys(i) = 1
//
//      xs(0).a
//
////      xs(0).a + xs(0).b + xs(0).c + 42
//    }
//
//
//    println(s"${x} $xs")
////
////    xs(0) = Foo(3, 2, 1)
////    println(xs)
////    xs(0) = Foo(1, 2, 3)
////    println(xs)
//
//  }

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
