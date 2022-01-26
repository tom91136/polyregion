package polyregion

import polyregion.compiletime.*

import scala.compiletime.*

class CaptureSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.CaptureSuite) {
    test(name)(assertOffload(r))
  }

  {

    object ConstA {
      final val MyConstantA = 42
      object ConstB{
        final val MyConstantB = 43
      }
    }

    {
      import ConstA._
      testCapture("constant-in-scope-x1") {
        MyConstantA
      }
      val a = MyConstantA
      class Bar{
        val e = 3
      }
      class Foo{
        val b = 1
        val c = 2
        val d = new Bar

      }
      val x = new Foo


      case class Node(elem: Int, next: Option[Node])
      case class A(ax: Int)
      given NativeStruct[A] = nativeStructOf
      val buffer  = Buffer(A(1), A(2))
      val node = Node(1, Some(Node(2, None)))
      val (b1, b2) = (1,2)

      // buffer
      // a
      // x.b
      // x.c
      // x.d.e
      // MyConstantA
      // MyConstantB
      // node.elem
      // b1

      testCapture("constant-in-scope-x2") {
        val u =  MyConstantA
        val v =  ConstB.MyConstantB

        val nodeA = node.elem

//        val b = buffer(u).ax
//        val c = buffer(0).ax

//        val m = x.d.e
//        val y1 = b1

//        MyConstantA + a + x.b + x.c
      }
    }

//    testCapture("constant-qualified-x1") {
//      Const.MyConstant
//    }
//    val a = Const.MyConstant
//    testCapture("constant-qualified-x2") {
//      Const.MyConstant + a
//    }
  }

//  testCapture("scala2x-constant-x1") {
//    Int.MaxValue
//  }
//
//  testCapture("scala2x-constant-x2") {
//    Int.MaxValue - Int.MinValue
//  }

//  {
//    val A: Int = 42
//    val B: Int = 43
//    testCapture("val-in-scope") {
//      A + B
//    }
//  }
//
//  {
//    val A: Int = 42
//    val B: Int = 43
//    testCapture("val-in-scope-ref") {
//      val a = A
//      val b = B
//      val c = a + b
//      c
//    }
//  }
//
//  {
//    case class X(n: Int)
//    val x = X(42)
//    testCapture("val-qualified-ref") {
//      val a = x.n
//      val b = x.n
//      val c = a + b
//      c
//    }
//  }

}
