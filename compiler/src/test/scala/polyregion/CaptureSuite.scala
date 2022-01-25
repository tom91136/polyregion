package polyregion

import polyregion.compiletime.*

import scala.compiletime.*

class CaptureSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.CaptureSuite) {
    test(name)(assertOffload(r))
  }

  {

    object Const {
      final val MyConstant = 42
      object Const2{
        final val myConstant2 = 43
      }
    }

    {
      import Const._
      testCapture("constant-in-scope-x1") {
        MyConstant
      }
      val a = MyConstant
      class V{val b = 1}
      val x = new V


      case class Node(elem: Int, next: Option[Node])

      case class A(ax: Int)
      given NativeStruct[A] = nativeStructOf
      val buffer  = Buffer(A(1), A(2))

      val node = Node(1, Some(Node(2, None)))

      // buffer
      // x.b
      // MyConstant
      // myConstant2
      // node.elem
      // node.next.isDefined

      testCapture("constant-in-scope-x2") {
        val u =  MyConstant
        val v =  Const2.myConstant2

        val nodeA = node.elem
        val nodeB = node.next.isDefined



        val b = buffer(u).ax
        val c = buffer(0).ax
        MyConstant + a + x.b
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
