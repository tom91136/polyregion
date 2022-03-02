package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*

class CaptureSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.CaptureSuite) {
    test(name)(assertOffload(r))
  }

  {

    object ConstA {
      final val MyConstantA = 42
      object ConstB {
        final val MyConstantB = 43
      }
    }

    {

      import ConstA.*

      testCapture("constant-in-scope-x1") {
        MyConstantA
      }
      val a = MyConstantA
      class Bar {
        val e = 3
      }
      class Foo {
        val b = 1
        val c = 2
        val d = new Bar

      }
      val x = new Foo
      case class Node(elem: Int, next: Option[Node])
      case class A(b: Int, c: Int)
      given NativeStruct[A] = nativeStructOf
      val bufferOfA         = Buffer(A(1, 2), A(3, 4))
      val node              = Node(1, Some(Node(2, None)))
      val (t1, t2)          = (1, 2)
      testCapture("complex-captures") {
        val u = MyConstantA
        val v = ConstB.MyConstantB
        val e = node.elem
        val b = bufferOfA(0).b
        val c = bufferOfA(1).c
        val m = x.d.e
        val y = t1
        u + v + e + b + c + m + y
      }
    }
  }

  {

    object Const {
      final val MyConstant = 42
    }

    testCapture("constant-qualified-x1") {
      Const.MyConstant
    }
    val a = Const.MyConstant
    testCapture("constant-qualified-x2") {
      Const.MyConstant + a
    }
  }

  testCapture("scala2x-constant-x1") {
    Int.MaxValue
  }

  testCapture("scala2x-constant-x2") {
    Int.MaxValue - Int.MinValue
  }

  {
    val A: Int = 42
    val B: Int = 43
    testCapture("val-in-scope") {
      A + B
    }
  }

  {
    val A: Int = 42
    val B: Int = 43
    testCapture("val-in-scope-ref") {
      val a = A
      val b = B
      val c = a + b
      c
    }
  }

  {
    case class X(n: Int)
    val x = X(42)
    testCapture("val-qualified-ref") {
      val a = x.n
      val b = x.n
      val c = a + b
      c
    }
  }

}
