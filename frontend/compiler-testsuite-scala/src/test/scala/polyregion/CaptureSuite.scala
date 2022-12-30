package polyregion

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class CaptureSuite extends BaseSuite {

  private inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.CaptureSuite) {
    test(name)(assertOffloadValue(offload1(r)))
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
      case class CaseClassA(b: Int, c: Int)
//      given NativeStruct[A] = nativeStructOf
      val as       = Array(CaseClassA(1, 2), CaseClassA(3, 4))
      val node     = Node(1, Some(Node(2, None)))
      val (t1, t2) = (1, 2)
      val b        = as(0).b
      val c        = as(1).c
      testCapture("complex-captures") {
        val u = MyConstantA
        val v = ConstB.MyConstantB
        val e = node.elem
        val m = x.d.e
        val y = t1
        u + v + e + b + c + m + y
      }
    }
  }

  {
    case class Foo(x: Int, y: Int)
    object Foo {
      final val FooConst = new Foo(41, 42)
    }

    object NotFooCompanion {
      final val FooConst = Foo(42, 43)
    }

    testCapture("constant-of-struct") {
      val u = NotFooCompanion.FooConst
      u.x + u.y
    }

    testCapture("constant-of-struct-companion") {
      val u = Foo.FooConst
      u.x + u.y
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

  {
    val A = 1.0
    testCapture("arbitrary-blocks") {
      val m = {
        val p = A * 3.0
        val n = {
          val o = p + 2.0
          o * 2.0
        }
        p * 2.0 + n
      }
      val n = m * A
      n
    }

  }

}
