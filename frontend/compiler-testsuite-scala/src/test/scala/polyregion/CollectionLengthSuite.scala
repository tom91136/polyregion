package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.collection.mutable.ArrayBuffer
import scala.compiletime.*

class CollectionLengthSuite extends BaseSuite {

  private inline def testExpr[A](inline r: A) = if (Toggles.LengthSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  case class U(a: Int)
  case class V(a: Int, u: U)

  {
    val xs = Buffer[Float](41, 42, 43)
    // val n = 42.toShort

    val v = V(1, U(2))
    // testExpr{
    //   val x = v
    //   x.a+x.a + x.u.a
    //  }
//    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
  }

  {
    val xs = Array[Int](1, 2, 3)
//    testExpr {
//      xs.size
//    }
    //    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
  }
//
//  {
//    val xs = ArrayBuffer[Int](1, 2, 3)
//    testExpr(xs.size)
//    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
//  }

  class Base(val a: Int) {
    def foo(n: Int): Int = a + n
  }

  class ClassA(a: Int) extends Base(a) {
    override def foo(n: Int): Int = 42 + n + super.foo(n)
  }

  class ClassB(val b: Int) extends Base(42) {
    override def foo(n: Int): Int = b
  }

  {
    val i = 0
    testExpr {

      val m = ClassB(123)

      val o = if (i == 0) ClassA(2) else ClassB(9)

      val n = Base(2)



      o.foo(2) + n.foo(42) + m.foo(12345) + m.foo(42)

    }
  }

}
