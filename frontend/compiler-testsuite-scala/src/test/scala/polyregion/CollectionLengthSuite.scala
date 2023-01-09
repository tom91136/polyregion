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

}
