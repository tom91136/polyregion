package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.collection.mutable.ArrayBuffer

class LengthSuite extends BaseSuite {

  inline def testExpr[A](inline r: A) = if (Toggles.LengthSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  {
    val xs = Buffer[Int](1, 2, 3)
    testExpr{ 2 max 2; xs(0) + xs.length }
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
