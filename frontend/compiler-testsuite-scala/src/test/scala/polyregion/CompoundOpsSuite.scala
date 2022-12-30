package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class CompoundOpsSuite extends BaseSuite {

  private inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.CompoundOpsSuite) {
    test(name)(assertOffloadValue(offload1(r)))
  }

  // FIXME LLVM ERROR: Cannot select: t40: f32 = fmaximum t36, t39...
//  {
//    val xs = Buffer.tabulate[Double](100)(_.toDouble)
//    val n  = xs.size
//    testExpr("max-double") {
//      var max = 0.0
//      var i   = 0
//      while (i < n) {
//        max = math.max(max, xs(i))
//        i += 1
//      }
//      max
//    }
//  }
//
//  {
//    val xs = Buffer.tabulate[Float](100)(_.toFloat)
//    val n  = xs.size
//    testExpr("max-float") {
//      var max = 0f
//      var i   = 0
//      while (i < n) {
//        max = math.max(max, xs(i))
//        i += 1
//      }
//      max
//    }
//  }

  {
    val xs = Buffer.tabulate[Int](100)(identity)
    val n  = xs.size
    testExpr("max-int") {
      var max = 0
      var i   = 0
      while (i < n) {
        max = math.max(max, xs(i))
        i += 1
      }
      max
    }
  }

}
