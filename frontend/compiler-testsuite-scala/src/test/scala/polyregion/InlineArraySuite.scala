package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.collection.mutable.ArrayBuffer
import _root_.scala.collection.{BuildFrom, Factory, mutable}
import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class InlineArraySuite extends BaseSuite {

  import Fixtures.*

  // inline def testExpr(inline name: String)(inline r: Any) = if (Toggles.InlineArraySuite) {
  //   test(name)(assertOffload(r))
  // }

  private inline def testFill[A <: AnyVal](inline n: Int, inline expected: A)(using C: ClassTag[A]) =
    if (Toggles.InlineArraySuite) test(s"${C.runtimeClass}-fill-x$n=$expected") {
      assertOffloadEffect {
        val actual = Buffer.ofDim[A](n)
        offload0 {
          val xs = Array.ofDim[A](n)
          unrollInclusive(n - 1)(i => xs(i) = expected)
          unrollInclusive(n - 1)(i => actual(i) = xs(i))
          ()
        }
        actual.foreach(x => assertValEquals(x, expected))
      }
    }

  testFill[Char](0, 0)
  testFill[Byte](0, 0)
  testFill[Short](0, 0)
  testFill[Int](0, 0)
  testFill[Long](0, 0)
  testFill[Float](0, 0)
  testFill[Double](0, 0)
  testFill[Boolean](0, false)

  inline val FillN = 20

  Bytes.foreach(testFill[Byte](FillN, _))
  Chars.foreach(testFill[Char](FillN, _))
  Shorts.foreach(testFill[Short](FillN, _))
  Ints.foreach(testFill[Int](FillN, _))
  Longs.foreach(testFill[Long](FillN, _))
  Floats.foreach(testFill[Float](FillN, _))
  Doubles.foreach(testFill[Double](FillN, _))
  Booleans.foreach(testFill[Boolean](FillN, _))

}
