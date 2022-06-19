package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.collection.{BuildFrom, Factory, mutable}
import _root_.scala.collection.mutable.ArrayBuffer
import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class InlineArraySuite extends BaseSuite {

  // inline def testExpr(inline name: String)(inline r: Any) = if (Toggles.InlineArraySuite) {
  //   test(name)(assertOffload(r))
  // }

  inline def assertOffloadFill[A <: AnyVal](inline n: Int, inline expected: A)(using C: ClassTag[A]) =
    if (Toggles.InlineArraySuite) {
      test(s"${C.runtimeClass}-fill-x$n=$expected") {


        val actual = Buffer.ofDim[A](n) 
        doOffload {
          val xs = Array.ofDim[A](n)
          unrollInclusive(n - 1)(i => xs(i) = expected)
          unrollInclusive(n - 1)(i =>  actual(i) = xs(i) )
          ()
        }
        
        actual.foreach(x => assertValEquals(x, expected))
      }
    }

  assertOffloadFill[Char](0, 0)
  assertOffloadFill[Byte](0, 0)
  assertOffloadFill[Short](0, 0)
  assertOffloadFill[Int](0, 0)
  assertOffloadFill[Long](0, 0)
  assertOffloadFill[Float](0, 0)
  assertOffloadFill[Double](0, 0)
  assertOffloadFill[Boolean](0, false)

  inline val FillN = 20

  Bytes.foreach(assertOffloadFill[Byte](FillN, _))
  Chars.foreach(assertOffloadFill[Char](FillN, _))
  Shorts.foreach(assertOffloadFill[Short](FillN, _))
  Ints.foreach(assertOffloadFill[Int](FillN, _))
  Longs.foreach(assertOffloadFill[Long](FillN, _))
  Floats.foreach(assertOffloadFill[Float](FillN, _))
  Doubles.foreach(assertOffloadFill[Double](FillN, _))
  Booleans.foreach(assertOffloadFill[Boolean](FillN, _))

}
