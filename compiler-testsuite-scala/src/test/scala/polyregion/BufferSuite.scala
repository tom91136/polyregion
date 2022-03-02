package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class BufferSuite extends BaseSuite {

  final val FillN = 20

  inline def assertFill[A <: AnyVal](inline n: Int, inline expected: A)(using C: ClassTag[A]) = if (
    Toggles.BufferSuite
  ) {
    test(s"${C.runtimeClass}-fill-x$n=$expected") {
      val xs = Buffer.ofDim[A](n)
      assertEquals(
        doOffload {
          unrollInclusive(n - 1)(i => xs(i) = expected)
        },
        ()
      )
      xs.foreach(assertValEquals(_, expected))
    }
    test(s"${C.runtimeClass}-fill-x-return$n=$expected") {
      val xs = Buffer.ofDim[A](n)
      // make sure we get the same backing buffer instance back
      assert(
        doOffload {
          unrollInclusive(n - 1)(i => xs(i) = expected)
          xs
        }.backingBuffer eq xs.backingBuffer
      )
      xs.foreach(assertValEquals(_, expected))
    }

    if (n != 0) {
      test(s"${C.runtimeClass}-fill-x-return-at-0$n=$expected") {
        val xs = Buffer.ofDim[A](n)
        assertValEquals(
          doOffload {
            unrollInclusive(n - 1)(i => xs(i) = expected)
            xs(0)
          },
          xs(0)
        )
      }
    }
  }

  assertFill[Char](0, 0)
  assertFill[Byte](0, 0)
  assertFill[Short](0, 0)
  assertFill[Int](0, 0)
  assertFill[Long](0, 0)
  assertFill[Float](0, 0)
  assertFill[Double](0, 0)
  assertFill[Boolean](0, false)
  // FIXME segfaults
  // assertFill[Unit](0, ())

  Bytes.foreach(assertFill[Byte](FillN, _))
  Chars.foreach(assertFill[Char](FillN, _))
  Shorts.foreach(assertFill[Short](FillN, _))
  Ints.foreach(assertFill[Int](FillN, _))
  Longs.foreach(assertFill[Long](FillN, _))
  Floats.foreach(assertFill[Float](FillN, _))
  Doubles.foreach(assertFill[Double](FillN, _))
  Booleans.foreach(assertFill[Boolean](FillN, _))


}
