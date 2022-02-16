package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag

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

  assertFill[Byte](FillN, -128)
  assertFill[Byte](FillN, 127)
  assertFill[Byte](FillN, 1)
  assertFill[Byte](FillN, -1)
  assertFill[Byte](FillN, 0)
  assertFill[Byte](FillN, 42)

  assertFill[Char](FillN, '\u0000')
  assertFill[Char](FillN, '\uFFFF')
  assertFill[Char](FillN, 1)
  assertFill[Char](FillN, 0)
  assertFill[Char](FillN, 42)

  assertFill[Short](FillN, -32768)
  assertFill[Short](FillN, 32767)
  assertFill[Short](FillN, 1)
  assertFill[Short](FillN, -1)
  assertFill[Short](FillN, 0)
  assertFill[Short](FillN, 42)

  assertFill[Int](FillN, 0x80000000)
  assertFill[Int](FillN, 0x7fffffff)
  assertFill[Int](FillN, 1)
  assertFill[Int](FillN, -1)
  assertFill[Int](FillN, 0)
  assertFill[Int](FillN, 42)

  assertFill[Long](FillN, 0x8000000000000000L)
  assertFill[Long](FillN, 0x7fffffffffffffffL)
  assertFill[Long](FillN, 1)
  assertFill[Long](FillN, -1)
  assertFill[Long](FillN, 0)
  assertFill[Long](FillN, 42)

  assertFill[Float](FillN, 1.4e-45f)
  assertFill[Float](FillN, 1.17549435e-38f)
  assertFill[Float](FillN, 3.4028235e+38) // XXX 3.4028235e+38f appears to not fit!
  assertFill[Float](FillN, 0.0f / 0.0f)
  assertFill[Float](FillN, 1)
  assertFill[Float](FillN, -1)
  assertFill[Float](FillN, 0)
  assertFill[Float](FillN, 42)
  assertFill[Float](FillN, 3.14159265358979323846)

  assertFill[Double](FillN, 4.9e-324d)
  assertFill[Double](FillN, 2.2250738585072014e-308d)
  assertFill[Double](FillN, 1.7976931348623157e+308d)
  assertFill[Double](FillN, 0.0f / 0.0d)
  assertFill[Double](FillN, 1)
  assertFill[Double](FillN, -1)
  assertFill[Double](FillN, 0)
  assertFill[Double](FillN, 42)
  assertFill[Double](FillN, 3.14159265358979323846d)

  assertFill[Boolean](FillN, true)
  assertFill[Boolean](FillN, false)

  // assertFill[Unit](())

}
