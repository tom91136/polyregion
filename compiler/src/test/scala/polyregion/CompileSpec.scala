package polyregion

import polyregion.compileTime._

class CompileSpec extends munit.FunSuite {

  inline def assertOffload[A](inline f: => A) =
    assertEquals(offload[A](f), f)

  inline def assertOffloadFloat(inline f: => Float) =
    assertEquals(java.lang.Float.floatToIntBits(offload[Float](f)), java.lang.Float.floatToIntBits(f))

  inline def assertOffloadDouble(inline f: => Double) =
    assertEquals(java.lang.Double.doubleToLongBits(offload[Double](f)), java.lang.Double.doubleToLongBits(f))

  test("const Char Char.MinValue")(assertOffload[Char](0))
  test("const Char Char.MaxValue")(assertOffload[Char](255))
  test("const Char 1")(assertOffload[Char](1))
  test("const Char 0")(assertOffload[Char](0))
  test("const Char 42")(assertOffload[Char](42))

  test("const Byte Byte.MinValue")(assertOffload[Byte](-128))
  test("const Byte Byte.MaxValue")(assertOffload[Byte](127))
  test("const Byte 1")(assertOffload[Byte](1))
  test("const Byte -1")(assertOffload[Byte](-1))
  test("const Byte 0")(assertOffload[Byte](0))
  test("const Byte 42")(assertOffload[Byte](42))

  test("const Short Short.MinValue")(assertOffload[Short](-32768))
  test("const Short Short.MaxValue")(assertOffload[Short](32767))
  test("const Short 1")(assertOffload[Short](1))
  test("const Short -1")(assertOffload[Short](-1))
  test("const Short 0")(assertOffload[Short](0))
  test("const Short 42")(assertOffload[Short](42))

  test("const Int Int.MinValue")(assertOffload[Int](0x80000000))
  test("const Int Int.MaxValue")(assertOffload[Int](0x7fffffff))
  test("const Int 1")(assertOffload[Int](1))
  test("const Int -1")(assertOffload[Int](-1))
  test("const Int 0")(assertOffload[Int](0))
  test("const Int 42")(assertOffload[Int](42))

  test("const Long Long.MinValue")(assertOffload[Long](0x8000000000000000L))
  test("const Long Long.MaxValue")(assertOffload[Long](0x7fffffffffffffffL))
  test("const Long 1")(assertOffload[Long](1))
  test("const Long -1")(assertOffload[Long](-1))
  test("const Long 0")(assertOffload[Long](0))
  test("const Long 42")(assertOffload[Long](42))

  test("const Float Float.MinValue")(assertOffloadFloat(1.4e-45f))
  test("const Float Float.MinNorm")(assertOffloadFloat(1.17549435e-38f))
  test("const Float Float.MaxValue")(assertOffloadFloat(3.4028235e+38)) // XXX 3.4028235e+38f appears to not fit!?
  test("const Float Float.NaN")(assertOffloadFloat(0.0f / 0.0f))
  test("const Float 1")(assertOffloadFloat(1))
  test("const Float -1")(assertOffloadFloat(-1))
  test("const Float 0")(assertOffloadFloat(0))
  test("const Float 42")(assertOffloadFloat(42))
  test("const Float Pi")(assertOffloadFloat(3.14159265358979323846))

  test("const Double Double.MinValue")(assertOffloadDouble(4.9e-324d))
  test("const Double Double.MinNorm")(assertOffloadDouble(2.2250738585072014e-308d))
  test("const Double Double.MaxValue")(assertOffloadDouble(1.7976931348623157e+308d))
  test("const Double Double.NaN")(assertOffloadDouble(0.0f / 0.0d))
  test("const Double 1")(assertOffloadDouble(1))
  test("const Double -1")(assertOffloadDouble(-1))
  test("const Double 0")(assertOffloadDouble(0))
  test("const Double 42")(assertOffloadDouble(42))
  test("const Double Pi")(assertOffloadDouble(3.14159265358979323846d))

  test("const Boolean true")(assertOffload(true))
  test("const Boolean false")(assertOffload(false))

  // test("const Boolean true")(assert(offloadValidate[Boolean](true)))
  // test("const Boolean false")(assert(offloadValidate[Boolean](false)))

  // test("copy capture") {
  //   val a = 1
  //   assertOffload(a)
  // }

  // test("while inc")(assertOffload {
  //   var i = 0
  //   while (i < 10) i += 1
  //   i
  // })

  test("statements")(assertOffload { 1; 2 })

  // test("const cond") {
  //   assertEquals(offload(if (true) 42 else 69), 42)
  //   assertEquals(offload(if (false) 42 else 69), 69)
  // }

  // test("copy capture as cond test") {
  //   var a = true
  //   assertEquals(offload(if (a) 42 else 69), 42)
  //   a = false
  //   assertEquals(offload(if (a) 42 else 69), 69)
  // }

  // test("copy capture expr as cond test") {
  //   var a = 10
  //   assertEquals(offload(if (a == 10) 42 else 69), 42)
  //   a = 0
  //   assertEquals(offload(if (a == 10) 42 else 69), 69)
  // }

  test("int math expr 1")(assertOffload(42 + 69))

}
