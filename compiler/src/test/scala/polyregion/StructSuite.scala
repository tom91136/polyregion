package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag
import polyregion.NativeStruct

class StructSuite extends BaseSuite {

  // case class T3[A <: AnyVal](val a: A, b: A, c: A)
  // given [A <:AnyVal ]: NativeStruct[T3[A]]    = nativeStructOf

  case class Char3(val a: Char, b: Char, c: Char)
  case class Byte3(val a: Byte, b: Byte, c: Byte)
  case class Short3(val a: Short, b: Short, c: Short)
  case class Int3(val a: Int, b: Int, c: Int)
  case class Long3(val a: Long, b: Long, c: Long)
  case class Float3(val a: Float, b: Float, c: Float)
  case class Double3(val a: Double, b: Double, c: Double)
  case class Boolean3(val a: Boolean, b: Boolean, c: Boolean)

  case class Char3x3(val a: Char3, b: Char3, c: Char3)
  case class Byte3x3(val a: Byte3, b: Byte3, c: Byte3)
  case class Short3x3(val a: Short3, b: Short3, c: Short3)
  case class Int3x3(val a: Int3, b: Int3, c: Int3)
  case class Long3x3(val a: Long3, b: Long3, c: Long3)
  case class Float3x3(val a: Float3, b: Float3, c: Float3)
  case class Double3x3(val a: Double3, b: Double3, c: Double3)
  case class Boolean3x3(val a: Boolean3, b: Boolean3, c: Boolean3)

  given NativeStruct[Char3]    = nativeStructOf
  given NativeStruct[Byte3]    = nativeStructOf
  given NativeStruct[Short3]   = nativeStructOf
  given NativeStruct[Int3]     = nativeStructOf
  given NativeStruct[Long3]    = nativeStructOf
  given NativeStruct[Float3]   = nativeStructOf
  given NativeStruct[Double3]  = nativeStructOf
  given NativeStruct[Boolean3] = nativeStructOf

  inline def testExpr(inline name: String)(inline r: => Any) = if (Toggles.StructSuite) {
    test(name)(r)
  }

  testExpr("buffer-param") {
    val xs = Buffer.tabulate(10)(x =>
      Float3(
        x * math.Pi.toFloat * 1, //
        x * math.Pi.toFloat * 2, //
        x * math.Pi.toFloat * 3  //
      )
    )
    assertOffload(xs(1).a + xs(3).b + xs(5).c)
  }

  testExpr("passthrough") {
    val x = Float3(42.0, 1.0, 2.0)
    assertOffload {
      val y = x
      val z = y
      z
    }
  }

  testExpr("arg-deref-member-mix") {
    val x = Float3(42.0, 1.0, 2.0)
    assertOffload {
      val y = x // ref elision
      y.a + y.b + y.c
    }
  }

  testExpr("deref-member-mix") {
    assertOffload {
      val x = Float3(42.0, 1.0, 2.0)
      x.a + x.b + x.c
    }
  }

  testExpr("arg-deref-member") {
    val x = Float3(42.0, 1.0, 2.0)
    assertOffload {
      val y = x // ref elision
      y.c
    }
  }

  testExpr("deref-member") {
    assertOffload {
      val x = Float3(42.0, 1.0, 2.0)
      x.c
    }
  }

  testExpr("deref-member-direct") {
    assertOffload {
      Float3(42.0, 1.0, 2.0).c
    }
  }

  testExpr("return") {
    assertOffload(Float3(42.0, 1.0, 2.0))
  }

  testExpr("ctor-arg") {
    val a = 0.1f
    val b = 0.2f
    val c = 0.3f
    assertOffload(Float3(a, b, c))
  }

  testExpr("ctor") {
    assertOffload {
      val a = 0.1f
      val b = 0.2f
      val c = 0.3f
      Float3(a, b, c)
    }
  }

  // testExpr("param") {
  //   val v = Vec3(0.0, 1.0, 2.0)
  //   assertOffload {
  //     val x = v // ref elision, otherwise we'll be passing x.a, x.b, x.c as params
  //     x.a + x.b + x.c
  //   }
  // }

  // testExpr("nested-buffer-param") {
  //   val xs = Buffer.tabulate(10)(x =>
  //     Vec33(
  //       Vec3(
  //         x * math.Pi.toFloat * 1, //
  //         x * math.Pi.toFloat * 2, //
  //         x * math.Pi.toFloat * 3  //
  //       ),
  //       Vec3(
  //         x * math.Pi.toFloat * 4, //
  //         x * math.Pi.toFloat * 5, //
  //         x * math.Pi.toFloat * 6  //
  //       ),
  //       Vec3(
  //         x * math.Pi.toFloat * 7, //
  //         x * math.Pi.toFloat * 8, //
  //         x * math.Pi.toFloat * 9  //
  //       )
  //     )
  //   )
  //   assertOffload {
  //     xs(0).a.a + xs(0).b.b + xs(0).c.c
  //   }
  // }

  // testExpr("nested-return") {
  //   assertOffload(Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0)))
  // }

  // testExpr("nested-use-and-return") {
  //   assertOffload {
  //     val x = Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0))
  //     x.a.a + x.b.b + x.c.c
  //   }
  // }

  // testExpr("nested-return-ref") {
  //   val a = 0.1f
  //   val b = 0.2f
  //   val c = 0.3f
  //   assertOffload(Vec33(Vec3(a, a, a), Vec3(b, b, b), Vec3(c, c, c)))
  // }

  // testExpr("nested-param") {
  //   val v = Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0))
  //   assertOffload {
  //     val x = v // ref elision, otherwise we'll be passing x.a, x.b, x.c as params
  //     x.a.a + x.b.b + x.c.c
  //   }
  // }

}
