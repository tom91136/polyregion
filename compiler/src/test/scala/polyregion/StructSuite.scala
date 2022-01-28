package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag
import polyregion.NativeStruct

class StructSuite extends BaseSuite {

  case class Vec3(a: Float, b: Float, c: Float)
  object Vec3{
    def apply(a : Float) : Vec3 = Vec3(a,a,a)
  }

  // case class Vec3N(a: Float, b: Float, c: Float)(n: Int)

  case class Vec33(a: Vec3, b: Vec3, c: Vec3)

  given NativeStruct[Vec3]  = nativeStructOf
  // given NativeStruct[Vec3N] = nativeStructOf
  // given NativeStruct[Vec33] = nativeStructOf

  inline def testExpr(inline name: String)(inline r: Any) = if (Toggles.StructSuite) {
    test(name)(r)
  }

  testExpr("buffer-param") {
    val xs = Buffer.tabulate(10)(x =>
      Vec3(
        x * math.Pi.toFloat * 1, //
        x * math.Pi.toFloat * 2, //
        x * math.Pi.toFloat * 3  //
      )
    )
    assertOffload {
      xs(0).a + xs(0).b + xs(0).c
    }
  }

  testExpr("return") {
    assertOffload(
      Vec3(0.0, 1.0, 2.0)
    )
  }

  // testExpr("use-and-return") {
  //   assertOffload {
  //     val x = Vec3(0.0, 1.0, 2.0)
  //     x.a + x.b + x.c
  //   }
  // }

  // testExpr("return-ref") {
  //   val a = 0.1f
  //   val b = 0.2f
  //   val c = 0.3f
  //   assertOffload(Vec3(a, b, c))
  // }

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
