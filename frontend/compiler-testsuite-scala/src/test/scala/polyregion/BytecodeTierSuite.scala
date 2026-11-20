package polyregion

import polyregion.helpers.Kernels

import scala.compiletime.testing.typeCheckErrors

class BytecodeTierSuite extends BaseSuite {

  private inline def testExpr[A](inline name: String)(inline r: => A) =
    test(name)(assertOffloadValue(offload1(r)))

  testExpr("static-ints")(Kernels.clamp(9, 0, 4))
  testExpr("static-double")(Kernels.lerp(1.5, 4.0, 0.25))
  testExpr("static-long")(Kernels.mix(0x12345678L))
  testExpr("loop")(Kernels.sumTo(40))
  testExpr("overload-int")(Kernels.pick(5))
  testExpr("overload-long")(Kernels.pick(5L))
  testExpr("jdk-signum")(java.lang.Integer.signum(-9))
  testExpr("jdk-compare")(java.lang.Integer.compare(-9, 3))
  testExpr("jdk-floor-mod")(java.lang.Math.floorMod(-7, 3))

  testExpr("nested")(Kernels.clampTwice(9, 0, 4))
  testExpr("recursive")(Kernels.gcd(84, 36))
  testExpr("intrinsic-inside")(Kernels.hypotenuse(3.0, 4.0))
  testExpr("overloads-together")(Kernels.overloadsTogether(9))

  testExpr("array-read") {
    val a = Array.ofDim[Int](4)
    val b = Array.ofDim[Int](4)
    var i = 0
    while (i < 4) {
      a(i) = i + 1
      b(i) = i + 5
      i += 1
    }
    Kernels.dot(a, b, 4)
  }

  testExpr("array-write") {
    val xs = Array.ofDim[Double](3)
    xs(0) = 1.0
    xs(1) = 2.0
    xs(2) = 3.0
    Kernels.scale(xs, 3, 2.5)
    xs(0) + xs(1) + xs(2)
  }

  test("unsupported bytecode is named") {
    val errors = typeCheckErrors("""
      offload0(using AssertTarget.Offload) { Kernels.throwing(3); () }
    """)
    assert(
      errors.exists(m => m.message.contains("Kernels.throwing") && m.message.contains("throw")),
      errors.map(_.message).mkString("\n")
    )
  }
}
