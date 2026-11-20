package polyregion

import scala.compiletime.testing.typeCheckErrors

class EscapeSuite extends BaseSuite {

  case class Pair(a: Int, b: Int)
  case class Counter(var n: Int, var total: Int)
  case class Wrapped(p: Pair, c: Int)

  private inline def testExpr[A](inline name: String)(inline r: => A) =
    test(name)(assertOffloadValue(offload1(r)))

  testExpr("struct-temporary") {
    val p = Pair(40, 2)
    p.a + p.b
  }

  testExpr("struct-temporary-loop-state") {
    val c = Counter(0, 0)
    while (c.n < 10) {
      c.total += c.n
      c.n += 1
    }
    c.total
  }

  testExpr("struct-temporary-nested") {
    val w = Wrapped(Pair(1, 2), 3)
    w.p.a + w.p.b + w.c
  }

  testExpr("struct-temporary-branch") {
    val flag = Pair(1, 2).a == 1
    var p    = Pair(1, 2)
    if (flag) p = Pair(7, 8) else p = Pair(9, 10)
    p.a + p.b
  }

  testExpr("array-constant-indices") {
    val xs = Array.ofDim[Int](3)
    xs(0) = 1
    xs(1) = 2
    xs(2) = 3
    xs(0) + xs(1) + xs(2)
  }

  testExpr("array-dynamic-indices") {
    val xs = Array.ofDim[Int](4)
    var i  = 0
    while (i < 4) {
      xs(i) = i * i
      i += 1
    }
    xs(3) + xs(2)
  }

  test("an array stored into a capture is refused") {
    val errors = typeCheckErrors("""
      val holder = new Array[Array[Int]](1)
      offload0(using AssertTarget.Offload) { holder(0) = Array.ofDim[Int](4); () }
    """)
    assert(
      errors.exists(m => m.message.contains("may not let an allocation escape") && m.message.contains("stored into")),
      errors.map(_.message).mkString("\n")
    )
  }
}
