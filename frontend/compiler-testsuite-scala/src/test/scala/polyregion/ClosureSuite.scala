package polyregion

import polyregion.scalalang.compiletime.*

class ClosureSuite extends BaseSuite {

  private inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.ClosureSuite) {
    test(name)(assertOffloadValue(offload1(r)))
  }

  testExpr("lambda-value") {
    val f = (x: Int) => x + 1
    f(41)
  }
  testExpr("lambda-value-capture") {
    val y = 2
    val f = (x: Int) => x + y
    f(40)
  }
  testExpr("lambda-value-two-args") {
    val f = (x: Int, y: Int) => x * y
    f(6, 7)
  }
  testExpr("lambda-value-alias") {
    val f = (x: Int) => x + 1
    val g = f
    g(41)
  }
  testExpr("lambda-applied")(
    ((x: Int) => x + 1)(41)
  )
  testExpr("generic-method") {
    def id[A](a: A): A = a
    id(42)
  }
  testExpr("null-literal") {
    val xs: Array[Int] = null
    1
  }
  testExpr("string-literal") {
    val s = "abc"
    3
  }
}
