package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.compiletime.*

class StructCastSuite extends BaseSuite {

  private inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.StructCastSuite) {
    test(name)(assertOffloadValue(offload1(r)))
  }

  class Base(val a: Int)
  class Derived(a: Int, val b: Int) extends Base(a)

  class Box(var v: Base)

  val box     = new Box(new Base(1))
  val derived = new Derived(3, 5)

  testExpr("narrowing struct assignment") {
    val bx = box
    val d  = derived
    if (d.b > 0) bx.v = d
    bx.v.a
  }

}
