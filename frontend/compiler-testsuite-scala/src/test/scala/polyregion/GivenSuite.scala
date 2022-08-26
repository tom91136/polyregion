package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*

class GivenSuite extends BaseSuite {

  inline def testExpr[A](inline r: A) = if (Toggles.GivenSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  trait Monoid[A] {
    def mempty: A
    def mappend(x: A, y: A): A
  }

  given Monoid[Int] = new Monoid {
    override def mappend(x: Int, y: Int): Int = x + y
    override def mempty: Int                  = 0
  }

  given Monoid[Float] = new Monoid {
    override def mappend(x: Float, y: Float): Float = x + y
    override def mempty: Float                      = 0
  }

  case class Float2(a: Float = 0, b: Float = 0) {
    infix def +(that: Float2): Float2 = Float2(a + that.a, b + that.b)
  }
  given Monoid[Float2] = new Monoid {
    override def mappend(x: Float2, y: Float2): Float2 = x + y
    override def mempty: Float2                        = Float2()
  }

  trait A {
    def a: Int
  }

  class M
//  trait B{}

  val u =
    if (sys.env.contains("a"))
      new A { def a = 1 }
    else
      new A { def a = 2 }

  {
    val a = 1
    val b = 2

    val mm = summon[Monoid[Int]]

    showExpr(u)

    ???
    //

//    testExpr {
//      val nil = mm.mempty
//      mm.mappend(nil, mm.mappend(a, b))
//    }
  }

//  {
//    val a = 1f
//    val b = 2f
//    testExpr {
//      val nil = summon[Monoid[Float]].mempty
//      summon[Monoid[Float]].mappend(nil, summon[Monoid[Float]].mappend(a, b))
//    }
//  }
//
//  {
//    val a = Float2(1f, 2f)
//    val b = Float2(2f, 3f)
//    testExpr {
//      val nil = summon[Monoid[Float2]].mempty
//      summon[Monoid[Float2]].mappend(nil, summon[Monoid[Float2]].mappend(a, b))
//    }
//  }

}
