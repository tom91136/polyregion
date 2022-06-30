package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class CompoundCaptureSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.CompoundCaptureSuite) {
    test(name)(assertOffload(r))
  }

  object Foo {
    var x                         = 10
    def bar                       = x + 1
    def baz                       = bar + 1
    def nBar(n: Int)              = n * bar + baz
    inline def nBarInline(n: Int) = n * bar + baz + nBar(baz)
    object Bar {
      val y1 = 3
      object Baz {
        val y2 = 4
      }
    }
  }

  object Bar { val y = Foo.bar }
  val a         = 10
  val b         = 20
  val x         = a
  private def c = (a + b) * 2

  {

    object Local {
      val g      = Foo.bar + Foo.baz + a + b + c + x
      val levelB = LevelA.LevelB
      object LevelA {
        val a1 = Foo.bar
        object LevelB { val a2 = g + a1 + x }
      }
    }

    val l = Local

    testCapture("complex") {
      val objRef = Foo
      val c      = Local.LevelA.LevelB
      val c2     = Foo.Bar.Baz
      val y1     = Foo.Bar.y1
      val m      = c.a2
      val n      = l
      val a40    = l.LevelA
      val a41    = l.LevelA.a1

      val m1      = l.levelB.a2
      val i       = n.g
      val objRef2 = objRef
      val a       = Local
      val a1      = a.LevelA
      val a2      = a.LevelA.a1
      val m2      = Foo.bar + objRef.nBar(2) + objRef2.nBarInline(42)
      val a3      = Local.LevelA.a1

      // aliases
      val alias1      = this
      val alias2      = alias1
      val alias3      = alias2
      val aliasX      = alias3
      val aliasY      = alias3
      val aliasResult = aliasX.a + aliasX.b + aliasY.a + aliasY.b

      val y =
        objRef.bar + Foo.baz + x + b + m + Local.g + x + Local.LevelA.a1 + Local.LevelA.LevelB.a2 + aliasResult + m1
      val z = 42 + x + y + Foo.nBar(y) + Foo.nBarInline(
        y
      ) + Bar.y + this.a + this.b + i + objRef2.x + a1.LevelB.a2 + y1 + c2.y2 + a40.a1
      val out = a3 + m2 + a2 + m + a41 + z
      Foo.bar + out + Foo.nBarInline(m)
    }

  }

}
