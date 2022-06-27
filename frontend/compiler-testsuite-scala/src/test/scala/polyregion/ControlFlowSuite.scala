package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class ControlFlowSuite extends BaseSuite {

  inline def testExpr[A <: AnyVal: ClassTag](inline name: String)(inline f: => A) = if (Toggles.ControlFlowSuite) {
    test(name)(assertOffload[A](f))
  }

  {
    case class Foo(x: Int)
    testExpr("ref-struct-if") {
      var foo = Foo(1)
      val x   = true
      if (x) {
        foo = Foo(42)
        2
      } else {
        foo = Foo(49)
        1
      }
      foo.x
    }
  }

  testExpr("stmts") { 1; 2 }

  testExpr("const-if-true") {
    (if (true) 42 else 69)
  }

  testExpr("ref-if-true") {
    val x = true
    (if (x) 42 else 69)
  }

  testExpr("while-le-inc") {
    val lim = 10
    var i   = 0
    while (i < lim)
      i += 1
    i
  }

  {
    val externalLim = 10
    testExpr("while-complex") {
      val lim = 10
      var i   = 0
      var j   = 0
      while ({
        i -= 1
        (i < lim && i != 0) && j < externalLim
      }) {
        i += 1
        j += 1
      }
      i + j
    }
  }

  testExpr("ref-if-if-true") {
    val x = true
    val y = true
    (if (x) (if (y) 1 else 2) else 3)
  }

  testExpr("const-if-if-true") {
    (if (true) (if (true) 1 else 2) else 3)
  }

}
