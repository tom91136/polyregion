package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*

class ControlFlowSuite extends BaseSuite {

  inline def testExpr[A](inline name: String)(inline f: => A) = if (Toggles.ControlFlowSuite) {
    test(name)(assertOffload[A](f))
  }

  {
    case class Foo(x: Int)
    testExpr("ref-struct-if") {
      var foo = Foo(1)
      val x   = true
      if (x) { foo = Foo(42) 
      2}
      else { foo = Foo(49)
      1 }
      foo.x
    }
  }

  // testExpr("stmts") { 1; 2 }

  // testExpr("const-if-true") {
  //   (if (true) 42 else 69)
  // }

  // testExpr("ref-if-true") {
  //   val x = true
  //   (if (x) 42 else 69)
  // }

  // testExpr("while-le-inc") {
  //   val lim = 10
  //   var i   = 0
  //   while (i < lim)
  //     i += 1
  //   i
  // }

  // {
  //   val externalLim = 10
  //   testExpr("while-complex") {
  //     val lim = 10
  //     var i   = 0
  //     var j   = 0
  //     while ({
  //       i -= 1
  //       (i < lim && i != 0) && j < externalLim
  //     }) {
  //       i += 1
  //       j += 1
  //     }
  //     i + j
  //   }
  // }

  // testExpr("ref-if-if-true") {
  //   val x = true
  //   val y = true
  //   (if (x) (if (y) 1 else 2) else 3)
  // }

  // testExpr("const-if-if-true") {
  //   (if (true) (if (true) 1 else 2) else 3)
  // }

  // test("const-if-false") {
  //   assertOffload((if (false) 42 else 69))
  // }

  // test("while inc")(assertOffload {
  //   var i = 1
  //   while (i < 10) i += 1
  //   i
  // })
  // case class V(n : Int)
  //  val a : Int = 42
  //  def inv = 42

  // def go = {
  //   // Select
  //    val b : Int = 42
  //    val c : Int = 42
  //    val v = V(1)
  // offload{
  //   // var counter = 1
  //   // while (counter < 10) counter += 1
  //   val in4 = v.n
  //   val in3 = Int.MaxValue
  //   val inInv = inv
  //   val in2 = a
  //   val in1 = b
  //   val u = in3
  //   val u2 = in4
  //   ()
  // }
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

  // test("int math expr 1")(assertOffload(42 + 69))

}
