package polyregion

import polyregion.compileTime._
import scala.compiletime._

class ControlFlowSuite extends munit.FunSuite {

  final val Enable = true

  inline def assertOffload[A](inline f: => A) = if (Enable) {
    assertEquals(offload[A](f), f)
  }

  test("stmts")(assertOffload { 1; 2 })

  test("const-if-true") {
    assertOffload((if (true) 42 else 69))
  }

  test("ref-if-true") {
    val x = true
    assertOffload((if (x) 42 else 69))
  }

  test("while-le-inc")(assertOffload {
    val lim = 10
    var i   = 0
    while (i < lim)
      i += 1
    i
  })

  test("ref-if-if-true") {
    val x = true
    val y = true
    assertOffload((if (x) (if (y) 1 else 2) else 3))
  }

  test("const-if-if-true") {
    assertOffload((if (true) (if (true) 1 else 2) else 3))
  }

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
