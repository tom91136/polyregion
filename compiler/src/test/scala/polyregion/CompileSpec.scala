package polyregion

import polyregion.compileTime._

class CompileSpec extends munit.FunSuite {

  test("unit") {
    assertEquals(offload(()), ())
  }

  test("1") {
    assertEquals(offload(1), 1)
  }

  test("copy capture") {
    val a = 1
    assertEquals(offload(a), a)
  }

  test("statements") {
    assertEquals(offload { 1; 2 }, 2)
  }

  test("const cond") {
    assertEquals(offload(if (true) 42 else 69), 42)
    assertEquals(offload(if (false) 42 else 69), 69)
  }

  test("copy capture as cond test") {
    var a = true
    assertEquals(offload(if (a) 42 else 69), 42)
    a = false
    assertEquals(offload(if (a) 42 else 69), 69)
  }

  test("copy capture expr as cond test") {
    var a = 10
    assertEquals(offload(if (a == 10) 42 else 69), 42)
    a = 0
    assertEquals(offload(if (a == 10) 42 else 69), 69)
  }

  test("math expr 1") {
    assertEquals(offload(42+69), 111)
  }

}
