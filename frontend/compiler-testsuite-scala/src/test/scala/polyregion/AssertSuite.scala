package polyregion

import polyregion.scalalang.*

class AssertSuite extends RangeSuiteBase {

  test("a message that is not a literal is refused") {
    val errors = compileErrors(
      """compiletime.offload0(blocking.Host.context, _ => ())({ val x = 7; Predef.assert(x < 5, "x=" + x) })"""
    )
    assert(errors.contains("must be a literal"), errors)
  }

  same("assert-holds") {
    val x = 3
    Predef.assert(x < 5)
    x + 1
  }
  same("assert-message-holds") {
    val x = 3
    Predef.assert(x < 5, "x is small")
    x + 1
  }
  same("require-holds") {
    val x = 3
    Predef.require(x < 5)
    x + 1
  }
  same("require-message-holds") {
    val x = 3
    Predef.require(x < 5, "x is small")
    x + 1
  }
  raisesSame("assert-fails") {
    val x = 7
    Predef.assert(x < 5)
  }
  raisesSame("assert-message-fails") {
    val x = 7
    Predef.assert(x < 5, "x is small")
  }
  raisesSame("require-fails") {
    val x = 7
    Predef.require(x < 5)
  }
  raisesSame("require-message-fails") {
    val x = 7
    Predef.require(x < 5, "x is small")
  }
  raisesSame("assert-in-loop") {
    var sum = 0
    var i   = 0
    while (i < 10) {
      Predef.assert(i < 6, "index too large")
      sum += i
      i += 1
    }
  }
  raisesSame("require-in-helper") {
    def half(n: Int): Int = {
      Predef.require(n % 2 == 0, "n must be even")
      n / 2
    }
    half(4) + half(7)
  }
}
