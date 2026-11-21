package polyregion

class RangeStepSuite extends RangeSuiteBase {

  raisesSame("step-zero")((0 until 4 by 0).length)
  raisesSame("step-zero-until")(0.until(4, 0).length)
  raisesSame("step-zero-to")(0.to(4, 0).length)
  raisesSame("step-zero-foreach") {
    var n = 0
    (0 until 4 by 0).foreach(_ => n += 1)
    n
  }
  raisesSame("step-zero-for-comprehension") {
    var n = 0
    for (i <- 0 to 4 by 0) n += i
    n
  }
}
