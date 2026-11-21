package polyregion

class RangeSumSuite extends RangeSuiteBase {

  same("sum")((0 until 10).sum)
  same("sum-inclusive")((1 to 100).sum)
  same("sum-step")((1 to 10 by 3).sum)
  same("sum-descending")((10 until 0 by -3).sum)
  same("sum-empty")((5 until 5).sum)
  same("sum-single")((7 to 7).sum)
  same("sum-wraps")((0 to 100000).sum)
}
