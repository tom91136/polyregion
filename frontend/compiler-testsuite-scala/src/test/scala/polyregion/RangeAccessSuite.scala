package polyregion

class RangeAccessSuite extends RangeSuiteBase {

  same("apply-first")((3 until 20 by 4)(0))
  same("apply-middle")((3 until 20 by 4)(2))
  same("apply-last")((0 until 5)(4))
  same("apply-descending")((10 to 0 by -2)(5))
  same("apply-near-max-int")((Int.MaxValue - 4 to Int.MaxValue)(4))
  raisesClass("apply-past-end")((0 until 5)(5))
  raisesClass("apply-negative")((0 until 5)(-1))
  raisesClass("apply-empty")((5 until 5)(0))

  same("head")((3 until 20 by 4).head)
  same("last")((3 until 20 by 4).last)
  same("last-exact")((3 to 19 by 4).last)
  same("last-descending")((20 until 0 by -6).last)
  same("last-max-int")((Int.MaxValue - 4 to Int.MaxValue by 3).last)
  raisesSame("head-empty")((5 until 5).head)
  raisesSame("last-empty")((5 until 5).last)

}
