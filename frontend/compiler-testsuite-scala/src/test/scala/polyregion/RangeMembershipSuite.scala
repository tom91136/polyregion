package polyregion

class RangeMembershipSuite extends RangeSuiteBase {

  same("is-empty-true")((5 until 5).isEmpty)
  same("is-empty-false")((0 until 5).isEmpty)
  same("is-empty-inclusive")((5 to 5).isEmpty)
  same("is-empty-inclusive-true")((5 to 4).isEmpty)
  same("is-empty-wrong-direction")((0 until 5 by -1).isEmpty)
  same("non-empty")((0 until 5).nonEmpty)

  same("contains-hit")((0 until 10).contains(7))
  same("contains-end-exclusive")((0 until 10).contains(10))
  same("contains-end-inclusive")((0 to 10).contains(10))
  same("contains-before-start")((3 until 10).contains(2))
  same("contains-step-hit")((1 until 20 by 4).contains(13))
  same("contains-step-miss")((1 until 20 by 4).contains(14))
  same("contains-descending-hit")((20 until 0 by -6).contains(8))
  same("contains-descending-miss")((20 until 0 by -6).contains(9))
  same("contains-descending-end")((20 to 2 by -6).contains(2))
  same("contains-empty")((5 until 5).contains(5))
  same("contains-wide-span")((Int.MinValue to Int.MaxValue by 7).contains(Int.MaxValue - 6))
  same("contains-wide-span-miss")((Int.MinValue to Int.MaxValue by 7).contains(Int.MaxValue))
  same("contains-max-int")((Int.MaxValue - 4 to Int.MaxValue).contains(Int.MaxValue))

}
