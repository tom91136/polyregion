package polyregion

class RangeLengthSuite extends RangeSuiteBase {

  same("length")((0 until 10).length)
  same("size-inclusive")((0 to 10).size)
  same("length-step")((0 until 10 by 3).length)
  same("length-descending")((10 until 0 by -3).length)
  same("length-descending-inclusive")((10 to 0 by -5).length)
  same("length-empty")((5 until 5).length)
  same("length-empty-inclusive")((5 to 4).length)
  same("length-empty-wrong-direction")((0 until 5 by -1).length)
  same("length-until-max-int")((Int.MaxValue - 5 until Int.MaxValue by 2).length)
  same("length-to-max-int")((Int.MaxValue - 2 to Int.MaxValue).length)
  same("length-to-min-int-descending")((Int.MinValue + 2 to Int.MinValue by -1).length)
  same("length-full-span-step")((Int.MinValue to Int.MaxValue by 3).length)
  raisesClass("length-too-long")((Int.MinValue to Int.MaxValue).length)

}
