package polyregion

class RangeFoldSuite extends RangeSuiteBase {

  same("fold-left")((1 to 10).foldLeft(0)(_ + _))
  same("fold-left-order")((1 to 5).foldLeft(0)((a, i) => a * 10 + i))
  same("fold-left-long")((1 to 20).foldLeft(1L)((a, i) => a * i))
  same("fold-left-double")((1 until 6).foldLeft(0.5)((a, i) => a * 2.0 + i))
  same("fold-left-descending")((9 until 0 by -4).foldLeft(0)((a, i) => a * 10 + i))
  same("fold-left-empty")((5 until 5).foldLeft(42)(_ + _))
  same("fold-left-capture") {
    val k = 3
    (0 until 6).foldLeft(1)((a, i) => a + i * k)
  }
  same("reduce")((1 to 6).reduce(_ * _))
  same("reduce-order")((3 until 9 by 2).reduce((a, b) => a * 2 - b))
  same("reduce-left")((1 to 6).reduceLeft((a, b) => a * 3 - b))
  same("reduce-single")((4 to 4).reduce(_ + _))
  raisesSame("reduce-empty")((5 until 5).reduce(_ + _))
  same("inclusive-fold")((0 to 10 by 5).foldLeft(0)(_ + _))
  same("inclusive-contains")((0 to 10 by 5).contains(10))
  same("inclusive-length")((0 to 10 by 5).length)
}
