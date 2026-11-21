package polyregion

class RangeQuerySuite extends RangeSuiteBase {

  same("exists-hit")((0 until 10).exists(_ > 8))
  same("exists-miss")((0 until 10).exists(_ > 9))
  same("exists-empty")((5 until 5).exists(_ => true))
  same("forall-hit")((0 until 10).forall(_ < 10))
  same("forall-miss")((0 until 10).forall(_ < 9))
  same("forall-empty")((5 until 5).forall(_ => false))
  same("count")((0 until 10).count(_ % 2 == 0))
  same("count-none")((0 until 10).count(_ > 100))
  same("count-empty")((5 until 5).count(_ => true))
  same("count-capture") {
    val m = 3
    (1 to 30).count(_ % m == 0)
  }

}
