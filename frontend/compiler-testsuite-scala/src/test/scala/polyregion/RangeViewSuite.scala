package polyregion

import polyregion.scalalang.*

class RangeViewSuite extends RangeSuiteBase {

  sameOnHost("map-fold")((1 to 6).map(_ + 1).foldLeft(0)(_ + _))
  sameOnHost("map-length")((0 until 9).map(_ * 2).length)
  sameOnHost("map-apply")((0 until 9).map(_ * 2)(4))
  sameOnHost("filter-length")((0 to 9).filter(_ % 3 == 0).length)
  sameOnHost("filter-none")((0 until 10).filter(_ > 100).length)
  sameOnHost("map-capture") {
    val k = 5
    (0 until 4).map(_ + k).foldLeft(0)(_ + _)
  }
  sameOnHost("map-sum")((0 until 8).map(i => i * i).sum)
  sameOnHost("filter-sum")((0 until 20).filter(_ % 2 == 1).sum)
  sameOnHost("filter-map-sum")((0 until 10).filter(_ % 2 == 1).map(_ * 3).sum)

  test("a mapped range stored into a capture is refused") {
    val errors = compileErrors("""
      val holder = new Array[scala.collection.immutable.IndexedSeq[Int]](1)
      compiletime.offload0[Config[Target.Host.type, Opt.O2]](blocking.Host.context, _ => ())({ holder(0) = (0 until 4).map(_ + 1); () })
    """)
    assert(errors.contains("may not let an allocation escape"), errors)
  }

  test("a filtered range stored into a capture is refused") {
    val errors = compileErrors("""
      val holder = new Array[scala.collection.immutable.IndexedSeq[Int]](1)
      compiletime.offload0[Config[Target.Host.type, Opt.O2]](blocking.Host.context, _ => ())({ holder(0) = (0 until 4).filter(_ > 1); () })
    """)
    assert(errors.contains("may not let an allocation escape"), errors)
  }
}
