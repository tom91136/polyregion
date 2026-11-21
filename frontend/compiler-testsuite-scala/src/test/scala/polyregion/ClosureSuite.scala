package polyregion

class ClosureSuite extends BaseSuite {

  private inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.ClosureSuite) {
    test(name)(assertOffloadValue(offload1(r)))
  }

  testExpr("lambda-value") {
    val f = (x: Int) => x + 1
    f(41)
  }
  testExpr("lambda-value-capture") {
    val y = 2
    val f = (x: Int) => x + y
    f(40)
  }
  testExpr("lambda-capture-used-directly-and-in-nested-lambda") {
    val y = 2
    val z = y * 10
    val f = (x: Int) => {
      val g = (w: Int) => w + y
      g(x) + z
    }
    f(40) + y
  }
  testExpr("lambda-value-two-args") {
    val f = (x: Int, y: Int) => x * y
    f(6, 7)
  }
  testExpr("lambda-value-alias") {
    val f = (x: Int) => x + 1
    val g = f
    g(41)
  }
  testExpr("lambda-applied")(
    ((x: Int) => x + 1)(41)
  )
  testExpr("generic-method") {
    def id[A](a: A): A = a
    id(42)
  }
  testExpr("null-literal") {
    val xs: Array[Int] = null
    if (xs eq xs) 1 else 0
  }
  testExpr("string-literal") {
    val s = "abc"
    3
  }

  testExpr("range-foreach") {
    val xs = Array.ofDim[Int](10)
    (0 until 10).foreach(i => xs(i) = i * 2)
    xs(3) + xs(9)
  }
  testExpr("for-range") {
    val xs = Array.ofDim[Int](10)
    for (i <- 0 until 10) xs(i) = i + 1
    xs(0) + xs(9)
  }
  testExpr("for-range-inclusive") {
    val xs = Array.ofDim[Int](11)
    for (i <- 0 to 10) xs(i) = i * i
    xs(10) + xs(5)
  }
  testExpr("for-range-inclusive-step") {
    val xs = Array.ofDim[Int](11)
    for (i <- 0 to 10 by 2) xs(i) = i + 1
    xs(10) + xs(4)
  }
  testExpr("for-range-step") {
    val xs = Array.ofDim[Int](20)
    for (i <- 0 until 20 by 3) xs(i) = i + 1
    xs(0) + xs(3) + xs(18)
  }
  testExpr("for-range-descending") {
    val xs = Array.ofDim[Int](11)
    for (i <- 10 until 0 by -3) xs(i) = i + 1
    xs(10) + xs(7) + xs(1)
  }
  testExpr("for-range-nested") {
    val xs = Array.ofDim[Int](16)
    for {
      i <- 0 until 4
      j <- 0 until 4
    } xs(i * 4 + j) = i * 10 + j
    xs(0) + xs(5) + xs(15)
  }
  testExpr("hof-local") {
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    twice(_ + 1, 40)
  }
  testExpr("hof-local-capture") {
    val k                                 = 3
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    twice(x => x * k, 5)
  }
  testExpr("hof-two-lambdas") {
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    twice(_ + 1, 40) + twice(_ * 2, 3)
  }
  testExpr("hof-two-levels") {
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    def quad(g: Int => Int, x: Int): Int  = twice(g, twice(g, x))
    quad(_ + 1, 0)
  }
  testExpr("hof-inline") {
    inline def twice(inline f: Int => Int, x: Int): Int = f(f(x))
    twice(_ + 1, 40)
  }
  testExpr("hof-callee-captures") {
    val k                                = 5
    def addK(f: Int => Int, x: Int): Int = f(x) + k
    addK(_ * 2, 3) + addK(_ + 1, 4)
  }
  testExpr("hof-array-write") {
    val xs = Array.ofDim[Int](4)
    def fill(f: Int => Int): Unit = {
      var i = 0
      while (i < 4) {
        xs(i) = f(i)
        i += 1
      }
    }
    fill(_ * 3)
    xs(3)
  }
  testExpr("hof-array-write-capture-bound") {
    val xs = Array.ofDim[Int](4)
    val s  = 0
    def fill(f: Int => Int): Unit = {
      var i = s
      while (i < 4) {
        xs(i) = f(i)
        i += 1
      }
    }
    fill(_ * 3)
    xs(3)
  }
  testExpr("ref-eq-null") {
    val xs = Array.ofDim[Int](4)
    if (xs eq null) 0 else 1
  }
  testExpr("ref-ne-null") {
    val xs: Array[Int] = null
    if (xs ne null) 0 else 1
  }
  testExpr("ref-eq-alias") {
    val xs = Array.ofDim[Int](4)
    val ys = xs
    val zs = Array.ofDim[Int](4)
    (if (xs eq ys) 1 else 0) + (if (xs ne zs) 10 else 0)
  }
  testExpr("tuple2") {
    val t = (1, 2)
    t._1 + t._2
  }
  testExpr("option-get") {
    val o = Some(3)
    o.get
  }
  testExpr("range-until-max-int") {
    val n = Array.ofDim[Int](2)
    n(0) = 0
    for (i <- Int.MaxValue - 5 until Int.MaxValue by 2) {
      n(0) += 1
      n(1) = i
    }
    n(0) + n(1)
  }
  testExpr("range-to-max-int") {
    val n = Array.ofDim[Int](2)
    n(0) = 0
    for (i <- Int.MaxValue - 2 to Int.MaxValue) {
      n(0) += 1
      n(1) = i
    }
    n(0) + n(1)
  }
  testExpr("range-to-min-int-descending") {
    val n = Array.ofDim[Int](2)
    n(0) = 0
    for (i <- Int.MinValue + 2 to Int.MinValue by -1) {
      n(0) += 1
      n(1) = i
    }
    n(0) + n(1)
  }
  testExpr("range-until-min-int-descending-step") {
    val n = Array.ofDim[Int](2)
    n(0) = 0
    for (i <- Int.MinValue + 4 until Int.MinValue by -3) {
      n(0) += 1
      n(1) = i
    }
    n(0) + n(1)
  }
  testExpr("range-empty") {
    val n = Array.ofDim[Int](1)
    n(0) = 0
    (5 until 5).foreach(_ => n(0) += 1)
    (5 to 4).foreach(_ => n(0) += 1)
    (0 until 5 by -1).foreach(_ => n(0) += 1)
    n(0)
  }
  test("range-step-zero-runs-no-iterations") {
    val n = offload1(using AssertTarget.Offload) {
      val n = Array.ofDim[Int](1)
      n(0) = 0
      (0 until 4 by 0).foreach(_ => n(0) += 1)
      n(0)
    }
    assertEquals(n, 0)
  }
  testExpr("generic-body-apply") {
    def dup[T](x: T): (T, T) = (x, x)
    val t                    = dup(3)
    t._1 + t._2
  }
  testExpr("generic-body-ref") {
    def id[A](a: A): A    = a
    def again[B](b: B): B = id(id(b))
    again(5)
  }
  test("ref-eq-unrelated-pointers") {
    val errors = compileErrors(
      "offload1(using AssertTarget.Offload) { val a = Array.ofDim[Int](1); val b = Array.ofDim[Long](1); if (a eq b) 1 else 0 }"
    )
    assert(errors.contains("unrelated pointer types"), errors)
  }
}
