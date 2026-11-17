package polyregion.javalang

import polyregion.ast.{PolyAST as p, *}

class BytecodeScalaSuite extends Differential {

  private val KernelSource =
    """
      |object ScalaKernels {
      |
      |  def sumTo(n: Int): Int = {
      |    var s = 0
      |    var i = 0
      |    while (i < n) { s += i * i; i += 1 }
      |    s
      |  }
      |
      |  @annotation.tailrec
      |  def gcd(a: Int, b: Int): Int = if (b == 0) (if (a < 0) -a else a) else gcd(b, a % b)
      |
      |  @annotation.tailrec
      |  def fact(n: Int, acc: Long): Long = if (n <= 1) acc else fact(n - 1, acc * n)
      |
      |  def collatz(n0: Int): Int = {
      |    var n     = n0
      |    var steps = 0
      |    while (n != 1 && steps <= 1000) {
      |      n = if ((n & 1) == 0) n / 2 else 3 * n + 1
      |      steps += 1
      |    }
      |    steps
      |  }
      |
      |  def nested(n: Int, m: Int): Int = {
      |    var t = 0
      |    var i = 0
      |    while (i < n) {
      |      var j = 0
      |      while (j < m) {
      |        if (j != 3 && i * j <= 20) t += i * j
      |        j += 1
      |      }
      |      t += 1
      |      i += 1
      |    }
      |    t
      |  }
      |
      |  def branchy(a: Int, b: Int): Int = {
      |    var r = 0
      |    if (a > 0 && b > 0) r = a + b
      |    else if (a < 0 || b < 0) r = a - b
      |    else r = a * b
      |    r * 2
      |  }
      |
      |  def longOps(a: Long, b: Long): Long = {
      |    var c = a * b + (a >> 3) - (b >>> 5)
      |    if (a > b) c = c ^ a
      |    c / (b | 1L)
      |  }
      |
      |  def compareLongs(a: Long, b: Long): Int = if (a < b) -1 else if (a > b) 1 else 0
      |
      |  def narrowing(x: Int): Int = {
      |    val b = x.toByte
      |    val s = (x >> 3).toShort
      |    val c = (x * 7).toChar
      |    b + s + c
      |  }
      |
      |  def dot(xs: Array[Int], ys: Array[Int], n: Int): Int = {
      |    var acc = 0
      |    var i   = 0
      |    while (i < n) { acc += xs(i) * ys(i); i += 1 }
      |    acc
      |  }
      |
      |  def scale(xs: Array[Int], n: Int, k: Int): Unit = {
      |    var i = 0
      |    while (i < n) { xs(i) = xs(i) * k + i; i += 1 }
      |  }
      |
      |  def norm(xs: Array[Double], n: Int): Double = {
      |    var acc = 0.0
      |    var i   = 0
      |    while (i < n) { val v = xs(i); acc += v * v; i += 1 }
      |    acc
      |  }
      |
      |  def anyNegative(xs: Array[Int], n: Int): Boolean = {
      |    var i = 0
      |    while (i < n) { if (xs(i) < 0) return true; i += 1 }
      |    false
      |  }
      |
      |  def poly(x: Double, n: Int): Double = {
      |    var acc = 0.0
      |    var pow = 1.0
      |    var i   = 0
      |    while (i <= n) { acc += pow / (i + 1); pow *= x; i += 1 }
      |    acc
      |  }
      |}
      |""".stripMargin

  private lazy val kernels            = ScalaInMemory.compile("ScalaKernels", KernelSource)
  override protected lazy val subject = Differential.moduleOf(kernels, "ScalaKernels")

  private val ints = List(0, 1, -1, 2, 7, -13, 1000, Int.MaxValue, Int.MinValue)
  private val pairs = for {
    a <- ints
    b <- ints if b != -1
  } yield List[Any](a, b)

  test("a method on an object lowers even though it is not static") {
    val fn = lower("sumTo")
    assertEquals(fn.args.map(_.named.tpe), List(p.Type.IntS32))
    assertEquals(fn.rtn, p.Type.IntS32)
    check("sumTo", List(0, 1, 2, 10, 100, -5).map(n => List[Any](n)))
  }

  test("tail recursion arrives as a loop") {
    val accumulators = List[Long](1L, 3L)
    check("gcd", pairs)
    check(
      "fact",
      for {
        n <- List(0, 1, 2, 5, 12, 20, -3)
        a <- accumulators
      } yield List[Any](n, a)
    )
    assert(lower("fact").repr.contains("while("), lower("fact").repr)
  }

  test("while loops and nesting") {
    check("collatz", (1 to 60).toList.map(n => List[Any](n)))
    check(
      "nested",
      (for {
        n <- 0 to 6
        m <- 0 to 8
      } yield List[Any](n, m)).toList
    )
  }

  test("short circuit operators") {
    check("branchy", pairs)
  }

  test("long arithmetic") {
    val longs = List[Long](0L, 1L, -1L, 5L, -77L, 1L << 40, Long.MaxValue, Long.MinValue)
    check(
      "longOps",
      for {
        a <- longs
        b <- longs
      } yield List[Any](a, b)
    )
    check(
      "compareLongs",
      for {
        a <- longs
        b <- longs
      } yield List[Any](a, b)
    )
  }

  test("double arithmetic") {
    val exponents = List(0, 1, 5, 12)
    check(
      "poly",
      for {
        x <- List(0.0, 0.5, -1.25, 2.0)
        n <- exponents
      } yield List[Any](x, n)
    )
  }

  test("narrowing conversions") {
    check("narrowing", ints.map(n => List[Any](n)))
  }

  test("arrays") {
    val xs = Array(1, -2, 3, -4, 5, 6)
    val ys = Array(9, 8, 7, 6, 5, 4)
    check("dot", (0 to 6).toList.map(n => List[Any](xs, ys, n)))
    check("norm", (0 to 4).toList.map(n => List[Any](Array(1.5, -2.25, 0.0, 4.0), n)))
    check("anyNegative", (0 to 6).toList.map(n => List[Any](xs, n)))
    val factors = List(0, 1, -3)
    check(
      "scale",
      (for {
        n <- 0 to 6
        k <- factors
      } yield List[Any](Array(1, 2, 3, 4, 5, 6), n, k)).toList
    )
  }
}
