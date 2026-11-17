package polyregion.javalang

import polyregion.ast.{PolyAST as p, *}

class BytecodeRemapperSuite extends Differential {

  private val KernelSource =
    """
      |public final class Kernels {
      |
      |  public static int add(int a, int b) { return a + b; }
      |
      |  public static int arith(int a, int b) {
      |    int c = a * b - a / (b + 1);
      |    c += a % 7;
      |    c = (c ^ (a & b)) | (a >> 2);
      |    c -= (a >>> 3) + (a << 1);
      |    return ~c;
      |  }
      |
      |  public static int absDiff(int a, int b) {
      |    int d = a - b;
      |    if (d < 0) { d = -d; }
      |    return d;
      |  }
      |
      |  public static int ternary(int a, int b) { return a > b ? a - b : b - a; }
      |
      |  public static int sumTo(int n) {
      |    int s = 0;
      |    for (int i = 0; i < n; i++) { s += i * i; }
      |    return s;
      |  }
      |
      |  public static int gcd(int a, int b) {
      |    while (b != 0) { int t = b; b = a % b; a = t; }
      |    return a < 0 ? -a : a;
      |  }
      |
      |  public static int collatz(int n) {
      |    int steps = 0;
      |    while (n != 1) {
      |      if ((n & 1) == 0) { n = n / 2; } else { n = 3 * n + 1; }
      |      steps++;
      |      if (steps > 1000) { break; }
      |    }
      |    return steps;
      |  }
      |
      |  public static int nested(int n, int m) {
      |    int t = 0;
      |    for (int i = 0; i < n; i++) {
      |      for (int j = 0; j < m; j++) {
      |        if (j == 3) { continue; }
      |        if (i * j > 20) { break; }
      |        t += i * j;
      |      }
      |      t++;
      |    }
      |    return t;
      |  }
      |
      |  public static int shortCircuit(int a, int b) {
      |    int r;
      |    if (a > 0 && b > 0) { r = a + b; } else { r = a - b; }
      |    return r * 2;
      |  }
      |
      |  public static int orChain(int a, int b, int c) {
      |    int r;
      |    if (a > 0 || b > 0 || c > 0) { r = 1; } else { r = -1; }
      |    return r + a;
      |  }
      |
      |  public static long longOps(long a, long b) {
      |    long c = a * b + (a >> 3) - (b >>> 5);
      |    if (a > b) { c = c ^ a; }
      |    return c / (b | 1L);
      |  }
      |
      |  public static int compareLongs(long a, long b) {
      |    if (a < b) { return -1; }
      |    if (a > b) { return 1; }
      |    return 0;
      |  }
      |
      |  public static double poly(double x, int n) {
      |    double acc = 0.0;
      |    double pow = 1.0;
      |    for (int i = 0; i <= n; i++) { acc += pow / (i + 1); pow *= x; }
      |    return acc;
      |  }
      |
      |  public static int narrowing(int x) {
      |    byte b = (byte) x;
      |    short s = (short) (x >> 3);
      |    char c = (char) (x * 7);
      |    return b + s + c;
      |  }
      |
      |  public static long widening(int x) { return ((long) x) * x + (long) (double) x; }
      |
      |  public static int dotProduct(int[] xs, int[] ys, int n) {
      |    int acc = 0;
      |    for (int i = 0; i < n; i++) { acc += xs[i] * ys[i]; }
      |    return acc;
      |  }
      |
      |  public static void scale(int[] xs, int n, int k) {
      |    for (int i = 0; i < n; i++) { xs[i] = xs[i] * k + i; }
      |  }
      |
      |  public static double norm(double[] xs, int n) {
      |    double acc = 0.0;
      |    for (int i = 0; i < n; i++) { double v = xs[i]; acc += v * v; }
      |    return acc;
      |  }
      |
      |  public static int labelled(int n, int m) {
      |    int t = 0;
      |    outer:
      |    for (int i = 0; i < n; i++) {
      |      for (int j = 0; j < m; j++) {
      |        if (i * j > 30) { continue outer; }
      |        if (t > 200) { break outer; }
      |        t += i * j;
      |      }
      |      t++;
      |    }
      |    return t;
      |  }
      |
      |  public static int stepping(int[] xs, int n) {
      |    int i = 0;
      |    int s = 0;
      |    while (i < n) { s += xs[i++] * 2; }
      |    int j = n - 1;
      |    while (j >= 0) { s -= xs[j--]; }
      |    if (n > 0) { xs[0]++; s += --xs[n - 1]; }
      |    return s;
      |  }
      |
      |  public static int ternaryChain(int a, int b, int c) {
      |    return (a > b ? a : b) + (b > c ? b - c : c - b) * (a < 0 ? -1 : 1);
      |  }
      |
      |  public static boolean anyNegative(int[] xs, int n) {
      |    for (int i = 0; i < n; i++) { if (xs[i] < 0) { return true; } }
      |    return false;
      |  }
      |}
      |""".stripMargin

  private lazy val kernels            = JavaInMemory.compile("Kernels", KernelSource)
  override protected lazy val subject = Differential.staticsOf(kernels, "Kernels")

  private val ints = List(0, 1, -1, 2, 7, -13, 1000, Int.MaxValue, Int.MinValue)
  private val pairs = for {
    a <- ints
    b <- ints if b != -1
  } yield List[Any](a, b)

  test("arithmetic and locals") {
    check("add", pairs)
    check("arith", pairs)
  }

  test("conditionals") {
    check("absDiff", pairs)
    check("ternary", pairs)
  }

  test("short circuit operators") {
    check("shortCircuit", pairs)
    check(
      "orChain",
      for {
        a <- ints
        b <- ints
        c <- ints
      } yield List[Any](a, b, c)
    )
  }

  test("counted loop") {
    check("sumTo", List(0, 1, 2, 10, 100, -5).map(n => List[Any](n)))
  }

  test("while loop with a data dependent trip count") {
    check("gcd", pairs)
  }

  test("loop with an early break") {
    check("collatz", (1 to 60).toList.map(n => List[Any](n)))
  }

  test("nested loops with break and continue") {
    check(
      "nested",
      (for {
        n <- 0 to 6
        m <- 0 to 8
      } yield List[Any](n, m)).toList
    )
  }

  test("long arithmetic and lcmp") {
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

  test("narrowing and widening conversions") {
    check("narrowing", ints.map(n => List[Any](n)))
    check("widening", ints.map(n => List[Any](n)))
  }

  test("array reads") {
    val xs = Array(1, -2, 3, -4, 5, 6)
    val ys = Array(9, 8, 7, 6, 5, 4)
    check("dotProduct", (0 to 6).toList.map(n => List[Any](xs, ys, n)))
    check("norm", (0 to 4).toList.map(n => List[Any](Array(1.5, -2.25, 0.0, 4.0), n)))
    check("anyNegative", (0 to 6).toList.map(n => List[Any](xs, n)))
  }

  test("array writes are visible to the caller") {
    val factors = List(0, 1, -3)
    check(
      "scale",
      (for {
        n <- 0 to 6
        k <- factors
      } yield List[Any](Array(1, 2, 3, 4, 5, 6), n, k)).toList
    )
  }

  test("a plain counted loop emits exactly one loop and no jump register") {
    val body = lower("sumTo").repr
    assertEquals(body.linesIterator.count(_.contains("while(")), 1, body)
    assert(!body.contains("#jmp"), body)
  }

  test("increments and conditionals in value position") {
    val thirds = List(0, 3, -9)
    check(
      "stepping",
      (0 to 6).toList.map(n => List[Any](Array(1, -2, 3, -4, 5, 6), n))
    )
    check(
      "ternaryChain",
      for {
        a <- ints
        b <- ints
        c <- thirds
      } yield List[Any](a, b, c)
    )
  }

  test("nested loops with a labelled jump") {
    check(
      "labelled",
      (for {
        n <- 0 to 6
        m <- 0 to 8
      } yield List[Any](n, m)).toList
    )
  }

  test("the entry signature matches the descriptor") {
    val fn = lower("dotProduct")
    assertEquals(fn.args.map(_.named.tpe), List(intPtr, intPtr, p.Type.IntS32))
    assertEquals(fn.rtn, p.Type.IntS32)
    assertEquals(lower("scale").rtn, p.Type.Unit0)
    assertEquals(lower("anyNegative").rtn, p.Type.Bool1)
  }

  private def intPtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
}
