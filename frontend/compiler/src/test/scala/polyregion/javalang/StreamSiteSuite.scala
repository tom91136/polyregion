package polyregion.javalang

class StreamSiteSuite extends munit.FunSuite {

  private def locationOf(cls: Class[?]): String =
    java.nio.file.Path.of(cls.getProtectionDomain.getCodeSource.getLocation.toURI).toString

  private val classpath =
    List(locationOf(classOf[Streams]), locationOf(classOf[polyregion.jvm.runtime.Context]))
      .mkString(java.io.File.pathSeparator)

  private val Options: List[String] =
    List("-g", "-classpath", classpath, "-Apolyregion.library.path=/nonexistent/polyregion") ++
      Option.when(java.lang.Runtime.version.feature >= 21)("-proc:full")

  private def diagnostics(body: String): String =
    try {
      JavaInMemory.compile(
        "diag.Chains",
        s"""package diag;
           |
           |import java.util.Arrays;
           |import java.util.stream.IntStream;
           |import polyregion.javalang.Par;
           |import polyregion.javalang.ParInt;
           |
           |public class Chains {
           |$body
           |}
           |""".stripMargin,
        Options,
        List(Processor())
      )
      ""
    } catch { case e: RuntimeException => e.getMessage }

  test("a jdk stream chain compiles without a diagnostic when the launch library is missing") {
    val errors = diagnostics("""  public static int run(int[] xs) {
                               |    return Arrays.stream(xs).parallel().map(x -> x + 1).sum();
                               |  }""".stripMargin)
    assertEquals(errors, "")
  }

  test("a .par chain with a missing launch library is a compile error naming the operation") {
    val errors = diagnostics("""  public static int run(int[] xs) {
                               |    return Par.ints(xs).map(x -> x + 1).sum();
                               |  }""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot lower"), errors)
    assert(errors.contains("sum()"), errors)
  }

  test("a stored .par chain is a compile error") {
    val errors = diagnostics("""  public static int run(int[] xs) {
                               |    ParInt chain = Par.ints(xs).map(x -> x + 1);
                               |    return chain.sum();
                               |  }""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot lower"), errors)
    assert(errors.contains("a stream chain must be a single expression"), errors)
  }

  test("a .par stage that is not a lambda expression is a compile error") {
    val errors = diagnostics("""  public static int run(int[] xs) {
                               |    java.util.function.IntUnaryOperator f = x -> x + 1;
                               |    return Par.ints(xs).map(f).sum();
                               |  }""".stripMargin)
    assert(errors.contains("a stream operation must take a lambda expression or a static method reference"), errors)
  }

  test("a .par lambda that binds a receiver is a compile error") {
    val errors = diagnostics("""  private int bias = 1;
                               |
                               |  private int add(int x) { return x + bias; }
                               |
                               |  public int run(int[] xs) {
                               |    return Par.ints(xs).map(this::add).sum();
                               |  }""".stripMargin)
    assert(errors.contains("a stream lambda cannot bind a receiver"), errors)
  }

  test("a second context selection is a compile error") {
    val errors = diagnostics(
      """  public static int run(int[] xs, polyregion.jvm.runtime.Context a, polyregion.jvm.runtime.Context b) {
                               |    return Par.ints(xs).on(a).map(x -> x + 1).on(b).sum();
                               |  }""".stripMargin
    )
    assert(errors.contains("a .par chain selects its context once"), errors)
  }

  test("an array call with a missing launch library is a compile error") {
    val errors = diagnostics("""  public static void run(int[] xs) {
                               |    Par.sort(xs);
                               |    Par.prefix(xs, java.lang.Math::max);
                               |  }""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot lower"), errors)
  }
}
