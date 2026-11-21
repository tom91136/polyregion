package polyregion.javalang

class LaunchSiteSuite extends munit.FunSuite {

  private val runtimeClasses =
    java.nio.file.Path.of(classOf[Interface].getProtectionDomain.getCodeSource.getLocation.toURI)

  private val Options: List[String] =
    List("-g", "-classpath", runtimeClasses.toString) ++
      Option.when(java.lang.Runtime.version.feature >= 21)("-proc:full")

  private def diagnostics(code: String): String =
    try {
      JavaInMemory.compile("diag.Sites", code, Options, List(Processor()))
      ""
    } catch { case e: RuntimeException => e.getMessage }

  private val api =
    """    public static class Api {
      |      @Interface("demo:demo.sum")
      |      public static int sum(int[] xs, int n) { throw new UnsupportedOperationException(); }
      |
      |      @Interface("demo:demo.map")
      |      public static void map(int[] xs, java.util.function.IntUnaryOperator f) { throw new UnsupportedOperationException(); }
      |    }
      |""".stripMargin

  test("an interface call inside an offload region is a compile error naming the call") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.javalang.Interface;
                                |import polyregion.javalang.Runtime;
                                |
                                |public class Sites {
                                |$api
                                |  public static int run() {
                                |    int[] xs = {1, 2, 3};
                                |    return Runtime.offload(() -> Api.sum(xs, 3));
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot offload"), errors)
    assert(
      errors.contains("`diag.Sites.Api.sum` launches `demo:demo.sum` and cannot be called inside an offload region"),
      errors
    )
  }

  test("a launch callable that is not a lambda expression is a compile error naming the call") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.javalang.Interface;
                                |
                                |public class Sites {
                                |$api
                                |  public static void run(int[] xs) {
                                |    java.util.function.IntUnaryOperator f = x -> x + 1;
                                |    Api.map(xs, f);
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot launch diag.Sites.Api.map"), errors)
    assert(errors.contains("a launch callable must be a lambda expression"), errors)
  }
}
