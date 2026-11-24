package polyregion.javalang

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

class LaunchSiteSuite extends munit.FunSuite {

  private val runtimeClasses =
    java.nio.file.Path.of(classOf[Interface].getProtectionDomain.getCodeSource.getLocation.toURI)

  private def options(classpath: String): List[String] =
    List("-g", "-classpath", classpath, "-Apolyregion.library.path=/nonexistent/polyregion") ++
      Option.when(java.lang.Runtime.version.feature >= 21)("-proc:full")

  private val Options: List[String] = options(runtimeClasses.toString)

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

  test("region sugar inside an offload region is a compile error naming the call") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.Offload;
                                |import polyregion.javalang.Runtime;
                                |
                                |public class Sites {
                                |  public static void run() {
                                |    int[] xs = new int[4];
                                |    Runtime.offload(() -> Offload.for_each(2L, 2L, (i, j) -> xs[(int) (i * 2 + j)] = 0));
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot offload"), errors)
    assert(
      errors.contains("`polyregion.Offload.for_each` launches `polyregion.core:polyregion.core.for_each`"),
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

  private val functional =
    """    public interface Sum { int apply(int[] xs, int n); }
      |""".stripMargin

  test("a method reference naming an interface stub is a compile error naming the stub") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.javalang.Interface;
                                |
                                |public class Sites {
                                |$api
                                |$functional
                                |  public static int run(int[] xs) {
                                |    Sum f = Api::sum;
                                |    return f.apply(xs, 3);
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot launch diag.Sites.Api.sum"), errors)
    assert(errors.contains("cannot be referenced as a method"), errors)
  }

  test("a method reference naming an interface stub inside an offload region is a compile error") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.javalang.Interface;
                                |import polyregion.javalang.Runtime;
                                |
                                |public class Sites {
                                |$api
                                |$functional
                                |  public static int run() {
                                |    int[] xs = {1, 2, 3};
                                |    return Runtime.offload(() -> {
                                |      Sum f = Api::sum;
                                |      return f.apply(xs, 3);
                                |    });
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot offload"), errors)
    assert(
      errors.contains("`diag.Sites.Api.sum` launches `demo:demo.sum` and cannot be called inside an offload region"),
      errors
    )
  }

  test("a launch callable that binds a receiver is a compile error naming the call") {
    val errors = diagnostics(s"""package diag;
                                |
                                |import polyregion.javalang.Interface;
                                |
                                |public class Sites {
                                |$api
                                |  int add(int x) { return x + 1; }
                                |
                                |  public void run(int[] xs) {
                                |    Api.map(xs, this::add);
                                |  }
                                |}
                                |""".stripMargin)
    assert(errors.contains("error: [PolyRegion] Cannot launch diag.Sites.Api.map"), errors)
    assert(errors.contains("cannot bind a receiver"), errors)
  }

  test("a failure to find launch sites is a compile error") {
    val failing = Processor(_ => throw RuntimeException("scan failed"))
    val errors =
      try {
        JavaInMemory.compile(
          "diag.Sites",
          """package diag;
            |
            |public class Sites {
            |  public static int run() { return 1; }
            |}
            |""".stripMargin,
          Options,
          List(failing)
        )
        ""
      } catch { case e: RuntimeException => e.getMessage }
    assert(errors.contains("error: [PolyRegion] Cannot find launch sites in diag.Sites"), errors)
    assert(errors.contains("scan failed"), errors)
  }

  test("an interface call reached through a class-file helper inside an offload region is a compile error") {
    val dir = Files.createTempDirectory("launch-sites")
    try {
      JavaInMemory
        .compile(
          "lib.Lib",
          """package lib;
            |
            |import polyregion.javalang.Interface;
            |
            |public class Lib {
            |  public static class Api {
            |    @Interface("demo:demo.sum")
            |    public static int sum(int[] xs, int n) { throw new UnsupportedOperationException(); }
            |  }
            |
            |  public static class Helper {
            |    public static int total(int[] xs) { return Api.sum(xs, xs.length); }
            |  }
            |}
            |""".stripMargin,
          Options
        )
        .classes
        .foreach { (name, bytes) =>
          val file = dir.resolve(name.replace('.', '/') + ".class")
          Files.createDirectories(file.getParent)
          Files.write(file, bytes)
        }
      val withLibrary = options(s"$runtimeClasses${java.io.File.pathSeparator}$dir")
      val errors =
        try {
          JavaInMemory.compile(
            "diag.Sites",
            """package diag;
              |
              |import polyregion.javalang.Runtime;
              |
              |public class Sites {
              |  public static int run() {
              |    int[] xs = {1, 2, 3};
              |    return Runtime.offload(() -> lib.Lib.Helper.total(xs));
              |  }
              |}
              |""".stripMargin,
            withLibrary,
            List(Processor())
          )
          ""
        } catch { case e: RuntimeException => e.getMessage }
      assert(
        errors.contains("`lib.Lib.Api.sum` launches `demo:demo.sum` and cannot be called inside an offload region"),
        errors
      )
    } finally Files.walk(dir).sorted(java.util.Comparator.reverseOrder[Path]).toList.asScala.foreach(Files.delete(_))
  }
}
