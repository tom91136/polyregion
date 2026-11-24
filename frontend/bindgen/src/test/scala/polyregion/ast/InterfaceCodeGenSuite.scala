package polyregion.ast

import java.io.{ByteArrayOutputStream, PrintStream}
import java.net.URLClassLoader
import java.nio.file.{Files, Path}
import javax.tools.ToolProvider
import scala.io.Source
import scala.util.Using

import polyregion.ast.PolyAST as p

class InterfaceCodeGenSuite extends munit.FunSuite {

  private def golden(name: String): String =
    Using.resource(Source.fromResource(s"bindgen/$name"))(_.mkString)

  private val t = p.Type.Var("T")
  private val u = p.Type.Var("U")
  private val transform = p.FunctionDecl(
    p.Sym("example.transform"),
    List(p.Type.Var("T"), p.Type.Var("U")),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(
        p.Named("out", p.Type.Ptr(u, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32)),
      p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), u)))
    ),
    Nil,
    Nil,
    p.Type.Unit0,
    p.Function.Affinity.Host
  )
  private val count = p.FunctionDecl(
    p.Sym("example.count"),
    List(p.Type.Var("T")),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32))
    ),
    Nil,
    Nil,
    p.Type.IntS32,
    p.Function.Affinity.Host
  )
  private val countValue   = count.copy(args = count.args :+ p.Arg(p.Named("value", t)))
  private val interfaceDef = p.Interface(p.Sym("example"), List(transform, count))
  private val fortran      = FortranConfig("example_ffi")
  private val scala        = ScalaConfig("example.bindings", "ExampleInterface")
  private val jvm          = JavaConfig("example.bindings", "ExampleInterface")
  private val reduceInit = p.FunctionDecl(
    p.Sym("example.reduce_init"),
    List(p.Type.Var("T"), p.Type.Var("U")),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32)),
      p.Arg(p.Named("init", u)),
      p.Arg(p.Named("map", p.Type.Exec(Nil, List(t), u))),
      p.Arg(p.Named("combine", p.Type.Exec(Nil, List(u, u), u)))
    ),
    Nil,
    Nil,
    u,
    p.Function.Affinity.Host
  )

  test("interface declarations project exact identities into C++, Fortran and Scala") {
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef)
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef, fortran)
    val sc  = InterfaceCodeGen.scalaObject(interfaceDef, scala)

    assertEquals(cpp, golden("example.hpp"))
    assertEquals(f90, golden("example.f90"))
    assertEquals(sc, golden("ExampleInterface.scala"))

    assert(cpp.contains("template <class T, class U, class Op> //"))
    assert(
      cpp.contains(
        "POLYREGION_EXAMPLE_ANNOTATE(\"polyregion_interface:example:example.transform\") POLYREGION_EXAMPLE_FALLBACK //"
      )
    )
    assert(cpp.contains("inline void transform(const T *in, U *out, std::int32_t n, Op op)"))
    assert(cpp.contains("POLYREGION_EXAMPLE_IMPLEMENT(count, in, n);"))
    assert(cpp.contains("POLYREGION_EXAMPLE_IMPLEMENT(transform, in, out, n, op);"))
    assert(cpp.contains("}\n\ntemplate <"))
    assert(!cpp.contains("}\ntemplate <"))
    assert(cpp.contains("#pragma push_macro(\"POLYREGION_EXAMPLE_IMPLEMENT\")"))
    assert(cpp.contains("#pragma pop_macro(\"POLYREGION_EXAMPLE_IMPLEMENT\")"))
    assert(!cpp.contains("IMPLEMENT_WAS_UNDEFINED"))

    assert(f90.contains("module example_ffi"))
    assert(f90.contains("polyregion_interface:example:example.transform"))
    assert(f90.contains("type(*), dimension(*), intent(in) :: in"))
    assert(f90.contains("type(*), dimension(*), intent(inout) :: out"))
    assert(f90.contains("procedure() :: op"))
    assert(!f90.contains("polyregion_transform_r4"))

    assert(sc.contains("package example.bindings"))
    assert(sc.contains("object ExampleInterface"))
    assert(!sc.contains("trait ExampleInterface"))
    assert(sc.contains("@compileTimeOnly(\"polyregion_interface:example:example.transform\")"))
    assert(sc.contains("inline def transform[T, U](in: Array[T], out: Array[U], n: Int, inline op: T => U): Unit"))
  }

  test("Fortran adapts erased generic results to output arguments") {
    val reduce = p.FunctionDecl(
      p.Sym("example.reduce"),
      List(p.Type.Var("T")),
      None,
      List(
        transform.args.head.copy(
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("init", t)),
        p.Arg(p.Named("op", p.Type.Exec(Nil, List(t, t), t)))
      ),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(reduce)), fortran)

    assert(f90.contains("subroutine polyregion_reduce(in, n, init, op, polyregion_result)"))
    assert(f90.contains("type(*), intent(in) :: init"))
    assert(f90.contains("procedure() :: op"))
    assert(f90.contains("type(*), intent(inout) :: polyregion_result"))
    assert(!f90.contains("function polyregion_reduce"))

    val collision = reduce.copy(args = reduce.args :+ p.Arg(p.Named("polyregion_result", p.Type.IntS32)))
    val renamed   = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(collision)), fortran)
    assert(renamed.contains("polyregion_result, polyregion_result_)"))
    assert(renamed.contains("type(*), intent(inout) :: polyregion_result_"))
  }

  test("Fortran rejects overloads that collide after generic result adaptation") {
    val returned = p.FunctionDecl(
      p.Sym("example.select"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("value", t))),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val unit = returned.copy(args = List(p.Arg(p.Named("lhs", t)), p.Arg(p.Named("rhs", t))), rtn = p.Type.Unit0)
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(returned, unit)), fortran)
    )

    assert(error.getMessage.contains("after result adaptation with 2 arguments"))
  }

  test("Fortran rejects overloads that mix projected functions and subroutines") {
    val function = p.FunctionDecl(
      p.Sym("example.select"),
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val subroutine = function.copy(
      args = List(p.Arg(p.Named("lhs", p.Type.IntS32)), p.Arg(p.Named("rhs", p.Type.IntS32))),
      rtn = p.Type.Unit0
    )
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(function, subroutine)), fortran)
    )

    assert(error.getMessage.contains("cannot combine function and subroutine overloads"))
  }

  test("Fortran erases explicitly named struct values") {
    val point = p.Type.Struct(p.Sym("model.Point"), Nil)
    val inspect = p.FunctionDecl(
      p.Sym("example.inspect_point"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named("points", p.Type.Ptr(point, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("point", point))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(inspect)), fortran)

    assert(f90.contains("type(*), dimension(*), intent(in) :: points"))
    assert(f90.contains("type(*), intent(in) :: point"))
  }

  test("projection order and stale-output checks are deterministic") {
    val reversed = interfaceDef.copy(declarations = interfaceDef.declarations.reverse)
    val cpp      = InterfaceCodeGen.cppHeader(interfaceDef)

    assertEquals(InterfaceCodeGen.cppHeader(reversed), cpp)
    assertEquals(
      InterfaceCodeGen.fortranModule(reversed, fortran),
      InterfaceCodeGen.fortranModule(interfaceDef, fortran)
    )
    assertEquals(InterfaceCodeGen.scalaObject(reversed, scala), InterfaceCodeGen.scalaObject(interfaceDef, scala))

    val path = Files.createTempFile("polyregion-bindgen", ".hpp")
    try {
      Files.writeString(path, "stale")
      assert(InterfaceCodeGen.checkCurrent(path, cpp).isLeft)
      Files.writeString(path, cpp)
      assertEquals(InterfaceCodeGen.checkCurrent(path, cpp), Right(()))
      assert(InterfaceCodeGen.checkCurrent(path.resolveSibling("missing-generated-output"), cpp).isLeft)
    } finally Files.deleteIfExists(path)
  }

  test("overloads retain distinct Fortran procedures") {
    val overloaded = interfaceDef.copy(declarations = countValue :: interfaceDef.declarations)
    val f90        = InterfaceCodeGen.fortranModule(overloaded, fortran)

    assert(f90.contains("module procedure polyregion_count_o0"))
    assert(f90.contains("module procedure polyregion_count_o1"))
    assertEquals(
      InterfaceCodeGen.fortranModule(overloaded.copy(declarations = overloaded.declarations.reverse), fortran),
      f90
    )
  }

  test("pointer declarations require boundary metadata") {
    val invalid =
      interfaceDef.copy(declarations =
        List(count.copy(args = count.args.updated(0, count.args.head.copy(boundary = None))))
      )
    val error = intercept[IllegalArgumentException](InterfaceCodeGen.cppHeader(invalid))

    assert(error.getMessage.contains("has no boundary"))
  }

  test("C++ arrays and ordinary functions produce valid header definitions") {
    val array = p.FunctionDecl(
      p.Sym("example.array"),
      Nil,
      None,
      List(p.Arg(p.Named("values", p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global)))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef.copy(declarations = List(array)))

    assertEquals(cpp, golden("example-array.hpp"))
    assert(cpp.contains("inline void array(std::int32_t (&values)[4])"))
  }

  test("C++ declarations become fallbacks when the includer supplies an implementation") {
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef)

    assert(
      cpp.contains(
        "#ifdef POLYREGION_EXAMPLE_IMPLEMENT\n#define POLYREGION_EXAMPLE_FALLBACK " +
          "POLYREGION_EXAMPLE_ANNOTATE(\"polyregion_interface_fallback\")\n#else\n#define POLYREGION_EXAMPLE_FALLBACK\n"
      ),
      cpp
    )
    assert(
      cpp.contains(
        "POLYREGION_EXAMPLE_ANNOTATE(\"polyregion_interface:example:example.transform\") POLYREGION_EXAMPLE_FALLBACK //"
      ),
      cpp
    )
  }

  test("C++ callables are checked against const lvalue arguments") {
    val inspect = p.FunctionDecl(
      p.Sym("example.inspect"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t)))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef.copy(declarations = List(inspect)))

    assert(cpp.contains("std::invoke_result_t<Op &, const T &>"))
  }

  test("Scala preserves nested callable structure and qualifies struct types from the root") {
    val nested = p.FunctionDecl(
      p.Sym("example.nested"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named(
            "op",
            p.Type.Exec(Nil, List(p.Type.Exec(Nil, List(p.Type.IntS32), p.Type.IntS32)), p.Type.Bool1)
          )
        ),
        p.Arg(p.Named("value", p.Type.Struct(p.Sym("model.Value"), Nil)))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val source = InterfaceCodeGen.scalaObject(interfaceDef.copy(declarations = List(nested)), scala)

    assert(source.contains("op: (Int => Int) => Boolean"))
    assert(source.contains("value: _root_.model.Value"))
  }

  test("lossy overloads and Fortran case collisions are rejected") {
    val sameArity     = interfaceDef.copy(declarations = List(count, count.copy(rtn = p.Type.IntS64)))
    val overloadError = intercept[IllegalArgumentException](InterfaceCodeGen.scalaObject(sameArity, scala))
    assert(overloadError.getMessage.contains("cannot portably distinguish overloads"))

    val lower = count.copy(name = p.Sym("example.lookup"))
    val upper = countValue.copy(name = p.Sym("example.Lookup"))
    val caseError = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(lower, upper)), fortran)
    )
    assert(caseError.getMessage.contains("differ only by case"))
  }

  test("target identifiers are validated before source emission") {
    val cppReserved =
      interfaceDef.copy(name = p.Sym("class"), declarations = List(count.copy(name = p.Sym("class.count"))))
    assert(intercept[IllegalArgumentException](InterfaceCodeGen.cppHeader(cppReserved)).getMessage.contains("reserved"))

    val scalaReserved = interfaceDef.copy(declarations = List(count.copy(name = p.Sym("example.match"))))
    assert(
      intercept[IllegalArgumentException](InterfaceCodeGen.scalaObject(scalaReserved, scala)).getMessage
        .contains("reserved")
    )

    val fortranInvalid =
      interfaceDef.copy(
        declarations = List(count.copy(args = count.args.updated(1, p.Arg(p.Named("_n", p.Type.IntS32)))))
      )
    assert(
      intercept[IllegalArgumentException](InterfaceCodeGen.fortranModule(fortranInvalid, fortran)).getMessage.contains(
        "invalid Fortran"
      )
    )
  }

  private def compiled[A](sources: Map[String, String])(use: Path => A): Either[String, A] = {
    val dir = Files.createTempDirectory("polyregion-bindgen-java")
    try {
      val files = sources.map { case (name, text) =>
        val path = dir.resolve(name)
        Files.createDirectories(path.getParent)
        Files.writeString(path, text)
        path.toString
      }.toList
      val err = ByteArrayOutputStream()
      val rc = ToolProvider.getSystemJavaCompiler.run(
        null,
        null,
        PrintStream(err),
        List("--release", "8", "-Xlint:all,-options", "-Werror", "-proc:none", "-d", dir.resolve("out").toString)
          .appendedAll(files)*
      )
      if (rc == 0) Right(use(dir.resolve("out"))) else Left(err.toString)
    } finally Using.resource(Files.walk(dir))(_.sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete(_)))
  }

  private def javac(sources: Map[String, String]): String =
    compiled(sources)(_ => ()).fold(identity, _ => "")

  private val annotationSource =
    """package polyregion.javalang;
      |
      |import java.lang.annotation.*;
      |
      |@Retention(RetentionPolicy.CLASS)
      |@Target(ElementType.METHOD)
      |public @interface Interface {
      |  String value();
      |}
      |""".stripMargin

  private val launchSource =
    """package polyregion.javalang;
      |
      |public final class Launch {
      |  public static Object call(String identity, Object context, Object[] args) {
      |    return null;
      |  }
      |
      |  public static Object callSugar(String identity, Object context, Object[] args, int rank, int map) {
      |    return null;
      |  }
      |}
      |""".stripMargin

  private val sugarSource =
    """package polyregion.javalang;
      |
      |import java.lang.annotation.ElementType;
      |import java.lang.annotation.Retention;
      |import java.lang.annotation.RetentionPolicy;
      |import java.lang.annotation.Target;
      |
      |@Retention(RetentionPolicy.CLASS)
      |@Target(ElementType.METHOD)
      |public @interface LaunchSugar {
      |  String value();
      |  int rank();
      |  int map();
      |}
      |""".stripMargin

  test("Java projects declarations as annotated static stubs with generated callable types") {
    val source = InterfaceCodeGen.javaClass(interfaceDef, jvm)

    assertEquals(source, golden("ExampleInterface.java"))
    assert(source.contains("package example.bindings;"))
    assert(source.contains("import polyregion.javalang.Interface;"))
    assert(source.contains("public final class ExampleInterface {"))
    assert(!source.contains("polyregion_interface"))
    assert(
      source.contains(
        "@FunctionalInterface\n  public interface FnI_J extends java.io.Serializable {\n    long apply(int arg0);\n  }"
      )
    )
    assert(source.contains("public interface FnL_L<T0, R> extends java.io.Serializable {\n    R apply(T0 arg0);\n  }"))
    assert(
      source.contains(
        "@Interface(\"example:example.transform\")\n  public static void transform(int[] in, long[] out, int n, FnI_J op) {"
      )
    )
    assert(
      source.contains(
        "@Interface(\"example:example.transform\")\n  public static <T, U> void transform(T[] in, U[] out, int n, FnL_L<T, U> op) {"
      )
    )
    assert(source.contains("public static int count(float[] in, int n) {"))
    assert(source.contains("public static <T> int count(T[] in, int n) {"))
    assert(source.contains("return (Integer) Launch.call(\"example:example.count\", null, new Object[] {in, n});"))
  }

  test("Java emits a cross product over int, long, float and double plus one reference form") {
    val source     = InterfaceCodeGen.javaClass(interfaceDef, jvm)
    val Primitives = List("int" -> 'I', "long" -> 'J', "float" -> 'F', "double" -> 'D')

    assertEquals("""@Interface\("example:example.transform"\)""".r.findAllIn(source).size, 4 * 4 + 1)
    assertEquals("""@Interface\("example:example.count"\)""".r.findAllIn(source).size, 4 + 1)
    for {
      (a, ac) <- Primitives
      (b, bc) <- Primitives
    }
      assert(source.contains(s"void transform($a[] in, $b[] out, int n, Fn${ac}_$bc op)"), s"$a $b")
    assert(!source.contains("byte[]"))
    assert(!source.contains("boolean[]"))
  }

  test("Java boxes scalars that no array determines so widening cannot make overloads ambiguous") {
    val source = InterfaceCodeGen.javaClass(interfaceDef.copy(declarations = List(reduceInit)), jvm)

    assert(
      source.contains("public static long reduce_init(int[] in, int n, Long init, FnI_J map, FnJJ_J combine) {"),
      source
    )
    assert(
      source.contains("public static double reduce_init(float[] in, int n, Double init, FnF_D map, FnDD_D combine) {")
    )
    assert(source.contains("public static <T, U> U reduce_init("))
    assert(
      source.contains(
        "      T[] in,\n      int n,\n      U init,\n      FnL_L<T, U> map,\n      FnLL_L<U, U, U> combine) {"
      )
    )
    assert(!source.contains("long init"))
  }

  test("Java keeps a literal call unambiguous when a scalar-only type variable is in no array") {
    val fold = p.FunctionDecl(
      p.Sym("example.fold"),
      List(p.Type.Var("U")),
      None,
      List(p.Arg(p.Named("init", u)), p.Arg(p.Named("op", p.Type.Exec(Nil, List(u), u)))),
      Nil,
      Nil,
      u,
      p.Function.Affinity.Host
    )
    val caller =
      """package example.bindings;
        |
        |public final class Caller {
        |  public static void run() {
        |    ExampleInterface.fold(0L, x -> x);
        |  }
        |}
        |""".stripMargin
    val errors = javac(
      Map(
        "polyregion/javalang/Interface.java"   -> annotationSource,
        "polyregion/javalang/Launch.java"      -> launchSource,
        "polyregion/javalang/LaunchSugar.java" -> sugarSource,
        "example/bindings/ExampleInterface.java" -> InterfaceCodeGen
          .javaClass(interfaceDef.copy(declarations = List(fold)), jvm),
        "example/bindings/Caller.java" -> caller
      )
    )

    assertEquals(errors, "")
  }

  test("Java keeps a literal call unambiguous when a scalar-only type variable sits beside a primitive array") {
    val mix = p.FunctionDecl(
      p.Sym("example.mix"),
      List(p.Type.Var("T")),
      None,
      List(
        p.Arg(p.Named("init", t)),
        p.Arg(
          p.Named("in", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(4))))
        ),
        p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t)))
      ),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val caller =
      """package example.bindings;
        |
        |public final class Caller {
        |  public static void run() {
        |    int[] ints = new int[4];
        |    ExampleInterface.mix(0, ints, arg0 -> 0);
        |  }
        |}
        |""".stripMargin
    val errors = javac(
      Map(
        "polyregion/javalang/Interface.java"   -> annotationSource,
        "polyregion/javalang/Launch.java"      -> launchSource,
        "polyregion/javalang/LaunchSugar.java" -> sugarSource,
        "example/bindings/ExampleInterface.java" -> InterfaceCodeGen
          .javaClass(interfaceDef.copy(declarations = List(mix)), jvm),
        "example/bindings/Caller.java" -> caller
      )
    )

    assertEquals(errors, "")
  }

  test("Java generated surface compiles for Java 8 and resolves overloads from the array arguments") {
    val caller =
      """package example.bindings;
        |
        |public final class Caller {
        |  public static final class Point {}
        |
        |  public static void run() {
        |    int[] ints = new int[4];
        |    long[] longs = new long[4];
        |    float[] floats = new float[4];
        |    double[] doubles = new double[4];
        |    Point[] points = new Point[4];
        |    Integer[] boxed = new Integer[4];
        |    ExampleInterface.transform(ints, longs, 4, x -> 0L);
        |    ExampleInterface.transform(floats, doubles, 4, x -> 0.0);
        |    ExampleInterface.transform(doubles, ints, 4, x -> 0);
        |    ExampleInterface.transform(points, boxed, 4, x -> 0);
        |    ExampleInterface.count(ints, 4);
        |    ExampleInterface.count(points, 4);
        |    long a = ExampleInterface.reduce_init(ints, 4, 0L, x -> 0L, (l, r) -> l + r);
        |    int b = ExampleInterface.reduce_init(ints, 4, 0, x -> x, (l, r) -> l + r);
        |    double c = ExampleInterface.reduce_init(floats, 4, 0.0, x -> x, (l, r) -> l + r);
        |    float d = ExampleInterface.reduce_init(longs, 4, 0.0f, x -> x, (l, r) -> l + r);
        |    Point e = ExampleInterface.reduce_init(points, 4, new Point(), x -> x, (l, r) -> l);
        |  }
        |}
        |""".stripMargin
    val withScalar = interfaceDef.copy(declarations = reduceInit :: interfaceDef.declarations)
    val errors = javac(
      Map(
        "polyregion/javalang/Interface.java"     -> annotationSource,
        "polyregion/javalang/Launch.java"        -> launchSource,
        "polyregion/javalang/LaunchSugar.java"   -> sugarSource,
        "example/bindings/ExampleInterface.java" -> InterfaceCodeGen.javaClass(withScalar, jvm),
        "example/bindings/Caller.java"           -> caller
      )
    )

    assertEquals(errors, "")
  }

  test("Java overloads hand the element to the SAM for a primitive array and a generic array") {
    val applyOnce = p.FunctionDecl(
      p.Sym("example.apply_once"),
      List(p.Type.Var("T")),
      None,
      List(
        p.Arg(
          p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("f", p.Type.Exec(Nil, List(t), p.Type.IntS32)))
      ),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val stubCall = """return (Integer) Launch.call("example:example.apply_once", null, new Object[] {in, n, f});"""
    val source = InterfaceCodeGen
      .javaClass(p.Interface(p.Sym("example"), List(applyOnce)), jvm)
      .replace(stubCall, "return f.apply(in[0]);")
    val caller =
      """package example.bindings;
        |
        |public final class Caller {
        |  public static String run() {
        |    int[] ints = {7};
        |    String[] words = {"abc"};
        |    int primitive = ExampleInterface.apply_once(ints, 1, x -> x + 1);
        |    int generic = ExampleInterface.apply_once(words, 1, x -> x.length());
        |    return primitive + " " + generic;
        |  }
        |}
        |""".stripMargin
    val result = compiled(
      Map(
        "polyregion/javalang/Interface.java"     -> annotationSource,
        "polyregion/javalang/Launch.java"        -> launchSource,
        "polyregion/javalang/LaunchSugar.java"   -> sugarSource,
        "example/bindings/ExampleInterface.java" -> source,
        "example/bindings/Caller.java"           -> caller
      )
    ) { out =>
      Using.resource(URLClassLoader(Array(out.toUri.toURL), getClass.getClassLoader)) { loader =>
        loader.loadClass("example.bindings.Caller").getMethod("run").invoke(null).asInstanceOf[String]
      }
    }

    assertEquals(result, Right("8 3"))
  }

  test("Java rejects type variables that no array or scalar argument determines") {
    val undetermined = p.FunctionDecl(
      p.Sym("example.generate"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("n", p.Type.IntS32)), p.Arg(p.Named("op", p.Type.Exec(Nil, Nil, t)))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.javaClass(interfaceDef.copy(declarations = List(undetermined)), jvm)
    )

    assert(error.getMessage.contains("not determined by an array or scalar argument"), error.getMessage)
  }

  test("Java rejects a type variable whose exact width no primitive binding matches") {
    val wide: p.Type.Var = p.Type.Var("T", Some(12))
    val undersized = p.FunctionDecl(
      p.Sym("example.wide"),
      List(wide),
      None,
      List(
        p.Arg(
          p.Named("in", p.Type.Ptr(wide, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32))
      ),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.javaClass(interfaceDef.copy(declarations = List(undersized)), jvm)
    )

    assert(error.getMessage.contains("12 bytes"), error.getMessage)
  }

  test("Java qualifies struct types and keeps concrete callable shapes") {
    val point = p.Type.Struct(p.Sym("model.Point"), Nil)
    val inspect = p.FunctionDecl(
      p.Sym("example.inspect"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named("points", p.Type.Ptr(point, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("op", p.Type.Exec(Nil, List(point, p.Type.IntS16), p.Type.Bool1))),
        p.Arg(p.Named("done", p.Type.Exec(Nil, Nil, p.Type.Unit0)))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val source = InterfaceCodeGen.javaClass(interfaceDef.copy(declarations = List(inspect)), jvm)

    assert(
      source.contains(
        "public static void inspect(model.Point[] points, int n, FnLS_Z<model.Point> op, Fn_V done) {"
      ),
      source
    )
    assert(
      source.contains(
        "public interface FnLS_Z<T0> extends java.io.Serializable {\n    boolean apply(T0 arg0, short arg1);\n  }"
      )
    )
    assert(source.contains("public interface Fn_V extends java.io.Serializable {\n    void apply();\n  }"))
    assertEquals("@Interface".r.findAllIn(source).size, 1)
  }

  test("Java projection is deterministic and rejects reserved identifiers") {
    val reversed = interfaceDef.copy(declarations = interfaceDef.declarations.reverse)
    assertEquals(InterfaceCodeGen.javaClass(reversed, jvm), InterfaceCodeGen.javaClass(interfaceDef, jvm))
    assert(InterfaceCodeGen.javaClass(interfaceDef, jvm).forall(_ < 128))

    val reserved = interfaceDef.copy(declarations = List(count.copy(name = p.Sym("example.default"))))
    assert(
      intercept[IllegalArgumentException](InterfaceCodeGen.javaClass(reserved, jvm)).getMessage.contains("reserved")
    )

    val parameter = count.copy(args = count.args.updated(1, p.Arg(p.Named("new", p.Type.IntS32))))
    assert(
      intercept[IllegalArgumentException](
        InterfaceCodeGen.javaClass(interfaceDef.copy(declarations = List(parameter)), jvm)
      ).getMessage.contains("reserved")
    )
  }

  private val regionContext = RegionConfig("polyregion.jvm.runtime.Context")
  private val regionJava    = JavaConfig("polyregion", "Offload", Some(regionContext))
  private val regionScala   = ScalaConfig("polyregion", "offload", Some(regionContext))
  private val regionCpp     = CppRegionConfig("polyregion::offload", List("polystl/algorithm_impl.h"))

  private val contextSource =
    """package polyregion.jvm.runtime;
      |
      |public final class Context {}
      |""".stripMargin

  test("C++ core header adds a context overload per declaration only when asked") {
    val plain = InterfaceCodeGen.cppHeader(Core.interfaceDef)
    assert(!plain.contains("polyregion_interface_context"))
    assert(!plain.contains("void *context"))

    val withContext = InterfaceCodeGen.cppHeader(Core.interfaceDef, true)
    assertEquals(
      """polyregion_interface_context""".r.findAllIn(withContext).size,
      Core.interfaceDef.declarations.size * 1
    )
    assert(withContext.contains("inline void for_each(void *context, std::int64_t n, Body body)"))
    assert(withContext.contains("POLYREGION_POLYREGION_CORE_IMPLEMENT(for_each, context, n, body);"))
    assert(
      withContext.contains(
        "inline T transform_reduce(void *context, std::int64_t n, T init, Map map, Combine combine)"
      )
    )
    assert(
      withContext.contains("POLYREGION_POLYREGION_CORE_IMPLEMENT(transform_reduce, context, n, init, map, combine);")
    )
    assertEquals("""polyregion_interface:polyregion.core:polyregion.core.for_each""".r.findAllIn(withContext).size, 2)
  }

  test("C++ context overloads reject a declaration that already has a context parameter") {
    val clash = count.copy(
      name = p.Sym("example.clash"),
      args = List(p.Arg(p.Named("context", p.Type.IntS64)))
    )
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.cppHeader(p.Interface(p.Sym("example"), List(clash)), true)
    )
    assert(error.getMessage.contains("context"))
  }

  test("C++ region header wraps the core interface with context, linearised and task forms") {
    val header = InterfaceCodeGen.cppRegionHeader(Core.interfaceDef, regionCpp)
    assert(header.contains("namespace polyregion::offload {"))
    assert(header.contains("#include \"polystl/algorithm_impl.h\""))
    assert(header.contains("polyrt_context_create"))
    assert(header.contains("::polyregion::core::for_each(n, body);"))
    assert(header.contains("::polyregion::core::for_each(context.handle(), n, body);"))
    assert(header.contains("inline void for_each(const Context &context, std::int64_t n, Body body)"))
    assert(header.contains("inline void for_each(std::int64_t n0, std::int64_t n1, std::int64_t n2, Body body)"))
    assert(header.contains("return ::polyregion::core::transform_reduce(n, init, map, combine);"))
    assert(header.contains("inline void task(const Context &context, Body body)"))
    assert(header.forall(_ < 128))
    assertEquals(header, InterfaceCodeGen.cppRegionHeader(Core.interfaceDef, regionCpp))
  }

  test("Scala region projection overloads every stub with a context and adds linearised sugar") {
    val source = InterfaceCodeGen.scalaObject(Core.interfaceDef, regionScala)
    assert(source.contains("inline def for_each(n: Long, inline body: Long => Unit): Unit ="))
    assert(
      source.contains(
        "inline def for_each(context: polyregion.jvm.runtime.Context, n: Long, inline body: Long => Unit): Unit ="
      )
    )
    assert(
      source.contains(
        "def transform_reduce[T](\n      context: polyregion.jvm.runtime.Context,\n      n: Long,\n      init: T"
      )
    )
    assertEquals("""@compileTimeOnly""".r.findAllIn(source).size, 2 * Core.interfaceDef.declarations.size)
    assert(source.contains("inline def for_each(n0: Long, n1: Long, inline body: (Long, Long) => Unit): Unit ="))
    assert(source.contains("body(i / n1, i % n1)"))
    assert(source.contains("body(i / (n1 * n2), i / n2 % n1, i % n2)"))
    assert(source.contains("inline def task(inline body: () => Unit): Unit ="))
    assert(
      !InterfaceCodeGen
        .scalaObject(Core.interfaceDef, ScalaConfig("polyregion", "offload"))
        .contains("context: polyregion.jvm.runtime.Context")
    )
  }

  test("Java region projection compiles for Java 8 and resolves context overloads with implicit lambdas") {
    val caller =
      """package polyregion;
        |
        |import polyregion.jvm.runtime.Context;
        |
        |public final class Caller {
        |  public static String run(Context ctx) {
        |    Offload.for_each(8L, i -> { });
        |    Offload.for_each(ctx, 8L, i -> { });
        |    long a = Offload.transform_reduce(8L, 0L, i -> i, (l, r) -> l + r);
        |    int b = Offload.transform_reduce(8L, 0, i -> (int) i, (l, r) -> l + r);
        |    double c = Offload.transform_reduce(ctx, 8L, 0.0, i -> i, (l, r) -> l + r);
        |    float d = Offload.transform_reduce(ctx, 8L, 0.0f, i -> i, (l, r) -> l + r);
        |    return a + " " + b + " " + c + " " + d;
        |  }
        |}
        |""".stripMargin
    val errors = javac(
      Map(
        "polyregion/javalang/Interface.java"   -> annotationSource,
        "polyregion/javalang/Launch.java"      -> launchSource,
        "polyregion/javalang/LaunchSugar.java" -> sugarSource,
        "polyregion/jvm/runtime/Context.java"  -> contextSource,
        "polyregion/Offload.java"              -> InterfaceCodeGen.javaClass(Core.interfaceDef, regionJava),
        "polyregion/Caller.java"               -> caller,
        "polyregion/OffloadSurface.java" -> InterfaceCodeGen.javaSurface(
          Core.interfaceDef,
          regionJava,
          "OffloadSurface"
        )
      )
    )
    assertEquals(errors, "")
  }

  test("Java region projection without a region config has no context overloads") {
    val source = InterfaceCodeGen.javaClass(Core.interfaceDef, JavaConfig("polyregion", "Offload"))
    assert(!source.contains("Context"))
    val withRegion = InterfaceCodeGen.javaClass(Core.interfaceDef, regionJava)
    assertEquals("""@Interface""".r.findAllIn(withRegion).size, 2 + 8 * (Core.interfaceDef.declarations.size - 1))
  }

  test("Scala region surface calls every overload and sugar form") {
    val surface = InterfaceCodeGen.scalaSurface(Core.interfaceDef, regionScala, "OffloadSurface")
    assert(surface.contains("InterfaceSurfaceCheck.erase"))
    assert(surface.contains("offload.for_each(context, 0L, (arg0: Long) => ())"))
    assert(
      surface.contains("offload.transform_reduce(0L, 0.0, (arg0: Long) => 0.0, (arg0: Double, arg1: Double) => 0.0)")
    )
    assert(surface.contains("offload.for_each(0L, 0L, 0L, (arg0: Long, arg1: Long, arg2: Long) => ())"))
    assert(surface.contains("offload.task(context, () => ())"))
  }

  private def evaluate(expression: String, env: Map[String, Long]): Long = {
    val tokens         = "[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[()*/%]".r.findAllIn(expression).toList
    var rest           = tokens
    def next(): String = { val head = rest.head; rest = rest.tail; head }
    def atom(): Long = next() match {
      case "("                           => val value = product(); next(); value
      case number if number.head.isDigit => number.toLong
      case name                          => env(name)
    }
    def product(): Long = {
      var value = atom()
      while (rest.headOption.exists(Set("*", "/", "%"))) next() match {
        case "*" => value *= atom()
        case "/" => value /= atom()
        case _   => value %= atom()
      }
      value
    }
    product()
  }

  test("linearised indices recover the row-major coordinates of every flat index") {
    for {
      rank <- InterfaceCodeGenSupport.Ranks
      dims <- List(List(2L, 3L, 4L), List(5L, 1L, 3L), List(1L, 7L, 2L)).map(_.take(rank))
    } {
      val names       = InterfaceCodeGenSupport.extents(rank)
      val expressions = InterfaceCodeGenSupport.linearisedIndices(rank, "i")
      (0L until dims.product).foreach { flat =>
        val actual   = expressions.map(evaluate(_, names.zip(dims).toMap + ("i" -> flat)))
        val expected = dims.indices.map(axis => flat / dims.drop(axis + 1).product % dims(axis)).toList
        assertEquals(actual, expected, s"rank $rank dims $dims flat $flat")
      }
    }
  }

  test("region projections check the extent product and reject reserved parameter names") {
    assert(
      InterfaceCodeGen
        .scalaObject(Core.interfaceDef, regionScala)
        .contains("Math.multiplyExact(Math.multiplyExact(n0, n1), n2)")
    )
    assert(
      InterfaceCodeGen
        .cppRegionHeader(Core.interfaceDef, regionCpp)
        .contains("detail::extent(detail::extent(n0, n1), n2)")
    )
    List("i", "n0", "n2", "context").foreach { name =>
      val clash = p.FunctionDecl(
        p.Sym("example.clash"),
        Nil,
        None,
        List(
          p.Arg(p.Named("n", p.Type.IntS64)),
          p.Arg(p.Named(name, p.Type.IntS64)),
          p.Arg(p.Named("body", p.Type.Exec(Nil, List(p.Type.IntS64), p.Type.Unit0)))
        ),
        Nil,
        Nil,
        p.Type.Unit0,
        p.Function.Affinity.Host
      )
      val region = p.Interface(p.Sym("example"), List(clash))
      assert(
        intercept[IllegalArgumentException](InterfaceCodeGen.scalaObject(region, regionScala)).getMessage.contains(name)
      )
      assert(
        intercept[IllegalArgumentException](InterfaceCodeGen.cppRegionHeader(region, regionCpp)).getMessage
          .contains(name)
      )
    }
  }

  test("Java region sugar carries identities and compiles for rank two, three and tasks") {
    val source = InterfaceCodeGen.javaClass(Core.interfaceDef, regionJava)
    assert(
      source.contains(
        "@FunctionalInterface\n  public interface FnJJ_V extends java.io.Serializable {\n    void apply(long arg0, long arg1);\n  }"
      )
    )
    assert(
      source.contains(
        "@FunctionalInterface\n  public interface FnJJJ_J extends java.io.Serializable {\n    long apply(long arg0, long arg1, long arg2);\n  }"
      )
    )
    assert(source.contains("@LaunchSugar(\"polyregion.core:polyregion.core.for_each\", rank = 2, map = 0)"))
    assert(source.contains("@LaunchSugar(\"polyregion.core:polyregion.core.transform_reduce\", rank = 3, map = 1)"))
    assert(source.contains("public static void task(Fn_V body)"))
    assert(source.contains("public static void task(polyregion.jvm.runtime.Context context, Fn_V body)"))
    assert(
      source.contains(
        "Launch.callSugar(\"polyregion.core:polyregion.core.for_each\", null, new Object[] {n0, n1, body}, 2, 0);"
      )
    )

    val caller =
      """package polyregion;
        |
        |import polyregion.jvm.runtime.Context;
        |
        |public final class SugarCaller {
        |  public static long run(Context ctx) {
        |    int[] out = new int[24];
        |    Offload.for_each(2L, 3L, (i, j) -> out[(int) (i * 3 + j)] = 0);
        |    Offload.for_each(2L, 3L, 4L, (i, j, k) -> out[(int) ((i * 3 + j) * 4 + k)] = 0);
        |    long a = Offload.transform_reduce(2L, 3L, 0L, (i, j) -> i * 3 + j, (l, r) -> l + r);
        |    double b = Offload.transform_reduce(ctx, 2L, 3L, 4L, 0.0, (i, j, k) -> i + j + k, (l, r) -> l + r);
        |    Offload.task(() -> out[0] = 1);
        |    Offload.task(ctx, () -> out[1] = 1);
        |    return a + (long) b;
        |  }
        |}
        |""".stripMargin
    val errors = javac(
      Map(
        "polyregion/javalang/Interface.java"   -> annotationSource,
        "polyregion/javalang/Launch.java"      -> launchSource,
        "polyregion/javalang/LaunchSugar.java" -> sugarSource,
        "polyregion/jvm/runtime/Context.java"  -> contextSource,
        "polyregion/Offload.java"              -> source,
        "polyregion/SugarCaller.java"          -> caller
      )
    )
    assertEquals(errors, "")
  }
}
