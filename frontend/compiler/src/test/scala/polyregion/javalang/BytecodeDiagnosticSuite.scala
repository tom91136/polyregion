package polyregion.javalang

import polyregion.ast.{PolyAST as p, *}

class BytecodeDiagnosticSuite extends munit.FunSuite {

  private val UnsupportedSource =
    """
      |import java.util.function.IntUnaryOperator;
      |
      |public final class Unsupported {
      |  static int counter = 3;
      |  private int field = 7;
      |
      |  public int instanceMethod(int a) { return a + field; }
      |  public static int calls(int a) { return helper(a) + 1; }
      |  public static int helper(int a) { return a * 2; }
      |  public static int viaString(int a) { return String.valueOf(a).length(); }
      |  public static int strings(int a) { String s = "hello"; return s.length() + a; }
      |  public static int concat(int a) { String s = "n=" + a; return s.hashCode(); }
      |  public static int guarded(int a) { try { return 1 / a; } catch (ArithmeticException e) { return 0; } }
      |  public static int switched(int a) {
      |    switch (a) { case 0: return 10; case 1: return 20; case 2: return 30; default: return 40; }
      |  }
      |  public static int allocates(int n) { int[] xs = new int[n]; xs[0] = 1; return xs[0]; }
      |  public static int lengthOf(int[] xs) { return xs.length; }
      |  public static int staticField(int a) { return counter + a; }
      |  public static int lambda(int a) { IntUnaryOperator f = x -> x + 1; return f.applyAsInt(a); }
      |  public static int nulls(int[] xs) { return xs == null ? 0 : 1; }
      |  public static int locked(int a) { synchronized (Unsupported.class) { return a + 1; } }
      |}
      |""".stripMargin

  private lazy val unsupported = JavaInMemory.compile("Unsupported", UnsupportedSource)
  private lazy val bytes       = unsupported.bytesOf("Unsupported")
  private lazy val clazz       = unsupported.loadClass("Unsupported")

  private def failureOf(name: String): String = {
    val m    = clazz.getDeclaredMethods.find(_.getName == name).get
    val desc = net.bytebuddy.jar.asm.Type.getMethodDescriptor(m)
    BytecodeRemapper.lower(bytes, name, desc) match {
      case Left(e)   => e.getMessage
      case Right(fn) => fail(s"`$name` was expected to be refused but lowered to:\n${fn.repr}")
    }
  }

  private def refuses(name: String, fragment: String)(implicit loc: munit.Location): Unit = {
    val message = failureOf(name)
    assert(message.contains(fragment), s"`$name` failed with `$message`, which does not mention `$fragment`")
  }

  test("instance methods that read a field")(refuses("instanceMethod", "Unsupported field access `Unsupported.field`"))
  test("calls returning objects")(refuses("viaString", "java.lang.String"))
  test("string constants")(refuses("strings", "String"))
  test("string concatenation")(refuses("concat", "String"))
  test("exception handlers")(refuses("guarded", "catch"))
  test("switch")(refuses("switched", "switch"))
  test("array allocation")(refuses("allocates", "allocation"))
  test("array length")(refuses("lengthOf", "length"))
  test("static fields")(refuses("staticField", "Unsupported field access `Unsupported.counter`"))
  test("lambdas")(refuses("lambda", "Cannot lower the JVM type `java.util.function.IntUnaryOperator`"))
  test("reference comparisons")(refuses("nulls", "device heap"))
  test("monitors")(refuses("locked", "synchronized"))

  test("static calls are recorded for the resolver") {
    val m    = clazz.getDeclaredMethods.find(_.getName == "calls").get
    val desc = net.bytebuddy.jar.asm.Type.getMethodDescriptor(m)
    BytecodeRemapper.lowerWithCalls(bytes, "calls", desc) match {
      case Left(e) => fail(e.getMessage)
      case Right((fn, callees)) =>
        assertEquals(callees, List(MethodRef("Unsupported", "helper", "(I)I")))
        assert(fn.repr.contains("&Unsupported.helper"), fn.repr)
    }
  }

  test("a truncated class file is reported rather than silently mislowered") {
    Vineflower.classNameOf(Array[Byte](0xca.toByte, 0xfe.toByte, 0xba.toByte, 0xbe.toByte)) match {
      case Left(e)  => assert(Option(e).isDefined)
      case Right(_) => fail("a truncated class file should not parse")
    }
  }

  test("a method the class does not declare names the ones it does") {
    BytecodeRemapper.lower(bytes, "missing", "()V") match {
      case Left(e)  => assert(e.getMessage.contains("the class declares"), e.getMessage)
      case Right(_) => fail("a missing method should not lower")
    }
  }

  private val StructureSource =
    """
      |public final class Shapes {
      |  public static int diamond(int a) { int r; if (a > 0) { r = a; } else { r = -a; } return r + 1; }
      |  public static int loopBreak(int n) {
      |    int t = 0;
      |    for (int i = 0; i < n; i++) { if (i == 5) { break; } t += i; }
      |    return t;
      |  }
      |  public static int twoLoops(int n) {
      |    int t = 0;
      |    for (int i = 0; i < n; i++) { t += i; }
      |    for (int i = 0; i < n; i++) { t -= i; }
      |    return t;
      |  }
      |  public static int shortCircuit(int a, int b) {
      |    int r;
      |    if (a > 0 && b > 0) { r = a + b; } else { r = a - b; }
      |    return r * 2;
      |  }
      |  public static int labelledContinue(int n, int m) {
      |    int t = 0;
      |    outer:
      |    for (int i = 0; i < n; i++) {
      |      for (int j = 0; j < m; j++) { if (i * j > 30) { continue outer; } t += i * j; }
      |      t++;
      |    }
      |    return t;
      |  }
      |}
      |""".stripMargin

  private lazy val shapes     = JavaInMemory.compile("Shapes", StructureSource)
  private lazy val shapeBytes = shapes.bytesOf("Shapes")
  private lazy val shapeClass = shapes.loadClass("Shapes")

  private def lower(name: String): p.Function = {
    val m    = shapeClass.getDeclaredMethods.find(_.getName == name).get
    val desc = net.bytebuddy.jar.asm.Type.getMethodDescriptor(m)
    BytecodeRemapper.lower(shapeBytes, name, desc).fold(throw _, identity)
  }

  private def loops(fn: p.Function): Int = fn.repr.linesIterator.count(_.contains("while("))

  test("the loop count matches the source") {
    assertEquals(loops(lower("diamond")), 0)
    assertEquals(loops(lower("loopBreak")), 1)
    assertEquals(loops(lower("twoLoops")), 2)
    assertEquals(loops(lower("shortCircuit")), 0)
  }

  test("only the shapes that need it pay for the jump register") {
    assert(!lower("diamond").repr.contains("#jmp"))
    assert(!lower("twoLoops").repr.contains("#jmp"))
    assert(!lower("shortCircuit").repr.contains("#jmp"))
    assert(!lower("loopBreak").repr.contains("#jmp"))
    assert(lower("labelledContinue").repr.contains("#jmp"))
  }
}
