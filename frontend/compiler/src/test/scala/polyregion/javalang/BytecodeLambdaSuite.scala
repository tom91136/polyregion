package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

class BytecodeLambdaSuite extends munit.FunSuite {

  private val Source =
    """
      |import java.util.function.Function;
      |import java.util.function.IntBinaryOperator;
      |import java.util.function.IntPredicate;
      |import java.util.function.IntUnaryOperator;
      |
      |public final class Lambdas {
      |  public static int local(int a) { IntUnaryOperator f = x -> x + 1; return f.applyAsInt(a); }
      |  public static int capturing(int a, int b) {
      |    IntUnaryOperator f = x -> x * b;
      |    return f.applyAsInt(a) + f.applyAsInt(a + 1);
      |  }
      |  public static int binary(int a, int b) { IntBinaryOperator f = (x, y) -> x * y + 1; return f.applyAsInt(a, b); }
      |  public static int reference(int a) { IntUnaryOperator f = Math::abs; return f.applyAsInt(a); }
      |  public static int passed(int a) { IntUnaryOperator f = x -> x; return apply(f, a); }
      |  static int apply(IntUnaryOperator f, int a) { return f.applyAsInt(a); }
      |  public static IntUnaryOperator returned(int a) { return x -> x + a; }
      |  public static int boxed(int a) { Function<Integer, Integer> f = x -> x + 1; return f.apply(a); }
      |  public static int composed(int a) { IntPredicate f = x -> x > 0; return f.negate().test(a) ? 1 : 0; }
      |  public static int concat(int a) { return ("n=" + a).hashCode(); }
      |}
      |""".stripMargin

  private lazy val loaded = JavaInMemory.compile("Lambdas", Source)

  private def descOf(name: String): String =
    Type.getMethodDescriptor(loaded.loadClass("Lambdas").getDeclaredMethods.find(_.getName == name).get)

  private def lowered(name: String): Result[BytecodeRemapper.Lowered] =
    BytecodeRemapper.lowerWithCalls(loaded.bytesOf("Lambdas"), name, descOf(name), loaded.classes.get)

  private def ok(name: String): BytecodeRemapper.Lowered =
    lowered(name).fold(e => fail(s"`$name` was refused: ${e.getMessage}"), identity)

  private def refusal(name: String): String = lowered(name) match {
    case Left(e)   => e.getMessage
    case Right(fn) => fail(s"`$name` was expected to be refused but lowered to:\n${fn.fn.repr}")
  }

  private def invokes(fn: p.Function): List[p.Expr.Invoke] = fn.collectWhere[p.Expr] { case i: p.Expr.Invoke => i }

  test("a lambda called where it is made becomes a call to its body") {
    val l = ok("local")
    assertEquals(l.calls.map(_.name.takeWhile(_ != '$') + "$"), List("lambda$"))
    assertEquals(invokes(l.fn).map(_.args.size), List(1))
  }

  test("captured values go first") {
    val l = ok("capturing")
    assertEquals(l.calls.size, 1)
    assert(l.calls.head.desc == "(II)I", l.calls.head.desc)
    assertEquals(invokes(l.fn).map(_.args.size), List(2, 2))
  }

  test("a two argument lambda keeps both arguments") {
    assertEquals(invokes(ok("binary").fn).map(_.args.size), List(2))
  }

  test("a method reference to a static method is a call to it") {
    val l = ok("reference")
    assertEquals(l.fn.collectWhere[p.Expr] { case m: p.Expr.MathOp => m }.size + invokes(l.fn).size, 1)
  }

  test("the resolver lowers the body of the lambda") {
    BytecodeResolver.resolve(List(MethodRef("Lambdas", "local", descOf("local"))), loaded.classes.get) match {
      case Left(e)  => fail(e.getMessage)
      case Right(r) => assertEquals(r.functions.size, 2)
    }
  }

  test("a lambda handed to another method is refused") {
    val message = refusal("passed")
    assert(message.contains("functional value"), message)
  }

  test("a lambda that is returned is refused") {
    val message = refusal("returned")
    assert(message.contains("IntUnaryOperator"), message)
  }

  test("a default method on a functional value is refused") {
    val message = refusal("composed")
    assert(message.contains("on a functional value; only `test` is resolved"), message)
  }

  test("a lambda over boxed types is refused when its body is lowered") {
    BytecodeResolver.resolve(List(MethodRef("Lambdas", "boxed", descOf("boxed"))), loaded.classes.get) match {
      case Left(e)  => assert(e.getMessage.contains("java.lang.Integer"), e.getMessage)
      case Right(r) => fail(s"lowered:\n${r.functions.map(_.repr).mkString("\n")}")
    }
  }

  test("string concatenation is refused by name") {
    val message = refusal("concat")
    assert(message.contains("Unsupported string concatenation; PolyRegion has no device-side String"), message)
  }
}
