package polyregion.javalang

import net.bytebuddy.jar.asm.Type as AsmType
import polyregion.ast.{PolyAST as p, *}

class BytecodeNativesSuite extends munit.FunSuite {

  private val Source =
    """
      |public final class Natives {
      |  public static double root(double a) { return Math.sqrt(a) + Math.abs(a); }
      |  public static int bits(int a) { return Integer.bitCount(a) + Integer.signum(a); }
      |  public static int leading(long a) { return Long.numberOfLeadingZeros(a); }
      |  public static int rounded(double a) { return (int) Math.round(a); }
      |  public static double smaller(double a, double b) { return Math.min(a, b); }
      |  public static int modded(int a, int b) { return Math.floorMod(a, b); }
      |  public static double remainder(double a) { return Math.IEEEremainder(a, a); }
      |  public static int viaHelper(int a) { return helper(a) + Integer.bitCount(a); }
      |  public static int helper(int a) { return Integer.rotateLeft(a, 3); }
      |}
      |""".stripMargin

  private lazy val natives = JavaInMemory.compile("Natives", Source)
  private lazy val bytes   = natives.bytesOf("Natives")
  private lazy val clazz   = natives.loadClass("Natives")

  private val classes: BytecodeResolver.ClassBytes = {
    case "Natives" => Some(bytes)
    case other     => BytecodeResolver.runtimeClasses(other)
  }

  private def methodOf(name: String) = clazz.getDeclaredMethods.find(_.getName == name).get

  private def resolved(name: String): Result[List[p.Function]] =
    BytecodeResolver.resolve(List(MethodRef("Natives", name, AsmType.getMethodDescriptor(methodOf(name)))), classes)

  private val doubles  = List(0d, -0d, 1d, -1d, 0.5d, 2.5d, -2.5d, 1e30, -1e30, Double.NaN, 0.49999999999999994, 1d / 3)
  private val ints     = List(0, 1, -1, 7, -7, 255, 0x0f0f0f0f, Int.MaxValue, Int.MinValue)
  private val longs    = List(0L, 1L, -1L, 1L << 40, Long.MaxValue, Long.MinValue)
  private val divisors = ints.filter(b => b != 0 && b != -1)

  private def agree(name: String, args: List[List[Any]]): Unit = {
    val fns = resolved(name).fold(e => fail(e.getMessage), identity)
    args.foreach { a =>
      val expected = Differential.invoke(methodOf(name), None, a).result
      val actual   = Differential.interpret(fns.head, a, fns.tail).result
      assertEquals(
        Differential.render(actual),
        Differential.render(expected),
        s"$name(${a.map(Differential.render(_))})"
      )
    }
  }

  test("math intrinsics come from the table")(agree("root", doubles.filterNot(_.isNaN).map(List(_))))
  test("bit counts")(agree("bits", ints.map(List(_))))
  test("leading zeros")(agree("leading", longs.map(List(_))))
  test("round is half up")(agree("rounded", doubles.filter(_.abs < 1e9).map(List(_))))
  test("min propagates NaN and orders signed zero")(
    agree(
      "smaller",
      for {
        a <- doubles
        b <- doubles
      } yield List(a, b)
    )
  )
  test("floor mod")(
    agree(
      "modded",
      for {
        a <- ints
        b <- divisors
      } yield List(a, b)
    )
  )

  test("helpers are lowered once beside their callers") {
    val fns = resolved("viaHelper").fold(e => fail(e.getMessage), identity)
    assertEquals(fns.map(_.name.repr).distinct.size, fns.size)
    agree("viaHelper", ints.map(List(_)))
  }

  test("an intrinsic the table lacks is refused, not lowered from a native body") {
    resolved("remainder") match {
      case Left(e) =>
        assert(e.getMessage.contains("unsupported JVM intrinsic java/lang/Math.IEEEremainder(DD)D"), e.getMessage)
      case Right(_) => fail("Math.IEEEremainder has no table entry and must be refused")
    }
  }
}
