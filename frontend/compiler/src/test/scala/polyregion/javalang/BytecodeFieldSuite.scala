package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

class BytecodeFieldSuite extends munit.FunSuite {

  private val Source =
    """
      |public final class Fields {
      |  public static final class Point { int x; int y; double w; int[] cells; boolean on; }
      |  public static final class Holder { Point inner; }
      |  public static final class Outer {
      |    int tag;
      |    final class Inner { int v; int read() { return v + tag; } }
      |  }
      |  static int counter = 3;
      |  private int own = 1;
      |  public static int readX(Point p) { return p.x; }
      |  public static void writeY(Point p, int v) { p.y = v; }
      |  public static double readW(Point p) { return p.w; }
      |  public static boolean readOn(Point p) { return p.on; }
      |  public static int cell(Point p, int i) { return p.cells[i]; }
      |  public static void setCell(Point p, int i, int v) { p.cells[i] = v; }
      |  public static int bump(Point p) { p.x += 1; p.y++; return p.x + p.y; }
      |  public static int viaCallee(Point p) { return readX(p) + 1; }
      |  public static Point identity(Point p) { return p; }
      |  public static int nested(Holder h) { return h.inner.x; }
      |  public static int staticRead(int a) { return counter + a; }
      |  public static void staticWrite(int a) { counter = a; }
      |  public int instanceRead(int a) { return a + own; }
      |  public static int platform(java.util.List<String> xs) { return 1; }
      |}
      |""".stripMargin

  private lazy val loaded = JavaInMemory.compile("Fields", Source)
  private val Point       = p.Type.Struct(p.Sym("Fields.Point"), Nil)
  private val PointRef    = p.Type.Ptr(Point, p.Type.Space.Private)

  private def descOf(name: String): String = {
    val m = loaded.loadClass("Fields").getDeclaredMethods.find(_.getName == name).get
    Type.getMethodDescriptor(m)
  }

  private def lowered(name: String): BytecodeRemapper.Lowered =
    BytecodeRemapper
      .lowerWithCalls(loaded.bytesOf("Fields"), name, descOf(name), loaded.classes.get)
      .fold(e => fail(s"`$name` was refused: ${e.getMessage}"), identity)

  private def refusal(name: String): String =
    BytecodeRemapper.lowerWithCalls(loaded.bytesOf("Fields"), name, descOf(name), loaded.classes.get) match {
      case Left(e)   => e.getMessage
      case Right(fn) => fail(s"`$name` was expected to be refused but lowered to:\n${fn.fn.repr}")
    }

  private def selects(fn: p.Function, steps: List[p.PathStep]): List[p.Term.Select] =
    fn.collectWhere[p.Term] { case s: p.Term.Select if s.steps == steps => s }

  test("an object parameter is a private pointer to the struct of its class") {
    assertEquals(lowered("readX").fn.args.map(_.named.tpe), List(PointRef))
  }

  test("the struct of a class lists the instance fields it can carry, in declaration order") {
    val Some(sdef) = lowered("readX").structs.headOption: @unchecked
    assertEquals(sdef.name, p.Sym("Fields.Point"))
    assertEquals(
      sdef.members.map(m => m.symbol -> m.tpe),
      List(
        "x"     -> p.Type.IntS32,
        "y"     -> p.Type.IntS32,
        "w"     -> p.Type.Float64,
        "cells" -> p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global),
        "on"    -> p.Type.Bool1
      )
    )
  }

  test("a field read goes through the pointer") {
    val reads = selects(lowered("readX").fn, List(p.PathStep.Deref, p.PathStep.Field("x")))
    assertEquals(reads.map(_.tpe), List(p.Type.IntS32))
    assertEquals(
      selects(lowered("readW").fn, List(p.PathStep.Deref, p.PathStep.Field("w"))).map(_.tpe),
      List(p.Type.Float64)
    )
    assertEquals(
      selects(lowered("readOn").fn, List(p.PathStep.Deref, p.PathStep.Field("on"))).map(_.tpe),
      List(p.Type.Bool1)
    )
  }

  test("a field write stores through the pointer") {
    val writes = lowered("writeY").fn.collectWhere[p.Stmt] {
      case p.Stmt.Mut(lhs, _) if lhs.steps == List(p.PathStep.Deref, p.PathStep.Field("y")) => lhs
    }
    assertEquals(writes.size, 1)
  }

  test("compound assignment and increment on fields read and write the same field") {
    val fn = lowered("bump").fn
    def stores(field: String) = fn.collectWhere[p.Stmt] {
      case p.Stmt.Mut(lhs, _) if lhs.steps == List(p.PathStep.Deref, p.PathStep.Field(field)) => lhs
    }
    assertEquals(stores("x").size, 1)
    assertEquals(stores("y").size, 1)
  }

  test("an array field is indexed through the pointer") {
    val fn = lowered("cell").fn
    assert(selects(fn, List(p.PathStep.Deref, p.PathStep.Field("cells"))).nonEmpty, fn.repr)
    val set = lowered("setCell").fn
    assert(selects(set, List(p.PathStep.Deref, p.PathStep.Field("cells"))).nonEmpty, set.repr)
  }

  test("the resolver returns each struct once and lowers the callee") {
    val root = MethodRef("Fields", "viaCallee", descOf("viaCallee"))
    BytecodeResolver.resolve(List(root), loaded.classes.get) match {
      case Left(e) => fail(e.getMessage)
      case Right(resolved) =>
        assertEquals(resolved.functions.size, 2)
        assertEquals(resolved.structs.map(_.name), List(p.Sym("Fields.Point")))
    }
  }

  test("a returned object is refused") {
    val message = refusal("identity")
    assert(message.contains("cannot be returned"), message)
    assert(message.contains("Fields.Point"), message)
  }

  test("a field the layout cannot carry is refused where it is accessed") {
    val message = refusal("nested")
    assert(message.contains("Unsupported field access `Fields$Holder.inner`"), message)
  }

  test("static fields are refused") {
    assert(refusal("staticRead").contains("Unsupported field access `Fields.counter`"))
    assert(refusal("staticWrite").contains("Unsupported field access `Fields.counter`"))
  }

  test("an instance method that reads its receiver is refused") {
    assert(refusal("instanceRead").contains("uses its receiver"))
  }

  test("a JDK class has no device layout") {
    assert(refusal("platform").contains("Cannot lower the JVM type `java.util.List`"))
  }
}
