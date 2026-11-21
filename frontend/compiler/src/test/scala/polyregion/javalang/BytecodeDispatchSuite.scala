package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

class BytecodeDispatchSuite extends munit.FunSuite {

  private val Source =
    """
      |public final class Disp {
      |  public static abstract class Shape { abstract int area(); }
      |  public static final class Square extends Shape { int s; Square(int s) { this.s = s; } int area() { return s * s; } }
      |  public static final class Rect extends Shape {
      |    int w; int h;
      |    Rect(int w, int h) { this.w = w; this.h = h; }
      |    int area() { return w * h; }
      |  }
      |  public interface Scorer { int score(int x); }
      |  public static final class Doubler implements Scorer { public int score(int x) { return 2 * x; } }
      |  public static final class Negator implements Scorer { public int score(int x) { return -x; } }
      |  public static abstract class Base { int id() { return 1; } }
      |  public static final class Leaf extends Base { }
      |  public static abstract class Stateful { int k; abstract int get(); }
      |  public static final class Impl extends Stateful { int get() { return k; } }
      |  public static class Seed { int x; }
      |  public static class Middle extends Seed { }
      |  public static final class Bottom extends Middle { int get() { return x; } }
      |  public final class Holder { int k; public class Inner { int m() { return k; } } }
      |  public static abstract class Template {
      |    abstract int part();
      |    final int whole() { return part() * 2; }
      |  }
      |  public static final class Half extends Template { int part() { return 5; } }
      |  public static int areaOf(int a, int b) { Shape s = a > b ? new Square(a) : new Rect(a, b); return s.area(); }
      |  public static int scored(int x, boolean f) { Scorer s = f ? new Doubler() : new Negator(); return s.score(x); }
      |  public static int bare(Shape s) { return s.area(); }
      |  public static int inherits() { Base b = new Leaf(); return b.id(); }
      |  public static int stateful() { Stateful s = new Impl(); return s.get(); }
      |  public static int statefulBare(Stateful s) { return s.get(); }
      |  public static int templated() { Template t = new Half(); return t.whole(); }
      |  public static int threeLevels() { return new Bottom().get(); }
      |  public static int viaInner(Holder.Inner i) { return i.m(); }
      |  public static int viaStatic(int a) { return areaOf(a, a + 1) + areaOf(a + 1, a); }
      |}
      |""".stripMargin

  private lazy val loaded = JavaInMemory.compile("Disp", Source)

  private def descOf(name: String): String =
    Type.getMethodDescriptor(loaded.loadClass("Disp").getDeclaredMethods.find(_.getName == name).get)

  private def resolved(name: String): Result[BytecodeResolver.Resolved] =
    BytecodeResolver.resolve(List(MethodRef("Disp", name, descOf(name))), loaded.classes.get)

  private def refusal(name: String): String = resolved(name) match {
    case Left(e)  => e.getMessage
    case Right(_) => fail(s"`$name` was expected to be refused")
  }

  private val Area = OpenSite("Disp$Shape", "area", "()I")

  private def ref(name: String): p.Type = p.Type.Ptr(p.Type.Struct(p.Sym(name), Nil), p.Type.Space.Private)

  test("an open call is left to the devirtualiser with the receiver as its receiver") {
    val r = resolved("areaOf").fold(e => fail(e.getMessage), identity)
    assertEquals(r.sites, List(Area))
    val calls = r.functions.flatMap(_.collectWhere[p.Expr] {
      case i: p.Expr.Invoke if i.calleeSym.contains(Area.callee) => i
    })
    assertEquals(calls.size, 1)
    assertEquals(calls.head.receiver.map(_.tpe), Some(ref("Disp.Shape")))
    assertEquals(calls.head.args, Nil)
  }

  test("the classes the region builds are the candidates, in name order") {
    val r = resolved("areaOf").fold(e => fail(e.getMessage), identity)
    assertEquals(
      r.dispatch(Area.callee),
      List(
        Arm(p.Sym("Disp.Rect"), p.Sym("Disp$Rect.area"), Nil),
        Arm(p.Sym("Disp.Square"), p.Sym("Disp$Square.area"), Nil)
      )
    )
    assertEquals(r.owners, Map(p.Sym("Disp.Shape") -> List(p.Sym("Disp.Rect"), p.Sym("Disp.Square"))))
  }

  test("each candidate is lowered once, with its object as the receiver") {
    val r     = resolved("viaStatic").fold(e => fail(e.getMessage), identity)
    val names = r.functions.map(_.name.repr)
    assertEquals(names.count(_ == "Disp$Square.area"), 1)
    assertEquals(names.count(_ == "Disp$Rect.area"), 1)
    val square = r.functions.find(_.name.repr == "Disp$Square.area").get
    assertEquals(square.receiver.map(_.named.tpe), Some(ref("Disp.Square")))
    assertEquals(square.args, Nil)
    assertEquals(r.structs.map(_.name.repr).sorted, List("Disp.Rect", "Disp.Shape", "Disp.Square"))
  }

  test("an interface call is dispatched like a class call") {
    val r    = resolved("scored").fold(e => fail(e.getMessage), identity)
    val site = OpenSite("Disp$Scorer", "score", "(I)I")
    assertEquals(r.sites, List(site))
    assertEquals(
      r.dispatch(site.callee),
      List(
        Arm(p.Sym("Disp.Doubler"), p.Sym("Disp$Doubler.score"), List(0)),
        Arm(p.Sym("Disp.Negator"), p.Sym("Disp$Negator.score"), List(0))
      )
    )
  }

  test("the source tier's candidates join the ones the region builds") {
    val sourced = (site: OpenSite) =>
      List(Arm(p.Sym("Disp.Triangle"), p.Sym("Disp.Triangle.area"), Nil)).filter(_ => site == Area).success
    val r = BytecodeResolver
      .resolve(List(MethodRef("Disp", "areaOf", descOf("areaOf"))), loaded.classes.get, sourced)
      .fold(e => fail(e.getMessage), identity)
    assertEquals(r.dispatch(Area.callee).map(_.cls.repr), List("Disp.Rect", "Disp.Square", "Disp.Triangle"))
  }

  test("a receiver nothing implements is refused") {
    val message = refusal("bare")
    assert(message.contains("Cannot devirtualise `Disp$Shape.area()I`"), message)
    assert(message.contains("nothing to compile into the kernel"), message)
  }

  test("a candidate that inherits the method is refused") {
    val message = refusal("inherits")
    assert(message.contains("`Disp$Leaf` is a candidate receiver of `Disp$Base.id()I`"), message)
    assert(message.contains("inherits its implementation"), message)
  }

  test("an object whose superclass has state is refused where it is built") {
    assert(refusal("stateful").contains("declares instance state"))
  }

  test("state held anywhere on the superclass chain refuses the allocation") {
    val message = refusal("threeLevels")
    assert(message.contains("Unsupported allocation of `Disp$Bottom`"), message)
    assert(message.contains("declares instance state"), message)
  }

  test("an owner whose only state is the outer instance of an inner class is refused") {
    val sourced = (_: OpenSite) => List(Arm(p.Sym("Disp.Other"), p.Sym("Disp.Other.m"), Nil)).success
    val r =
      BytecodeResolver.resolve(List(MethodRef("Disp", "viaInner", descOf("viaInner"))), loaded.classes.get, sourced)
    val message = r.left.map(_.getMessage).left.getOrElse(fail("expected a refusal"))
    assert(message.contains("`Disp$Holder$Inner` carries instance state"), message)
  }

  test("a class both the source declares and the region builds is lowered from its source alone") {
    val sourced = (site: OpenSite) =>
      List(Arm(p.Sym("Disp.Square"), p.Sym("Disp.Square.area"), Nil)).filter(_ => site == Area).success
    val r = BytecodeResolver
      .resolve(List(MethodRef("Disp", "areaOf", descOf("areaOf"))), loaded.classes.get, sourced)
      .fold(e => fail(e.getMessage), identity)
    assertEquals(
      r.dispatch(Area.callee).find(_.cls == p.Sym("Disp.Square")).map(_.impl),
      Some(p.Sym("Disp.Square.area"))
    )
    assert(!r.functions.exists(_.name.repr == "Disp$Square.area"), r.functions.map(_.name.repr).toString)
  }

  test("an owner with instance state is refused for the candidates the source tier adds") {
    val sourced = (_: OpenSite) => List(Arm(p.Sym("Disp.Other"), p.Sym("Disp.Other.get"), Nil)).success
    val site    = OpenSite("Disp$Stateful", "get", "()I")
    val r =
      BytecodeResolver.resolve(
        List(MethodRef("Disp", "statefulBare", descOf("statefulBare"))),
        loaded.classes.get,
        sourced
      )
    val message = r.left.map(_.getMessage).left.getOrElse(fail("expected a refusal"))
    assert(message.contains("`" + site.repr + "`: `Disp$Stateful` carries instance state"), message)
  }

  test("a final method on an abstract class is still direct") {
    val r = resolved("templated").fold(e => fail(e.getMessage), identity)
    assertEquals(r.sites.map(_.repr), List("Disp$Template.part()I"))
    assert(r.functions.exists(_.name.repr == "Disp$Template.whole"), r.functions.map(_.name.repr).toString)
  }
}
