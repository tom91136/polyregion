package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

class BytecodeReceiverSuite extends munit.FunSuite {

  private val Source =
    """
      |public final class Receivers {
      |  public static final class Counter {
      |    int n;
      |    Counter(int n) { this.n = n; }
      |    int next() { return n++; }
      |    int add(int k) { n += k; return n; }
      |    private int twice() { return n * 2; }
      |    int viaPrivate() { return twice() + 1; }
      |    int get() { return n; }
      |  }
      |  public static class Open { int v; int get() { return v; } final int sealed() { return v; } }
      |  public static abstract class Shape { abstract int area(); }
      |  public static int counting(int a) { Counter c = new Counter(a); c.next(); c.add(3); return c.next() + c.get(); }
      |  public static int privately(int a) { return new Counter(a).viaPrivate(); }
      |  public static int open(Open o) { return o.get(); }
      |  public static int sealedMethod(Open o) { return o.sealed(); }
      |  public static int abstractCall(Shape s) { return s.area(); }
      |}
      |""".stripMargin

  private lazy val loaded = JavaInMemory.compile("Receivers", Source)
  private val CounterRef  = p.Type.Ptr(p.Type.Struct(p.Sym("Receivers.Counter"), Nil), p.Type.Space.Private)
  private val Counter     = "Receivers$Counter"

  private def descOf(klass: String, name: String): String =
    Type.getMethodDescriptor(loaded.loadClass(klass.replace('/', '.')).getDeclaredMethods.find(_.getName == name).get)

  private def lowered(klass: String, name: String, receiver: Boolean = false): Result[BytecodeRemapper.Lowered] =
    BytecodeRemapper.lowerWithCalls(
      loaded.bytesOf(klass),
      name,
      descOf(klass, name),
      loaded.classes.get,
      withReceiver = receiver
    )

  private def refusal(name: String): String = lowered("Receivers", name) match {
    case Left(e)   => e.getMessage
    case Right(fn) => fail(s"`$name` was expected to be refused but lowered to:\n${fn.fn.repr}")
  }

  test("calls on an object of a final class are direct calls with the object first") {
    val l = lowered("Receivers", "counting").fold(e => fail(e.getMessage), identity)
    assertEquals(
      l.calls.map(c => (c.name, c.instance)).sorted,
      List("<init>" -> false, "add" -> true, "get" -> true, "next" -> true).sorted
    )
    val invokes = l.fn.collectWhere[p.Expr] { case i: p.Expr.Invoke if i.calleeName.last == "next" => i }
    assertEquals(invokes.map(_.args.map(_.tpe)), List(List(CounterRef), List(CounterRef)))
  }

  test("a private method is direct even on a class that is not final") {
    val l = lowered("Receivers", "privately").fold(e => fail(e.getMessage), identity)
    assert(l.calls.exists(_.name == "viaPrivate"), l.calls.toString)
  }

  test("an instance method lowered for a receiver takes it as its first argument") {
    val l = lowered(Counter, "next", receiver = true).fold(e => fail(e.getMessage), identity)
    assertEquals(l.fn.args.map(_.named.tpe), List(CounterRef))
    val stores = l.fn.collectWhere[p.Stmt] {
      case p.Stmt.Mut(lhs, _) if lhs.steps == List(p.PathStep.Deref, p.PathStep.Field("n")) => lhs
    }
    assertEquals(stores.size, 1)
  }

  test("a method that reads its receiver is still refused without one") {
    assert(lowered(Counter, "next").left.exists(_.getMessage.contains("uses its receiver")))
  }

  test("the resolver lowers each method once and the struct once") {
    val root = MethodRef("Receivers", "counting", descOf("Receivers", "counting"))
    BytecodeResolver.resolve(List(root), loaded.classes.get) match {
      case Left(e) => fail(e.getMessage)
      case Right(r) =>
        assertEquals(
          r.functions.map(_.name.repr).sorted,
          List(
            "Receivers$Counter.<init>",
            "Receivers$Counter.add",
            "Receivers$Counter.get",
            "Receivers$Counter.next",
            "Receivers.counting"
          )
        )
        assertEquals(r.structs.map(_.name), List(p.Sym("Receivers.Counter")))
    }
  }

  test("a call that an unknown subclass could override is refused") {
    val message = refusal("open")
    assert(message.contains("Unsupported call `Receivers$Open.get()I`"), message)
    assert(message.contains("cannot resolve statically"), message)
  }

  test("an abstract method is refused") {
    assert(refusal("abstractCall").contains("cannot resolve statically"))
  }

  test("a final method on an open class is direct") {
    val l = lowered("Receivers", "sealedMethod").fold(e => fail(e.getMessage), identity)
    assertEquals(l.calls.map(_.name), List("sealed"))
  }
}
