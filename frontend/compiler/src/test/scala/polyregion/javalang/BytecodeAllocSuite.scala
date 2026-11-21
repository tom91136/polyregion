package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

class BytecodeAllocSuite extends munit.FunSuite {

  private val Source =
    """
      |public final class Alloc {
      |  public static final class Box {
      |    int v;
      |    Box(int v) { this.v = v; }
      |    Box() { this(7); }
      |  }
      |  public static class Base { }
      |  public static final class Derived extends Base { int z; }
      |  public static int scratch(int n) { int[] xs = new int[n]; xs[0] = 1; return xs[0]; }
      |  public static int literal() { int[] xs = {3, 1, 4}; return xs[2]; }
      |  public static int objects(int a) { Box b = new Box(a); return b.v; }
      |  public static int delegating() { return new Box().v; }
      |  public static int derived() { return new Derived().z; }
      |  public static int boxes(int n) { Box[] bs = new Box[n]; return n; }
      |  public static int platform() { return new java.util.ArrayList<Integer>().size(); }
      |  public static Box leak(int a) { return new Box(a); }
      |}
      |""".stripMargin

  private lazy val loaded = JavaInMemory.compile("Alloc", Source)
  private val Box         = p.Type.Struct(p.Sym("Alloc.Box"), Nil)
  private val BoxRef      = p.Type.Ptr(Box, p.Type.Space.Private)

  private def descOf(name: String): String =
    Type.getMethodDescriptor(loaded.loadClass("Alloc").getDeclaredMethods.find(_.getName == name).get)

  private def lowered(klass: String, name: String, desc: String): Result[BytecodeRemapper.Lowered] =
    BytecodeRemapper.lowerWithCalls(loaded.bytesOf(klass), name, desc, loaded.classes.get)

  private def lowered(name: String): BytecodeRemapper.Lowered =
    lowered("Alloc", name, descOf(name)).fold(e => fail(s"`$name` was refused: ${e.getMessage}"), identity)

  private def refusal(name: String): String =
    lowered("Alloc", name, descOf(name)) match {
      case Left(e)   => e.getMessage
      case Right(fn) => fail(s"`$name` was expected to be refused but lowered to:\n${fn.fn.repr}")
    }

  private def allocs(fn: p.Function): List[p.Expr.Alloc] = fn.collectWhere[p.Expr] { case a: p.Expr.Alloc => a }

  test("an array allocation is a global heap allocation of the component") {
    val found = allocs(lowered("scratch").fn)
    assertEquals(found.map(_.comp), List(p.Type.IntS32))
    assertEquals(found.map(_.space), List(p.Type.Space.Global))
    assertEquals(found.map(_.region), List(p.Region.Opaque))
  }

  test("an array initialiser sizes the allocation and stores each element") {
    val fn = lowered("literal").fn
    assertEquals(allocs(fn).map(_.size), List(p.Term.IntS32Const(3)))
    val stores = fn.collectWhere[p.Stmt] { case u: p.Stmt.Update => u }
    assertEquals(stores.size, 3)
  }

  test("an object is a stack struct built by calling its constructor on a reference") {
    val l      = lowered("objects")
    val locals = l.fn.collectWhere[p.Stmt] { case p.Stmt.Var(n, None, _) if n.tpe == Box => n }
    assertEquals(locals.size, 1)
    assertEquals(l.fn.collectWhere[p.Expr] { case r: p.Expr.RefTo => r.tpe }, List(BoxRef))
    assertEquals(l.calls, List(MethodRef("Alloc$Box", "<init>", "(I)V")))
    assertEquals(l.structs.map(_.name), List(p.Sym("Alloc.Box")))
  }

  test("a field of a new object starts at zero") {
    val zeroed = lowered("delegating").fn.collectWhere[p.Stmt] {
      case p.Stmt.Mut(lhs, p.Expr.Alias(p.Term.IntS32Const(0))) if lhs.steps == List(p.PathStep.Field("v")) => lhs
    }
    assertEquals(zeroed.size, 1)
  }

  test("a constructor takes its object first and stores through it") {
    val l = lowered("Alloc$Box", "<init>", "(I)V").fold(e => fail(e.getMessage), identity)
    assertEquals(l.fn.name, p.Sym("Alloc$Box.<init>"))
    assertEquals(l.fn.args.map(_.named.tpe), List(BoxRef, p.Type.IntS32))
    assertEquals(l.fn.rtn, p.Type.Unit0)
    assertEquals(l.calls, Nil)
    val stores = l.fn.collectWhere[p.Stmt] {
      case p.Stmt.Mut(lhs, _) if lhs.steps == List(p.PathStep.Deref, p.PathStep.Field("v")) => lhs
    }
    assertEquals(stores.size, 1)
  }

  test("a delegating constructor calls the constructor it delegates to") {
    val l = lowered("Alloc$Box", "<init>", "()V").fold(e => fail(e.getMessage), identity)
    assertEquals(l.calls, List(MethodRef("Alloc$Box", "<init>", "(I)V")))
  }

  test("the resolver lowers a constructor once and returns the struct once") {
    val root = MethodRef("Alloc", "delegating", descOf("delegating"))
    BytecodeResolver.resolve(List(root), loaded.classes.get) match {
      case Left(e) => fail(e.getMessage)
      case Right(r) =>
        assertEquals(
          r.functions.map(_.name.repr).sorted,
          List("Alloc$Box.<init>", "Alloc$Box.<init>", "Alloc.delegating").sorted
        )
        assertEquals(r.structs.map(_.name), List(p.Sym("Alloc.Box")))
    }
  }

  test("an object whose class has a superclass is refused") {
    val message = refusal("derived")
    assert(message.contains("Unsupported allocation of `Alloc$Derived`"), message)
    assert(message.contains("superclass"), message)
  }

  test("an array of objects is refused") {
    assert(refusal("boxes").contains("Alloc$Box"))
  }

  test("a JDK class is refused") {
    assert(refusal("platform").contains("java.util.ArrayList"))
  }

  test("an object cannot be returned") {
    assert(refusal("leak").contains("cannot be returned"))
  }
}
