package polyregion.scalalang

import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import polyregion.ast.PolyAST as p
import polyregion.ast.Traversal.*
import polyregion.ast.{*, given}

import scala.annotation.compileTimeOnly

object FooApi {
  @compileTimeOnly("polyregion_interface:foo:bar.increment")
  def increment(x: Int): Int = throw IllegalStateException("compiler did not replace bar.increment")
}

object LookalikeApi {
  @compileTimeOnly("generated for another purpose")
  def increment(x: Int): Int = x
}

class InterfaceResolverSuite extends munit.FunSuite {

  override def afterEach(context: AfterEach): Unit =
    System.clearProperty("polyregion.library.capabilities")

  test("read a generated interface identity") {
    assertEquals(InterfaceTestMacros.interfaceIdentityOf[FooApi.type], Some("foo" -> "bar.increment"))
  }

  test("ignore compile-time-only annotations without an interface marker") {
    assertEquals(InterfaceTestMacros.interfaceIdentityOf[LookalikeApi.type], None)
  }

  test("read configured library capabilities") {
    System.setProperty("polyregion.library.capabilities", " host,fast,,host ")
    assertEquals(InterfaceResolver.configuredCapabilities, Set("host", "fast"))
  }

  test("resolve, link and execute a scalar interface call") {
    val i32                = p.Type.IntS32
    val publicName         = p.Sym("bar.increment")
    val implementationName = p.Sym("bar.implementation.increment")
    val publicDecl = p.FunctionDecl(
      publicName,
      Nil,
      None,
      List(p.Arg(p.Named("x", i32), None)),
      Nil,
      Nil,
      i32,
      p.Function.Affinity.Host
    )
    val implementationDecl = publicDecl.copy(name = implementationName)
    val x                  = p.Term.Select(p.Named("x", i32), Nil, i32)
    val implementation = p.Function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.IntrOp(p.Intr.Add(x, p.Term.IntS32Const(1), i32)))),
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      false
    )
    val entry = implementation.copy(
      decl = implementationDecl.copy(name = p.Sym("package.entry"), args = Nil, rtn = p.Type.Unit0),
      body = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      isEntry = true
    )
    val pack = InterfaceResolver.LibraryPackage(
      p.PackageIndex(
        p.InterfaceDef(p.Sym("foo"), List(publicDecl)),
        List(p.ImplementationCandidate(publicName, implementationDecl, Nil, Nil))
      ),
      p.Program(entry, List(implementation), Nil)
    )
    val targetName = p.Sym("foo.bindings.increment")
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(targetName),
      Nil,
      None,
      List(p.Term.IntS32Const(41)),
      i32
    )

    val (functions, defs) =
      InterfaceResolver.link(pack, "bar.increment", target).fold(errors => fail(errors.mkString("\n")), identity)
    val linked = functions.find(_.name == targetName).getOrElse(fail("missing linked function"))
    val program = p.Program(
      linked.copy(isEntry = true),
      functions.filterNot(_ == linked),
      defs.toList,
      p.PassPhase.Initial,
      Nil
    )

    assertEquals(Interpreter.Vm(program).call(targetName, List(i32 -> V.I(41))), V.I(42))
  }

  test("link an inferred callable interface argument") {
    val i32       = p.Type.IntS32
    val element   = p.Type.Var("Element")
    val operation = p.Type.Exec(Nil, List(p.Type.Var("T")), p.Type.Var("T"))
    val publicDecl = p.FunctionDecl(
      p.Sym("bar.apply"),
      List("T"),
      None,
      List(p.Arg(p.Named("x", p.Type.Var("T")), None), p.Arg(p.Named("op", operation), None)),
      Nil,
      Nil,
      p.Type.Var("T"),
      p.Function.Affinity.Host
    )
    val implementationDecl = p.FunctionDecl(
      p.Sym("bar.implementation.apply"),
      List("Element", "Op"),
      None,
      List(p.Arg(p.Named("x", element), None), p.Arg(p.Named("op", p.Type.Var("Op")), None)),
      Nil,
      Nil,
      element,
      p.Function.Affinity.Host
    )
    val x = p.Term.Select(p.Named("x", element), Nil, element)
    val implementation = p.Function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.Invoke(p.Type.Var("Op"), Nil, None, List(x), element))),
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      false
    )
    val callableName = p.Sym("foo.callable.increment")
    val callableDecl = p.FunctionDecl(
      callableName,
      Nil,
      None,
      List(p.Arg(p.Named("x", i32), None)),
      Nil,
      Nil,
      i32,
      p.Function.Affinity.Host
    )
    val pack = InterfaceResolver.LibraryPackage(
      p.PackageIndex(
        p.InterfaceDef(p.Sym("foo"), List(publicDecl)),
        List(p.ImplementationCandidate(publicDecl.name, implementationDecl, Nil, Nil))
      ),
      p.Program(implementation.copy(isEntry = true), List(implementation), Nil)
    )
    val callable = p.Term.Select(p.Named("op", p.Type.FnRef(callableName)), Nil, p.Type.FnRef(callableName))
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("foo.bindings.apply")),
      Nil,
      None,
      List(p.Term.IntS32Const(41), callable),
      i32
    )

    val (functions, _) = InterfaceResolver
      .link(pack, "bar.apply", target, List(callableDecl))
      .fold(errors => fail(errors.mkString("\n")), identity)
    val linked = functions.find(_.name == target.calleeName).getOrElse(fail("missing linked function"))

    assertEquals(linked.args.map(_.named.tpe), List(i32, p.Type.FnRef(callableName)))
    assert(linked.collectAll[p.Expr].exists {
      case p.Expr.Invoke(p.Type.FnRef(`callableName`), Nil, None, List(_), `i32`) => true
      case _                                                                      => false
    })
  }

  test("reject explicit generic interface calls until call sites can be specialized") {
    val i32 = p.Type.IntS32
    val tpe = p.Type.Var("T")
    val publicDecl = p.FunctionDecl(
      p.Sym("bar.identity"),
      List("T"),
      None,
      List(p.Arg(p.Named("x", tpe), None)),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val implementation = p.Function(
      publicDecl.copy(name = p.Sym("bar.implementation.identity")),
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(p.Named("x", tpe), Nil, tpe)))),
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      true
    )
    val pack = InterfaceResolver.LibraryPackage(
      p.PackageIndex(
        p.InterfaceDef(p.Sym("foo"), List(publicDecl)),
        List(p.ImplementationCandidate(publicDecl.name, implementation.decl, Nil, Nil))
      ),
      p.Program(implementation, List(implementation), Nil)
    )
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("foo.bindings.identity")),
      List(i32),
      None,
      List(p.Term.IntS32Const(41)),
      i32
    )

    assertEquals(
      InterfaceResolver.link(pack, "bar.identity", target).left.toOption,
      Some(List("generic Scala interface calls are not yet supported"))
    )
  }
}
