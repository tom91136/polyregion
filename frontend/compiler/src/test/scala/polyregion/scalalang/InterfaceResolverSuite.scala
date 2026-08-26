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

  private def importOne(
      pkg: p.Package,
      declaration: String,
      target: p.Expr.Invoke,
      callerDecls: List[p.FunctionDecl] = Nil
  ) =
    InterfaceResolver.importPackages(
      List(pkg),
      List(InterfaceResolver.Import(pkg.interface.name.fqcn, declaration, target)),
      callerDecls,
      InterfaceResolver.configuredCapabilities,
      Map.empty
    )

  override def afterEach(context: AfterEach): Unit = {
    System.clearProperty("polyregion.library.capabilities")
    System.clearProperty("polyregion.library.path")
  }

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

  test("reject a package identity that escapes its configured root") {
    System.setProperty("polyregion.library.path", System.getProperty("java.io.tmpdir"))
    assertEquals(
      InterfaceResolver.loadPackage("../outside").left.toOption,
      Some(List("invalid package identity `../outside`"))
    )
  }

  test("import and execute multiple scalar interface calls in one link") {
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
      p.CallConvention.RegularCall
    )
    val pkg = p.Package(
      p.Interface(p.Sym("foo"), List(publicDecl)),
      p.Program(None, List(implementation.copy(implements = Some(publicName))), Nil)
    )
    val targetName = p.Sym("foo.resolved.increment")
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(targetName),
      Nil,
      None,
      List(p.Term.IntS32Const(41)),
      i32
    )
    val secondName = p.Sym("foo.resolved.incrementAgain")
    val second = target.copy(
      callee = p.Type.FnRef(secondName),
      args = List(p.Term.IntS32Const(42))
    )

    val (functions, defs) = InterfaceResolver
      .importPackages(
        List(pkg),
        List(
          InterfaceResolver.Import("foo", "bar.increment", target),
          InterfaceResolver.Import("foo", "bar.increment", second)
        ),
        Nil,
        Set.empty,
        Map.empty
      )
      .fold(errors => fail(errors.mkString("\n")), identity)
    val resolved = functions.find(_.name == targetName).getOrElse(fail("missing resolved function"))
    val program = p.Program(
      Some(resolved.copy(convention = p.CallConvention.OffloadEntry)),
      functions.filterNot(_ == resolved),
      defs.toList,
      p.Pass.Phase.Initial,
      Nil
    )

    val vm = Interpreter.Vm(program)
    assertEquals(vm.call(targetName, List(i32 -> V.I(41))), V.I(42))
    assertEquals(vm.call(secondName, List(i32 -> V.I(42))), V.I(43))
  }

  test("link an inferred callable interface argument") {
    val i32       = p.Type.IntS32
    val element   = p.Type.Var("Element")
    val operation = p.Type.Exec(Nil, List(p.Type.Var("T")), p.Type.Var("T"))
    val publicDecl = p.FunctionDecl(
      p.Sym("bar.apply"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("x", p.Type.Var("T")), None), p.Arg(p.Named("op", operation), None)),
      Nil,
      Nil,
      p.Type.Var("T"),
      p.Function.Affinity.Host
    )
    val implementationDecl = p.FunctionDecl(
      p.Sym("bar.implementation.apply"),
      List(p.Type.Var("Element"), p.Type.Var("Op")),
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
      p.CallConvention.RegularCall
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
    val pkg = p.Package(
      p.Interface(p.Sym("foo"), List(publicDecl)),
      p.Program(None, List(implementation.copy(implements = Some(publicDecl.name))), Nil)
    )
    val callable = p.Term.Select(p.Named("op", p.Type.FnRef(callableName)), Nil, p.Type.FnRef(callableName))
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("foo.resolved.apply")),
      Nil,
      None,
      List(p.Term.IntS32Const(41), callable),
      i32
    )

    val (functions, _) = importOne(pkg, "bar.apply", target, List(callableDecl))
      .fold(errors => fail(errors.mkString("\n")), identity)
    val resolved = functions.find(_.name == target.calleeName).getOrElse(fail("missing resolved function"))

    assertEquals(resolved.args.map(_.named.tpe), List(i32, p.Type.FnRef(callableName)))
    assert(resolved.collectAll[p.Expr].exists {
      case p.Expr.Invoke(p.Type.FnRef(`callableName`), Nil, None, List(_), `i32`) => true
      case _                                                                      => false
    })
  }

  test("link an exact pointer-element width specialisation") {
    val element    = p.Type.Var("Element", Some(4))
    val publicName = p.Sym("bar.copy")
    val publicDecl = p.FunctionDecl(
      publicName,
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("in", p.Type.Ptr(p.Type.Var("T"), p.Type.Space.Global)), None)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementationDecl = publicDecl.copy(
      name = p.Sym("bar.implementation.copy_w4"),
      tpeVars = List(p.Type.Var("Element", Some(4))),
      args = List(p.Arg(p.Named("in", p.Type.Ptr(element, p.Type.Space.Global)), None))
    )
    val implementation = p.Function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.OffloadEntry
    )
    val pkg = p.Package(
      p.Interface(p.Sym("foo"), List(publicDecl)),
      p.Program(None, List(implementation.copy(implements = Some(publicName))), Nil)
    )
    val pointer = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("foo.resolved.copy")),
      Nil,
      None,
      List(p.Term.Select(p.Named("in", pointer), Nil, pointer)),
      p.Type.Unit0
    )

    val resolved = importOne(pkg, "bar.copy", target)
    assert(resolved.isRight, resolved.left.toOption.fold("")(_.mkString("\n")))
  }

  test("link exact pointer-width specialisations") {
    val publicVar = p.Type.Var("T")
    val pointerWidth =
      System.getProperty("sun.arch.data.model", "64").toIntOption.filter(_ > 0).getOrElse(64) / 8
    val implementationVar = p.Type.Var("Element", Some(pointerWidth))

    def assertLinked(publicArg: p.Type, implementationArg: p.Type, targetArg: p.Type, name: String): Unit = {
      val publicName = p.Sym(s"bar.$name")
      val publicDecl = p.FunctionDecl(
        publicName,
        List(p.Type.Var("T")),
        None,
        List(p.Arg(p.Named("value", publicArg), None)),
        Nil,
        Nil,
        p.Type.Unit0,
        p.Function.Affinity.Host
      )
      val implementationDecl = publicDecl.copy(
        name = p.Sym(s"bar.implementation.$name"),
        tpeVars = List(p.Type.Var("Element", Some(pointerWidth))),
        args = List(p.Arg(p.Named("value", implementationArg), None))
      )
      val implementation = p.Function(
        implementationDecl,
        List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
        p.Function.Visibility.Exported,
        p.Function.FpMode.Relaxed,
        p.CallConvention.OffloadEntry
      )
      val pkg = p.Package(
        p.Interface(p.Sym("foo"), List(publicDecl)),
        p.Program(None, List(implementation.copy(implements = Some(publicName))), Nil)
      )
      val target: p.Expr.Invoke = p.Expr.Invoke(
        p.Type.FnRef(p.Sym(s"foo.resolved.$name")),
        Nil,
        None,
        List(p.Term.Select(p.Named("value", targetArg), Nil, targetArg)),
        p.Type.Unit0
      )

      val resolved = importOne(pkg, publicName.fqcn, target)
      assert(resolved.isRight, resolved.left.toOption.fold("")(_.mkString("\n")))
    }

    val pointer       = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val nestedPointer = p.Type.Ptr(pointer, p.Type.Space.Global)
    assertLinked(publicVar, implementationVar, pointer, "pointer_value")
    assertLinked(
      p.Type.Ptr(publicVar, p.Type.Space.Global),
      p.Type.Ptr(implementationVar, p.Type.Space.Global),
      nestedPointer,
      "nested_pointer"
    )
  }

  test("link explicit generic interface calls through inferred concrete arguments") {
    val i32 = p.Type.IntS32
    val tpe = p.Type.Var("T")
    val publicDecl = p.FunctionDecl(
      p.Sym("bar.identity"),
      List(p.Type.Var("T")),
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
      p.CallConvention.OffloadEntry
    )
    val pkg = p.Package(
      p.Interface(p.Sym("foo"), List(publicDecl)),
      p.Program(None, List(implementation.copy(implements = Some(publicDecl.name))), Nil)
    )
    val target: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("foo.resolved.identity")),
      List(i32),
      None,
      List(p.Term.IntS32Const(41)),
      i32
    )

    val resolved = importOne(pkg, "bar.identity", target)
    assert(resolved.isRight, resolved.left.toOption.fold("")(_.mkString("\n")))
    assertEquals(
      resolved.toOption.toList.flatMap(_._1).find(_.name == target.calleeName).map(_.args.map(_.named.tpe)),
      Some(List(i32))
    )
  }
}
