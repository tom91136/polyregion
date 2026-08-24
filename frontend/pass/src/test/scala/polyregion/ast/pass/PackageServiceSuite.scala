package polyregion.ast.pass

import polyregion.ast.{InterfaceBinding, MsgPack, PackageLinker, PackageSymResolver, PolyAST as p, given}
import polyregion.ast.Traversal.*
import polyregion.ast.generated.PolyPackageWireSchema

class PackageServiceSuite extends munit.FunSuite {

  private val unitReturn = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))

  private def function(
      decl: p.FunctionDecl,
      body: List[p.Stmt] = unitReturn,
      implements: Option[p.Sym] = None,
      capabilities: List[String] = Nil,
      visibility: p.Function.Visibility = p.Function.Visibility.Internal
  ) = p.Function(
    decl,
    body,
    visibility,
    p.Function.FpMode.Relaxed,
    p.CallConvention.RegularCall,
    implements,
    capabilities
  )

  test("linking composes a context-aware implementation ABI") {
    val name    = p.Sym("library.copy")
    val pointer = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(
        p.Arg(
          p.Named("values", pointer),
          boundary = Some(
            p.Arg.Boundary(p.Arg.Access.ReadWrite, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1)))
          )
        ),
        p.Arg(p.Named("n", p.Type.IntS32))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val harvested = public.copy(
      name = p.Sym("implementation.copy"),
      args = p.Arg(p.Named("#context", p.Spec.ContextType)) :: public.args.map(_.copy(boundary = None))
    )
    val request = p.Package.LinkRequest(
      p.Interface(p.Sym("library"), List(public)),
      List(p.Program(None, List(function(harvested, implements = Some(name), capabilities = List("gpu"))), Nil)),
      List("gpu")
    )

    val linked = PackageLinker.link(request)
    assert(linked.isRight, linked)
    val implementation = linked.toOption.get.program.functions.head
    assertEquals(implementation.visibility, p.Function.Visibility.Exported)
    assertEquals(
      implementation.args(1).boundary,
      Some(p.Arg.Boundary(p.Arg.Access.ReadWrite, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
    )
    assertEquals(
      PackageSymResolver.bindImplementation(implementation.decl, public).map(_.systemArguments),
      Right(1)
    )
  }

  test("linking filters capabilities and isolates conflicting fragment-local helpers") {
    val publicName = p.Sym("library.apply")
    val public = p.FunctionDecl(
      publicName,
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val helperName = p.Sym("vendor.helper")
    def helper(value: Int) = {
      val decl = public.copy(name = helperName)
      function(
        decl,
        List(
          p.Stmt.Var(p.Named("value", p.Type.IntS32), Some(p.Expr.Alias(p.Term.IntS32Const(value))), false),
          unitReturn.head
        )
      )
    }
    val implementationDecl = public.copy(name = p.Sym("implementation.apply"))
    val implementation = function(
      implementationDecl,
      List(
        p.Stmt.Var(
          p.Named("call", p.Type.Unit0),
          Some(p.Expr.Invoke(p.Type.FnRef(helperName), Nil, None, Nil, p.Type.Unit0)),
          false
        ),
        unitReturn.head
      ),
      Some(publicName),
      List("gpu")
    )
    val interface = p.Interface(p.Sym("library"), List(public))
    val missing = PackageLinker.link(
      p.Package.LinkRequest(interface, List(p.Program(None, List(implementation, helper(1)), Nil)), List("cpu"))
    )
    assert(missing.left.exists(_.exists(_.contains("no compatible implementation"))))

    val linked = PackageLinker.link(
      p.Package.LinkRequest(
        interface,
        List(
          p.Program(None, List(implementation, helper(1)), Nil),
          p.Program(None, List(helper(2)), Nil)
        ),
        List("gpu")
      )
    )
    assert(linked.isRight, linked)
    val helperNames = linked.toOption.get.program.functions.map(_.name).filter(_.fqn.contains("helper"))
    assertEquals(
      helperNames.toSet,
      Set(p.Sym(List("#fragment", "0", "vendor", "helper")), p.Sym(List("#fragment", "1", "vendor", "helper")))
    )
  }

  test("Sym resolution preserves context and a direct result") {
    val name = p.Sym("library.increment")
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val implementationDecl = public.copy(
      name = p.Sym("implementation.increment"),
      args = p.Arg(p.Named("#context", p.Spec.ContextType)) :: public.args
    )
    val value = implementationDecl.args(1).named
    val implementation = function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(value, Nil, value.tpe)))),
      Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))
    val resolved = PackageSymResolver.resolveSym(
      p.Package.SymRequest(
        pkg,
        p.InvokeSignature(name, Nil, None, List(p.Type.IntS32), p.Type.IntS32),
        Nil,
        Nil,
        Nil,
        Nil,
        List(p.Package.TypeSize(p.Type.IntS32, 4)),
        "#entry"
      )
    )
    assert(resolved.isRight, resolved)
    val resolvedProgram = resolved.toOption.get
    assertEquals(
      resolvedProgram.entryArgs,
      List(
        p.Package.EntryArgBinding.Context,
        p.Package.EntryArgBinding.CallAddress(0),
        p.Package.EntryArgBinding.ResultAddress
      )
    )
    val entry = resolvedProgram.program.entry.get
    assertEquals(entry.args.map(_.named.symbol), List("#context", "a0", "result"))
    val invoke = entry.collectAll[p.Expr].collectFirst { case value: p.Expr.Invoke => value }.get
    assertEquals(invoke.args.head, p.Term.Select(entry.args.head.named, Nil, p.Spec.ContextType))
  }

  test("Sym resolution maps an erased result to its source call argument") {
    val name = p.Sym("library.increment")
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val implementation = function(
      public.copy(
        name = p.Sym("implementation.increment"),
        args = p.Arg(p.Named("#context", p.Spec.ContextType)) :: public.args
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = PackageSymResolver.resolveSym(
      p.Package.SymRequest(
        p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil)),
        p.InvokeSignature(name, Nil, None, List(p.Type.IntS32), p.Type.Nothing),
        Nil,
        Nil,
        Nil,
        Nil,
        List(p.Package.TypeSize(p.Type.IntS32, 4)),
        "#entry",
        p.Package.ReturnConvention.OutParam(1)
      )
    )
    assert(resolved.isRight, resolved)
    assertEquals(
      resolved.toOption.get.entryArgs,
      List(
        p.Package.EntryArgBinding.Context,
        p.Package.EntryArgBinding.CallAddress(0),
        p.Package.EntryArgBinding.CallValue(1)
      )
    )

    val leadingOutParam = PackageSymResolver.resolveSym(
      p.Package.SymRequest(
        p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil)),
        p.InvokeSignature(name, Nil, None, List(p.Type.IntS32), p.Type.Nothing),
        Nil,
        Nil,
        Nil,
        Nil,
        List(p.Package.TypeSize(p.Type.IntS32, 4)),
        "#entry-leading",
        p.Package.ReturnConvention.OutParam(0)
      )
    )
    assertEquals(
      leadingOutParam.map(_.entryArgs),
      Right(
        List(
          p.Package.EntryArgBinding.Context,
          p.Package.EntryArgBinding.CallAddress(1),
          p.Package.EntryArgBinding.CallValue(0)
        )
      )
    )

    List(-1, 2).foreach { index =>
      val invalid = PackageSymResolver.resolveSym(
        p.Package.SymRequest(
          p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil)),
          p.InvokeSignature(name, Nil, None, List(p.Type.IntS32), p.Type.Nothing),
          Nil,
          Nil,
          Nil,
          Nil,
          List(p.Package.TypeSize(p.Type.IntS32, 4)),
          "#entry-invalid",
          p.Package.ReturnConvention.OutParam(index)
        )
      )
      assert(invalid.left.exists(_.exists(_.contains("outside source argument range"))), invalid)
    }
  }

  test("Sym resolution rejects invalid or conflicting type layouts") {
    val name = p.Sym("library.noop")
    val declaration = p.FunctionDecl(
      name,
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementation = function(
      declaration.copy(name = p.Sym("implementation.noop")),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(declaration)), p.Program(None, List(implementation), Nil))
    def request(layouts: List[p.Package.TypeSize]) = p.Package.SymRequest(
      pkg,
      p.InvokeSignature(name, Nil, None, Nil, p.Type.Unit0),
      Nil,
      Nil,
      Nil,
      Nil,
      layouts,
      "#entry"
    )

    assert(
      PackageSymResolver
        .resolveSym(request(List(p.Package.TypeSize(p.Type.IntS32, 4), p.Package.TypeSize(p.Type.IntS32, 8))))
        .left
        .exists(_.exists(_.contains("conflicts")))
    )
    assert(
      PackageSymResolver
        .resolveSym(request(List(p.Package.TypeSize(p.Type.IntS32, 0))))
        .left
        .exists(_.exists(_.contains("must be positive")))
    )
  }

  test("package linking and Sym resolution preserve a trailing-output result") {
    val name = p.Sym("library.increment")
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val resultType = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val harvested = public.copy(
      name = p.Sym("implementation.increment"),
      args = public.args :+ p.Arg(p.Named("out", resultType), None),
      rtn = p.Type.Unit0
    )
    val implementation = function(harvested, implements = Some(name))
    val linked = PackageLinker.link(
      p.Package.LinkRequest(
        p.Interface(p.Sym("library"), List(public)),
        List(p.Program(None, List(implementation), Nil))
      )
    )
    assert(linked.isRight, linked)
    val composed = linked.toOption.get.program.functions.head
    assertEquals(
      composed.args.last.boundary,
      Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))))
    )

    val resolved = PackageSymResolver.resolveSym(
      p.Package.SymRequest(
        linked.toOption.get,
        p.InvokeSignature(name, Nil, None, List(p.Type.IntS32), p.Type.IntS32),
        Nil,
        Nil,
        Nil,
        Nil,
        List(p.Package.TypeSize(p.Type.IntS32, 4)),
        "#entry"
      )
    )
    assert(resolved.isRight, resolved)
    val resolvedProgram = resolved.toOption.get
    assertEquals(
      resolvedProgram.entryArgs,
      List(
        p.Package.EntryArgBinding.Context,
        p.Package.EntryArgBinding.CallAddress(0),
        p.Package.EntryArgBinding.ResultAddress
      )
    )
    val entry  = resolvedProgram.program.entry.get
    val invoke = entry.collectAll[p.Expr].collectFirst { case value: p.Expr.Invoke => value }.get
    assertEquals(invoke.rtn, p.Type.Unit0)
    assertEquals(invoke.args.last, p.Term.Select(entry.args.last.named, Nil, resultType))
  }

  test("Sym resolution closes and substitutes a source callable") {
    val name              = p.Sym("library.apply")
    val callable          = p.Sym("caller.increment")
    val element: p.Type   = p.Type.Var("Element")
    val operation: p.Type = p.Type.Var("Operation")
    val exec              = p.Type.Exec(Nil, List(p.Type.IntS32), p.Type.IntS32)
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32)), p.Arg(p.Named("operation", exec))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val implementationDecl = p.FunctionDecl(
      p.Sym("implementation.apply"),
      List(p.Type.Var("Element"), p.Type.Var("Operation")),
      None,
      List(p.Arg(p.Named("value", element)), p.Arg(p.Named("operation", operation))),
      Nil,
      Nil,
      element,
      p.Function.Affinity.Host
    )
    val implementation = function(
      implementationDecl,
      List(
        p.Stmt.Return(
          p.Expr.Invoke(
            operation,
            Nil,
            None,
            List(p.Term.Select(implementationDecl.args.head.named, Nil, element)),
            element
          )
        )
      ),
      Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val callableDecl = p.FunctionDecl(
      callable,
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val callableFunction = function(
      callableDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(callableDecl.args.head.named, Nil, p.Type.IntS32))))
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))
    val resolved = PackageSymResolver.resolveSym(
      p.Package.SymRequest(
        pkg,
        p.InvokeSignature(name, Nil, None, List(p.Type.IntS32, p.Type.FnRef(callable)), p.Type.IntS32),
        List(callableDecl),
        List(callableFunction),
        Nil,
        Nil,
        List(p.Package.TypeSize(p.Type.IntS32, 4)),
        "#entry"
      )
    )
    assert(resolved.isRight, resolved)
    val resolvedProgram = resolved.toOption.get
    assertEquals(
      resolvedProgram.entryArgs,
      List(
        p.Package.EntryArgBinding.Context,
        p.Package.EntryArgBinding.CallAddress(0),
        p.Package.EntryArgBinding.ResultAddress
      )
    )
    assert(resolvedProgram.program.functions.exists(_.name == callable))
    val selected = resolvedProgram.program.functions.find(_.name != callable).get
    assertEquals(selected.args.map(_.named.symbol), List("value"))
    assert(selected.collectAll[p.Type].contains(p.Type.FnRef(callable)))
  }

  test("callable return types participate in public type inference") {
    val name     = p.Sym("library.make")
    val callable = p.Sym("caller.makeInt")
    val element  = p.Type.Var("Element")
    val public = p.FunctionDecl(
      name,
      List(p.Type.Var("Element")),
      None,
      List(p.Arg(p.Named("operation", p.Type.Exec(Nil, Nil, element)))),
      Nil,
      Nil,
      element,
      p.Function.Affinity.Host
    )
    val callableDecl = p.FunctionDecl(
      callable,
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val bound = PackageSymResolver.bindCall(
      public,
      p.InvokeSignature(name, Nil, None, List(p.Type.FnRef(callable)), p.Type.IntS32),
      List(callableDecl)
    )
    assertEquals(bound.map(_.types), Right(Map("Element" -> p.Type.IntS32)))
  }

  test("package-service envelopes have an independent explicit fingerprint") {
    val request = p.Package.LinkRequest(p.Interface(p.Sym("library"), Nil), Nil)
    val encoded = MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request))
    assertEquals(
      MsgPack.decode[MsgPack.Versioned[p.Package.LinkRequest]](encoded),
      Right(MsgPack.Versioned(PolyPackageWireSchema.Hash, request))
    )
    assertNotEquals(PolyPackageWireSchema.Hash, "8457f51aea3fd94550eb5bbf794b980d")
    assertNotEquals(PolyPackageWireSchema.Hash, "c857f2efd9fc578eb2f6ceac870f43d8")
  }
}
