package polyregion.ast.pass

import polyregion.ast.{MsgPack, PackageLinker, PolyAST as p, ProgramLinker, given}
import polyregion.ast.Traversal.*
import polyregion.ast.generated.PolyPackageWireSchema

class PackageServiceSuite extends munit.FunSuite {

  private val unitReturn = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))

  private def fragment(identity: String, functions: List[p.Function]): p.Package.Fragment =
    p.Package.Fragment(identity, p.Program(None, functions, Nil))

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

  test("program linking retains transitively specialised remote-launch targets") {
    val publicName = p.Sym("library.launch")
    val publicDecl = p.FunctionDecl(
      publicName,
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementationName  = p.Sym("implementation.launch")
    val firstKernelName     = p.Sym("kernel.first")
    val secondKernelName    = p.Sym("kernel.second")
    val element: p.Type.Var = p.Type.Var("T")
    val context             = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val one                 = p.Term.IntU32Const(1)
    def launch(name: p.Sym, tpeArgs: List[p.Type]) = p.Expr.SpecOp(
      p.Spec.RemoteLaunch(
        context,
        p.Term.Poison(p.Type.FnRef(name)),
        tpeArgs,
        one,
        one,
        one,
        one,
        one,
        one,
        p.Term.IntU32Const(0),
        Nil
      )
    )
    def launcher(decl: p.FunctionDecl, target: p.Sym, tpeArgs: List[p.Type]) = function(
      decl,
      List(
        p.Stmt.Var(p.Named("launch", p.Type.Unit0), Some(launch(target, tpeArgs)), isMutable = false),
        unitReturn.head
      )
    )
    val implementation =
      launcher(publicDecl.copy(name = implementationName), firstKernelName, List(p.Type.IntS32)).copy(
        visibility = p.Function.Visibility.Exported,
        implements = Some(publicName)
      )
    val firstKernel = launcher(
      publicDecl.copy(name = firstKernelName, tpeVars = List(element), affinity = p.Function.Affinity.Offload),
      secondKernelName,
      List(element)
    ).copy(convention = p.CallConvention.OffloadEntry)
    val secondKernel = function(
      publicDecl.copy(name = secondKernelName, tpeVars = List(element), affinity = p.Function.Affinity.Offload)
    ).copy(convention = p.CallConvention.OffloadEntry)
    val pkg = p.Package(
      p.Interface(p.Sym("library"), List(publicDecl)),
      p.Program(None, List(implementation, firstKernel, secondKernel), Nil)
    )
    val root = function(
      publicDecl.copy(name = p.Sym("consumer.launch")),
      implements = Some(publicName),
      visibility = p.Function.Visibility.Exported
    )

    val linked = ProgramLinker.link(p.Program.LinkRequest(List(pkg), p.Program(None, List(root), Nil)))

    assert(linked.isRight, linked)
    val program   = linked.toOption.get
    val functions = program.entry.toList ::: program.functions
    val targets = functions
      .flatMap(_.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe })
      .collect { case p.Type.FnRef(name) => name }
    val expectedTargets = Set(
      Specialisation.monomorphicName(firstKernelName, List(p.Type.IntS32)),
      Specialisation.monomorphicName(secondKernelName, List(p.Type.IntS32))
    )
    assertEquals(targets.toSet, expectedTargets)
    targets.foreach(target => assertEquals(functions.count(_.name == target), 1))
  }

  test("host affinity follows calls rather than stored function handles") {
    val publicName = p.Sym("library.handle")
    val publicDecl = p.FunctionDecl(
      publicName,
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val deviceName = p.Sym("kernel.handle")
    val implementation = function(
      publicDecl.copy(name = p.Sym("implementation.handle")),
      List(
        p.Stmt.Var(
          p.Named("handle", p.Type.FnRef(deviceName)),
          Some(p.Expr.Alias(p.Term.Poison(p.Type.FnRef(deviceName)))),
          isMutable = false
        ),
        unitReturn.head
      ),
      implements = Some(publicName),
      visibility = p.Function.Visibility.Exported
    )
    val device = function(publicDecl.copy(name = deviceName, affinity = p.Function.Affinity.Offload))
      .copy(convention = p.CallConvention.OffloadEntry)
    val pkg =
      p.Package(p.Interface(p.Sym("library"), List(publicDecl)), p.Program(None, List(implementation, device), Nil))
    val root = function(
      publicDecl.copy(name = p.Sym("consumer.handle")),
      implements = Some(publicName),
      visibility = p.Function.Visibility.Exported
    )

    val linked = ProgramLinker.link(p.Program.LinkRequest(List(pkg), p.Program(None, List(root), Nil)))

    assert(linked.isRight, linked)
    val retained =
      linked.toOption.get.functions.find(_.name == deviceName).getOrElse(fail("missing stored device handle"))
    assertEquals(retained.affinity, p.Function.Affinity.Offload)
  }

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
      List(fragment("copy", List(function(harvested, implements = Some(name), capabilities = List("gpu"))))),
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
      ProgramLinker.matchImplementation(implementation.decl, public).map(_.systemArguments),
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
      p.Package.LinkRequest(interface, List(fragment("first", List(implementation, helper(1)))), List("cpu"))
    )
    assert(missing.left.exists(_.exists(_.contains("no compatible implementation"))))

    val linked = PackageLinker.link(
      p.Package.LinkRequest(
        interface,
        List(
          fragment("first", List(implementation, helper(1))),
          fragment("second", List(helper(2)))
        ),
        List("gpu")
      )
    )
    assert(linked.isRight, linked)
    val helperNames = linked.toOption.get.program.functions.map(_.name).filter(_.fqn.contains("helper"))
    assertEquals(
      helperNames.toSet,
      Set(
        p.Sym(List("#fragment", "first", "vendor", "helper")),
        p.Sym(List("#fragment", "second", "vendor", "helper"))
      )
    )
  }

  test("independent package links use their explicit fragment identity") {
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
    val helper     = function(public.copy(name = helperName))
    val implementation = function(
      public.copy(name = p.Sym("implementation.apply")),
      List(
        p.Stmt.Var(
          p.Named("call", p.Type.Unit0),
          Some(p.Expr.Invoke(p.Type.FnRef(helperName), Nil, None, Nil, p.Type.Unit0)),
          false
        ),
        unitReturn.head
      ),
      Some(publicName)
    )
    val linked = PackageLinker.link(
      p.Package.LinkRequest(
        p.Interface(p.Sym("library"), List(public)),
        List(fragment("harvest-17", List(implementation, helper)))
      )
    )
    assert(linked.isRight, linked)
    val names = linked.toOption.get.program.functions.map(_.name).toSet
    assert(names(p.Sym(List("#fragment", "harvest-17", "vendor", "helper"))))
  }

  test("fragment identities are non-empty and unique") {
    val interface = p.Interface(p.Sym("library"), Nil)
    val empty     = p.Program(None, Nil, Nil)

    val unnamed = PackageLinker.link(
      p.Package.LinkRequest(interface, List(p.Package.Fragment("", empty)))
    )
    assert(unnamed.left.exists(_.contains("package fragment identity must not be empty")))

    val duplicate = PackageLinker.link(
      p.Package.LinkRequest(
        interface,
        List(p.Package.Fragment("same", empty), p.Package.Fragment("same", empty))
      )
    )
    assert(duplicate.left.exists(_.contains("package fragment identities must be unique")))
  }

  test("fragment links retain only the overload implemented by that fragment") {
    val name = p.Sym("library.apply")
    def declaration(tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("value", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val i32 = declaration(p.Type.IntS32)
    val f32 = declaration(p.Type.Float32)
    val implementation = function(
      i32.copy(name = p.Sym("implementation.apply")),
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(p.Named("value", p.Type.IntS32), Nil, p.Type.IntS32)))),
      Some(name)
    )
    val linked = PackageLinker.link(
      p.Package.LinkRequest(
        p.Interface(p.Sym("library"), List(i32, f32)),
        List(fragment("i32", List(implementation))),
        pruneUnimplementedDeclarations = true
      )
    )

    assertEquals(linked.map(_.interface.declarations), Right(List(i32)))
  }

  test("program linking materializes a canonical consumer entry without an ABI recipe") {
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
    val implementation = function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(implementationDecl.args(1).named, Nil, p.Type.IntS32)))),
      Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val root = function(
      public.copy(name = p.Sym("consumer.increment")),
      body = Nil,
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val linked = ProgramLinker.link(
      p.Program.LinkRequest(
        List(p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))),
        p.Program(None, List(root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(linked.isRight, linked)
    val program = linked.toOption.get
    val entry   = (program.entry.toList ::: program.functions).find(_.name == root.name).get
    assertEquals(entry.args.map(_.named.symbol), List("#context", "a0", "result"))
    assertEquals(entry.rtn, p.Type.Unit0)
  }

  test("materialized consumer entries use transaction-scoped boundary allocations") {
    val name    = p.Sym("library.copy")
    val pointer = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val public = p.FunctionDecl(
      name,
      Nil,
      None,
      List(
        p.Arg(
          p.Named("in", pointer),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(4))))
        ),
        p.Arg(
          p.Named("out", pointer),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(4))))
        )
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementation = function(
      public.copy(
        name = p.Sym("implementation.copy"),
        args = p.Arg(p.Named("#context", p.Spec.ContextType)) :: public.args
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val root = function(
      public.copy(name = p.Sym("consumer.copy")),
      body = Nil,
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val linked = ProgramLinker.link(
      p.Program.LinkRequest(
        List(p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))),
        p.Program(None, List(root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )

    assert(linked.isRight, linked)
    val entry = (linked.toOption.get.entry.toList ::: linked.toOption.get.functions).find(_.name == root.name).get
    val allocations = entry.body.flatMap(_.collectAll[p.Expr]).collect {
      case p.Expr.SpecOp(p.Spec.RemoteTempAlloc(_, _)) => ()
    }
    assertEquals(allocations.size, 2)
    assertEquals(
      entry.body.flatMap(_.collectAll[p.Expr]).count {
        case p.Expr.SpecOp(p.Spec.RemoteAlloc(_, _)) => true
        case _                                       => false
      },
      0
    )
    val outer =
      entry.body.collectFirst { case value: p.Stmt.Try => value }.getOrElse(fail("missing outer allocation guard"))
    val inner =
      outer.body.collectFirst { case value: p.Stmt.Try => value }.getOrElse(fail("missing inner allocation guard"))
    def freedBy(value: p.Stmt.Try) = value.fin.flatMap(_.collectAll[p.Expr]).collect {
      case p.Expr.SpecOp(p.Spec.RemoteFree(_, p.Term.Select(named, Nil, _))) => named.symbol
    }
    assertEquals(freedBy(outer), List("remote0"))
    assertEquals(freedBy(inner), List("remote1"))
  }

  test("program linking uses the canonical result-address ABI for an erased frontend result") {
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
    val root = function(
      public.copy(name = p.Sym("consumer.increment"), rtn = p.Type.Nothing),
      body = Nil,
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val linked = ProgramLinker.link(
      p.Program.LinkRequest(
        List(p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))),
        p.Program(None, List(root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(linked.isRight, linked)
    val program = linked.toOption.get
    val entry   = (program.entry.toList ::: program.functions).find(_.name == root.name).get
    assertEquals(entry.args.map(_.named.symbol), List("#context", "a0", "result"))
  }

  test("program linking batches roots across packages") {
    def packageAndRoot(prefix: String): (p.Package, p.Function) = {
      val publicName = p.Sym(s"$prefix.increment")
      val public = p.FunctionDecl(
        publicName,
        Nil,
        None,
        List(p.Arg(p.Named("value", p.Type.IntS32))),
        Nil,
        Nil,
        p.Type.IntS32,
        p.Function.Affinity.Host
      )
      val implementation = function(
        public.copy(name = p.Sym(s"implementation.$prefix.increment")),
        implements = Some(publicName),
        visibility = p.Function.Visibility.Exported
      )
      val root = function(
        public.copy(name = p.Sym(s"consumer.$prefix.increment")),
        implements = Some(publicName),
        visibility = p.Function.Visibility.Exported
      )
      p.Package(p.Interface(p.Sym(prefix), List(public)), p.Program(None, List(implementation), Nil)) -> root
    }
    val (firstPackage, firstRoot)   = packageAndRoot("first")
    val (secondPackage, secondRoot) = packageAndRoot("second")
    val linked = ProgramLinker.link(
      p.Program.LinkRequest(
        List(firstPackage, secondPackage),
        p.Program(None, List(firstRoot, secondRoot), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(linked.isRight, linked)
    val entries = linked.toOption.get.entry.toList ::: linked.toOption.get.functions
    assertEquals(entries.count(function => function.name == firstRoot.name || function.name == secondRoot.name), 2)
  }

  test("program linking records the implementation of each bound root and the reason no implementation fits the rest") {
    val publicName    = p.Sym("fit.identity")
    val t: p.Type.Var = p.Type.Var("T")
    val public = p.FunctionDecl(
      publicName,
      List(t),
      None,
      List(p.Arg(p.Named("value", t))),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val value: p.Type.Var = p.Type.Var("Value", Some(4))
    val implementationDecl = public.copy(
      name = p.Sym("implementation.fit.identity"),
      tpeVars = List(value),
      args = List(p.Arg(p.Named("value", value))),
      rtn = value
    )
    val implementation = function(
      implementationDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(implementationDecl.args.head.named, Nil, value)))),
      implements = Some(publicName),
      visibility = p.Function.Visibility.Exported
    )
    val pkg = p.Package(p.Interface(p.Sym("fit"), List(public)), p.Program(None, List(implementation), Nil))
    def root(name: String, tpe: p.Type) = function(
      public.copy(name = p.Sym(name), tpeVars = Nil, args = List(p.Arg(p.Named("value", tpe))), rtn = tpe),
      implements = Some(publicName),
      visibility = p.Function.Visibility.Exported
    )
    val narrow = root("consumer.narrow", p.Type.IntS32)
    val wide   = root("consumer.wide", p.Type.Float64)
    val sizes  = List(p.Program.TypeSize(p.Type.IntS32, 4), p.Program.TypeSize(p.Type.Float64, 8))
    val linked =
      ProgramLinker.link(p.Program.LinkRequest(List(pkg), p.Program(None, List(narrow, wide), Nil), Nil, sizes))
    assert(linked.isRight, linked)
    val program = linked.toOption.get
    assertEquals(
      program.metadata,
      List(
        p.MetaEntry(p.Conventions.LinkImplementationMeta + "consumer.narrow", "implementation.fit.identity"),
        p.MetaEntry(
          p.Conventions.LinkUnfitMeta + "consumer.wide",
          "no implementation fits `F64` of 8 bytes (sizes: 4)"
        )
      )
    )
    val names = (program.entry.toList ::: program.functions).map(_.name)
    assert(names.contains(narrow.name), names)
    assert(!names.contains(wide.name), names)

    val unfit = ProgramLinker.link(p.Program.LinkRequest(List(pkg), p.Program(None, List(wide), Nil), Nil, sizes))
    assertEquals(unfit.map(program => (program.entry, program.functions)), Right((None, Nil)))
    assert(
      ProgramLinker
        .link(p.Program.LinkRequest(List(pkg), p.Program(None, List(wide), Nil), Nil, sizes.take(1)))
        .left
        .exists(_.exists(_.contains("has no layout for `F64`")))
    )
  }

  test("program linking rejects invalid or conflicting type layouts") {
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
    val root = function(
      declaration.copy(name = p.Sym("consumer.noop")),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    def request(layouts: List[p.Program.TypeSize]) =
      p.Program.LinkRequest(List(pkg), p.Program(None, List(root), Nil), typeSizes = layouts)

    assert(
      ProgramLinker
        .link(request(List(p.Program.TypeSize(p.Type.IntS32, 4), p.Program.TypeSize(p.Type.IntS32, 8))))
        .left
        .exists(_.exists(_.contains("conflicts")))
    )
    assert(
      ProgramLinker
        .link(request(List(p.Program.TypeSize(p.Type.IntS32, 0))))
        .left
        .exists(_.exists(_.contains("must be positive")))
    )
  }

  test("package and program linking preserve a trailing-output result") {
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
        List(fragment("increment", List(implementation)))
      )
    )
    assert(linked.isRight, linked)
    val composed = linked.toOption.get.program.functions.head
    assertEquals(
      composed.args.last.boundary,
      Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))))
    )

    val root = function(
      public.copy(name = p.Sym("consumer.increment")),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = ProgramLinker.link(
      p.Program.LinkRequest(
        List(linked.toOption.get),
        p.Program(None, List(root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(resolved.isRight, resolved)
    val program = resolved.toOption.get
    val entry   = (program.entry.toList ::: program.functions).find(_.name == root.name).get
    val invoke  = entry.collectAll[p.Expr].collectFirst { case value: p.Expr.Invoke => value }.get
    assertEquals(invoke.rtn, p.Type.Unit0)
    assertEquals(invoke.args.last, p.Term.Select(entry.args.last.named, Nil, resultType))
  }

  test("program linking closes and substitutes a source callable") {
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
    val unrelatedType = p.Type.Struct(p.Sym("caller.Unrelated"), Nil)
    val unrelated = function(
      p.FunctionDecl(
        p.Sym("caller.unrelated"),
        Nil,
        None,
        List(p.Arg(p.Named("value", unrelatedType))),
        Nil,
        Nil,
        p.Type.Unit0,
        p.Function.Affinity.Host
      )
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))
    val root = function(
      public.copy(
        name = p.Sym("consumer.apply"),
        args = List(p.Arg(p.Named("value", p.Type.IntS32)), p.Arg(p.Named("operation", p.Type.FnRef(callable))))
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = ProgramLinker.link(
      p.Program.LinkRequest(
        List(pkg),
        // The unrelated function deliberately has no matching struct definition. It must not leak into the
        // selected callable closure or make an otherwise valid import fail.
        p.Program(None, List(unrelated, callableFunction, root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(resolved.isRight, resolved)
    val resolvedProgram = resolved.toOption.get
    val functions       = resolvedProgram.entry.toList ::: resolvedProgram.functions
    val entry           = functions.find(_.name == root.name).get
    assertEquals(entry.args.map(_.named.symbol), List("#context", "a0", "result"))
    assert(functions.exists(_.name == callable))
    val resolvedImplementation = functions
      .find(function => function.name != callable && function.collectAll[p.Type].contains(p.Type.FnRef(callable)))
      .get
    assertEquals(resolvedImplementation.args.map(_.named.symbol), List("value"))
    assert(resolvedImplementation.collectAll[p.Type].contains(p.Type.FnRef(callable)))
    assertEquals(
      resolvedProgram.meta(p.Conventions.LinkCallableMeta + callable.fqcn),
      Some(root.name.fqcn)
    )
  }

  private def callableImplementation(name: p.Sym, publicName: p.Sym) = {
    val element: p.Type   = p.Type.Var("Element")
    val operation: p.Type = p.Type.Var("Operation")
    val decl = p.FunctionDecl(
      name,
      List(p.Type.Var("Element"), p.Type.Var("Operation")),
      None,
      List(p.Arg(p.Named("value", element)), p.Arg(p.Named("operation", operation))),
      Nil,
      Nil,
      element,
      p.Function.Affinity.Host
    )
    val self = p.Named("operation_ref", p.Type.Ptr(operation, p.Type.Space.Private))
    function(
      decl,
      List(
        p.Stmt.Var(
          self,
          Some(
            p.Expr.RefTo(
              p.Term.Select(decl.args(1).named, Nil, operation),
              None,
              operation,
              p.Type.Space.Private,
              p.Region.Rooted(decl.args(1).named)
            )
          )
        ),
        p.Stmt.Return(
          p.Expr.Invoke(
            operation,
            Nil,
            Some(p.Term.Select(self, Nil, self.tpe)),
            List(p.Term.Select(decl.args.head.named, Nil, element)),
            element
          )
        )
      ),
      Some(publicName),
      visibility = p.Function.Visibility.Exported
    )
  }

  private def applyPublic(name: p.Sym) = p.FunctionDecl(
    name,
    Nil,
    None,
    List(
      p.Arg(p.Named("value", p.Type.IntS32)),
      p.Arg(p.Named("operation", p.Type.Exec(Nil, List(p.Type.IntS32), p.Type.IntS32)))
    ),
    Nil,
    Nil,
    p.Type.IntS32,
    p.Function.Affinity.Host
  )

  test("program linking passes a closure environment to its call operator") {
    val name           = p.Sym("library.apply")
    val public         = applyPublic(name)
    val implementation = callableImplementation(p.Sym("implementation.apply"), name)
    val offset         = p.Sym("caller.Offset")
    val offsetTpe      = p.Type.Struct(offset, Nil)
    val self           = p.Named("#this", p.Type.Ptr(offsetTpe, p.Type.Space.Private))
    val callDecl = p.FunctionDecl(
      p.Sym("caller.Offset.call"),
      Nil,
      Some(p.Arg(self)),
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val call = function(
      callDecl,
      List(
        p.Stmt.Return(
          p.Expr.IntrOp(
            p.Intr.Add(
              p.Term.Select(self, List(p.PathStep.Deref, p.PathStep.Field("k")), p.Type.IntS32),
              p.Term.Select(callDecl.args.head.named, Nil, p.Type.IntS32),
              p.Type.IntS32
            )
          )
        )
      )
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))
    val root = function(
      public.copy(
        name = p.Sym("consumer.apply"),
        args = List(p.Arg(p.Named("value", p.Type.IntS32)), p.Arg(p.Named("operation", offsetTpe)))
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = ProgramLinker.link(
      p.Program.LinkRequest(
        List(pkg),
        p.Program(None, List(call, root), List(p.StructDef(offset, Nil, List(p.Named("k", p.Type.IntS32)), Nil))),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4), p.Program.TypeSize(offsetTpe, 4))
      )
    )
    assert(resolved.isRight, resolved)
    val functions = resolved.toOption.get.entry.toList ::: resolved.toOption.get.functions
    val entry     = functions.find(_.name == root.name).get
    assertEquals(entry.args.map(_.named.symbol), List("#context", "a0", "a1", "result"))
    val resolvedImplementation = functions.find(_.collectAll[p.Type].contains(p.Type.FnRef(callDecl.name))).get
    assertEquals(resolvedImplementation.args.map(_.named.tpe), List(p.Type.IntS32, offsetTpe))
    val invokes = resolvedImplementation.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke }
    assertEquals(invokes.map(_.callee), List(p.Type.FnRef(callDecl.name)))
    assertEquals(invokes.flatMap(_.receiver).map(_.tpe), List(self.tpe))
    assert(functions.exists(_.name == callDecl.name))
  }

  private def linkClosureEnvironment(launching: Boolean, hostCall: Boolean) = {
    val name       = p.Sym("library.apply")
    val public     = applyPublic(name)
    val kernelName = p.Sym("implementation.kernel")
    val kernel = function(
      p.FunctionDecl(kernelName, Nil, None, Nil, Nil, Nil, p.Type.Unit0, p.Function.Affinity.Offload),
      unitReturn
    )
    val launch = p.Stmt.Var(
      p.Named("launch", p.Type.Unit0),
      Some(
        p.Expr.SpecOp(
          p.Spec.RemoteLaunch(
            p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque),
            p.Term.Poison(p.Type.FnRef(kernelName)),
            Nil,
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(1),
            p.Term.IntU32Const(0),
            Nil
          )
        )
      ),
      isMutable = false
    )
    val callable = callableImplementation(p.Sym("implementation.apply"), name)
    val value    = callable.decl.args.head.named
    val implementation =
      if (!launching) callable
      else if (hostCall) callable.copy(body = launch :: callable.body)
      else callable.copy(body = List(launch, p.Stmt.Return(p.Expr.Alias(p.Term.Select(value, Nil, value.tpe)))))
    val lookup    = p.Sym("caller.Lookup")
    val lookupTpe = p.Type.Struct(lookup, Nil)
    val table     = p.Named("table", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val self      = p.Named("#this", p.Type.Ptr(lookupTpe, p.Type.Space.Private))
    val callDecl = p.FunctionDecl(
      p.Sym("caller.Lookup.call"),
      Nil,
      Some(p.Arg(self)),
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val call = function(callDecl, List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(0)))))
    val pkg = p.Package(
      p.Interface(p.Sym("library"), List(public)),
      p.Program(None, if (launching) List(implementation, kernel) else List(implementation), Nil)
    )
    val root = function(
      public.copy(
        name = p.Sym("consumer.apply"),
        args = List(p.Arg(p.Named("value", p.Type.IntS32)), p.Arg(p.Named("operation", lookupTpe)))
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    ProgramLinker.link(
      p.Program.LinkRequest(
        List(pkg),
        p.Program(None, List(call, root), List(p.StructDef(lookup, Nil, List(table), Nil))),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4), p.Program.TypeSize(lookupTpe, 8))
      )
    ) -> root.name
  }

  private def closureEnvironmentLink(launching: Boolean, hostCall: Boolean) = {
    val (resolved, rootName) = linkClosureEnvironment(launching, hostCall)
    assert(resolved.isRight, resolved)
    val functions = resolved.toOption.get.entry.toList ::: resolved.toOption.get.functions
    functions.find(_.name == rootName).get
  }

  test("program linking mirrors pointers captured by a closure environment for device launches") {
    val entry   = closureEnvironmentLink(launching = true, hostCall = false)
    val foreign = entry.collectAll[p.Expr].collect { case call: p.Expr.ForeignCall => call.name }
    assertEquals(
      foreign,
      List(
        p.Conventions.RuntimeAbi.SmaEnsureMin,
        p.Conventions.RuntimeAbi.SmaVisitClear,
        p.Conventions.RuntimeAbi.SmaReadAlloc
      )
    )
    val patched = entry.collectAll[p.Stmt].collect { case p.Stmt.Cond(_, List(p.Stmt.Mut(target, _)), Nil) =>
      target.root.symbol -> target.steps
    }
    assertEquals(patched, List("v1" -> List(p.PathStep.Field("table"))))
  }

  test("program linking keeps host closure environments for host-only implementations") {
    val entry = closureEnvironmentLink(launching = false, hostCall = true)
    assertEquals(entry.collectAll[p.Expr].collect { case call: p.Expr.ForeignCall => call.name }, Nil)
  }

  test("program linking rejects an implementation that launches and calls a captured closure on the host") {
    val (resolved, _) = linkClosureEnvironment(launching = true, hostCall = true)
    assert(resolved.isLeft, resolved)
    assert(resolved.left.toOption.get.exists(_.contains("on the host")), resolved)
  }

  test("program linking binds a closure invoked by a package helper") {
    val name              = p.Sym("library.apply")
    val public            = applyPublic(name)
    val element: p.Type   = p.Type.Var("Element")
    val operation: p.Type = p.Type.Var("Operation")
    val helperName        = p.Sym("implementation.invoke")
    val helperOperation   = p.Named("operation", operation)
    val helperValue       = p.Named("value", element)
    val helperReceiver    = p.Named("operation_ref", p.Type.Ptr(operation, p.Type.Space.Global))
    val helper = function(
      p.FunctionDecl(
        helperName,
        Nil,
        None,
        List(p.Arg(helperValue), p.Arg(helperOperation)),
        Nil,
        Nil,
        element,
        p.Function.Affinity.Host
      ),
      List(
        p.Stmt.Var(
          helperReceiver,
          Some(
            p.Expr.RefTo(
              p.Term.Select(helperOperation, Nil, operation),
              None,
              operation,
              p.Type.Space.Global,
              p.Region.Rooted(helperOperation)
            )
          )
        ),
        p.Stmt.Return(
          p.Expr.Invoke(
            operation,
            Nil,
            Some(p.Term.Select(helperReceiver, Nil, helperReceiver.tpe)),
            List(p.Term.Select(helperValue, Nil, element)),
            element
          )
        )
      )
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
            p.Type.FnRef(helperName),
            Nil,
            None,
            implementationDecl.args.map(argument => p.Term.Select(argument.named, Nil, argument.named.tpe)),
            element
          )
        )
      ),
      Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val offset    = p.Sym("caller.Offset")
    val offsetTpe = p.Type.Struct(offset, Nil)
    val self      = p.Named("#this", p.Type.Ptr(offsetTpe, p.Type.Space.Global))
    val callDecl = p.FunctionDecl(
      p.Sym("caller.Offset.call"),
      Nil,
      Some(p.Arg(self)),
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val call = function(
      callDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(callDecl.args.head.named, Nil, p.Type.IntS32))))
    )
    val pkg = p.Package(
      p.Interface(p.Sym("library"), List(public)),
      p.Program(None, List(implementation, helper), Nil)
    )
    val root = function(
      public.copy(
        name = p.Sym("consumer.apply"),
        args = List(p.Arg(p.Named("value", p.Type.IntS32)), p.Arg(p.Named("operation", offsetTpe)))
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = ProgramLinker.link(
      p.Program.LinkRequest(
        List(pkg),
        p.Program(None, List(call, root), List(p.StructDef(offset, Nil, List(p.Named("k", p.Type.IntS32)), Nil))),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4), p.Program.TypeSize(offsetTpe, 4))
      )
    )
    assert(resolved.isRight, resolved)
    val functions = resolved.toOption.get.entry.toList ::: resolved.toOption.get.functions
    val callees   = functions.flatMap(_.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.callee })
    assert(callees.contains(p.Type.FnRef(callDecl.name)), callees)
    assert(!callees.exists(_.isInstanceOf[p.Type.Struct]), callees)
  }

  test("program linking drops a package receiver for a stateless callable") {
    val name           = p.Sym("library.apply")
    val public         = applyPublic(name)
    val implementation = callableImplementation(p.Sym("implementation.apply"), name)
    val callableDecl = p.FunctionDecl(
      p.Sym("caller.increment"),
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val callable = function(
      callableDecl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Select(callableDecl.args.head.named, Nil, p.Type.IntS32))))
    )
    val pkg = p.Package(p.Interface(p.Sym("library"), List(public)), p.Program(None, List(implementation), Nil))
    val root = function(
      public.copy(
        name = p.Sym("consumer.apply"),
        args = List(
          p.Arg(p.Named("value", p.Type.IntS32)),
          p.Arg(p.Named("operation", p.Type.FnRef(callableDecl.name)))
        )
      ),
      implements = Some(name),
      visibility = p.Function.Visibility.Exported
    )
    val resolved = ProgramLinker.link(
      p.Program.LinkRequest(
        List(pkg),
        p.Program(None, List(callable, root), Nil),
        typeSizes = List(p.Program.TypeSize(p.Type.IntS32, 4))
      )
    )
    assert(resolved.isRight, resolved)
    val functions              = resolved.toOption.get.entry.toList ::: resolved.toOption.get.functions
    val resolvedImplementation = functions.find(_.collectAll[p.Type].contains(p.Type.FnRef(callableDecl.name))).get
    assertEquals(resolvedImplementation.args.map(_.named.symbol), List("value"))
    val invokes = resolvedImplementation.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke }
    assertEquals(invokes.map(_.receiver), List(None))
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
    val matched = ProgramLinker.matchCall(
      public,
      ProgramLinker.CallSignature(name, Nil, None, List(p.Type.FnRef(callable)), p.Type.IntS32),
      List(callableDecl)
    )
    assertEquals(matched.map(_.types), Right(Map("Element" -> p.Type.IntS32)))
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
