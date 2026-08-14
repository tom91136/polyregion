package polyregion.ast.pass

import polyregion.ast.{
  InterfaceBinding,
  MsgPack,
  PolyAST as p,
  bind,
  classifyArguments,
  conformsTo,
  modifyDecl,
  remapArgs,
  resolve,
  signature,
  validate,
  given
}
import polyregion.ast.Traversal.*

class InterfaceDeclarationSuite extends munit.FunSuite {

  private def transformDecl = {
    val t = p.Type.Var("T")
    val n = p.Arg(p.Named("n", p.Type.IntS32))
    val in = p.Arg(
      p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
      boundary = Some(
        p.Arg.Boundary(
          p.Arg.Access.Read,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))
        )
      )
    )
    val out = p.Arg(
      p.Named("out", p.Type.Ptr(t, p.Type.Space.Global)),
      boundary = Some(
        p.Arg.Boundary(
          p.Arg.Access.Write,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))
        )
      )
    )
    val op = p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t)))
    p.FunctionDecl(
      p.Sym(List("foo", "transform")),
      List("T"),
      None,
      List(in, out, n, op),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
  }

  test("a generic callable boundary round-trips as a bodyless declaration") {
    import p.repr

    val decl = transformDecl

    assertEquals(MsgPack.decode[p.FunctionDecl](MsgPack.encode(decl)), Right(decl))

    val library = p.InterfaceDef(p.Sym(List("foo")), List(decl), Nil)
    assertEquals(MsgPack.decode[p.InterfaceDef](MsgPack.encode(library)), Right(library))
    assertEquals(library.decls, List(decl))
    assertNotEquals(
      decl.args.head.repr,
      decl.args.head.copy(boundary = decl.args.head.boundary.map(_.copy(access = p.Arg.Access.Write))).repr
    )
    assert(decl.repr.contains("affinity=Host"))
    assert(library.repr.contains("read elements(arg[2])"))
  }

  test("a function composes its callable declaration") {
    val decl = transformDecl
    val fn = p.Function(
      decl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      p.Function.Visibility.Internal,
      p.Function.FpMode.Relaxed,
      false
    )
    val renamed = fn.modifyDecl(_.copy(name = p.Sym(List("foo", "renamed"))))
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(p.Type.IntS32),
      p.Type.Unit0
    )

    assertEquals(fn.decl, decl)
    assertEquals(fn.signature, decl.signature)
    assertEquals(renamed.name, p.Sym(List("foo", "renamed")))
    assertEquals(renamed.decl.args, decl.args)
    assertEquals(call.tpeArgs, List(p.Type.Float32))
    assertEquals(MsgPack.decode[p.Function](MsgPack.encode(fn)), Right(fn))
  }

  test("function traversal includes declaration types") {
    val decl = transformDecl
    val fn = p.Function(
      decl,
      List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      p.Function.Visibility.Internal,
      p.Function.FpMode.Relaxed,
      false
    )
    val rewritten = fn.modifyAll[p.Type] {
      case p.Type.Var("T") => p.Type.Float64
      case other           => other
    }

    assert(fn.collectAll[p.Type].contains(p.Type.Var("T")))
    assertEquals(rewritten.args.head.named.tpe, p.Type.Ptr(p.Type.Float64, p.Type.Space.Global))
  }

  test("argument rewrites preserve positional extents") {
    val decl    = transformDecl
    val error   = p.Arg(p.Named("#error", p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)))
    val shifted = decl.remapArgs(error :: decl.args)

    assertEquals(
      shifted.args(1).boundary.map(_.extent),
      Some(p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(3)))
    )
    intercept[IllegalArgumentException](decl.remapArgs(decl.args.patch(2, Nil, 1)))
  }

  test("wire-shape changes advance plugin ABI versions") {
    assertEquals(p.PolyPassAbi.Version, 2)
    assertEquals(p.PolyJitAbi.Version, 3)
  }

  test("extent expressions represent a combined output capacity") {
    val extent = p.Arg.Extent.Elements(
      p.Arg.SizeExpr.Add(p.Arg.SizeExpr.Param(1), p.Arg.SizeExpr.Param(3))
    )

    assertEquals(
      extent,
      p.Arg.Extent.Elements(p.Arg.SizeExpr.Add(p.Arg.SizeExpr.Param(1), p.Arg.SizeExpr.Param(3)))
    )
  }

  test("declaration validation rejects non-pointer boundaries and invalid extents") {
    val decl            = transformDecl
    val invalidBoundary = decl.copy(args = decl.args.updated(2, decl.args(2).copy(boundary = decl.args.head.boundary)))
    val invalidExtent = decl.copy(args =
      decl.args.updated(
        0,
        decl.args.head.copy(boundary =
          Some(
            p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(9)))
          )
        )
      )
    )
    val callableExtent = decl.copy(args =
      decl.args.updated(
        0,
        decl.args.head.copy(boundary =
          Some(
            p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(3)))
          )
        )
      )
    )

    assert(invalidBoundary.validate.exists(_.contains("not a pointer")))
    assert(invalidExtent.validate.exists(_.contains("out of range")))
    assert(callableExtent.validate.exists(_.contains("integral scalar")))
  }

  test("declaration validation rejects duplicate parameter symbols across parameter kinds") {
    val receiver = p.Arg(p.Named("x", p.Type.IntS32))
    val decl = p.FunctionDecl(
      p.Sym("foo.duplicate"),
      Nil,
      Some(receiver),
      List(p.Arg(p.Named("x", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )

    assert(decl.validate.exists(_.contains("duplicate parameter `x`")))
  }

  test("a concrete call binds type variables and a callable by structure") {
    val decl   = transformDecl
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val lambda = p.FunctionDecl(
      p.Sym(List("caller", "lambda")),
      Nil,
      None,
      List(p.Arg(p.Named("x", p.Type.Float32))),
      Nil,
      Nil,
      p.Type.Float32,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(f32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(lambda.name)),
      p.Type.Unit0
    )

    assertEquals(
      decl.bind(call, List(lambda)),
      Right(InterfaceBinding.Binding(Map("T" -> p.Type.Float32), Map(3 -> lambda.name)))
    )
  }

  test("call binding selects a callable overload by structure") {
    val decl   = transformDecl
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val name   = p.Sym(List("caller", "overloaded"))
    def callable(tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("x", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val f32 = callable(p.Type.Float32)
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(f32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(name)),
      p.Type.Unit0
    )

    assert(decl.bind(call, List(callable(p.Type.IntS32), f32)).isRight)
    assert(
      decl
        .bind(call, List(f32, f32))
        .left
        .exists(_.exists(_.contains("2 matching declarations")))
    )
  }

  test("callback overload selection incorporates later argument inference") {
    val t    = p.Type.Var("T")
    val name = p.Sym(List("caller", "overloaded"))
    val decl = p.FunctionDecl(
      p.Sym(List("foo", "apply")),
      List("T"),
      None,
      List(
        p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t))),
        p.Arg(p.Named("x", t))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    def callable(tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("x", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      decl.name,
      Nil,
      None,
      List(p.Type.FnRef(name), p.Type.Float32),
      p.Type.Unit0
    )

    assertEquals(
      decl.bind(call, List(callable(p.Type.IntS32), callable(p.Type.Float32))),
      Right(InterfaceBinding.Binding(Map("T" -> p.Type.Float32), Map(0 -> name)))
    )
  }

  test("callable arguments do not infer public type variables") {
    val t     = p.Type.Var("T")
    val first = p.Sym(List("caller", "first"))
    val last  = p.Sym(List("caller", "last"))
    val decl = p.FunctionDecl(
      p.Sym(List("foo", "zip")),
      List("T"),
      None,
      List(
        p.Arg(p.Named("first", p.Type.Exec(Nil, List(t), t))),
        p.Arg(p.Named("last", p.Type.Exec(Nil, List(t), t)))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    def callable(name: p.Sym, tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("x", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      decl.name,
      Nil,
      None,
      List(p.Type.FnRef(first), p.Type.FnRef(last)),
      p.Type.Unit0
    )

    assert(
      decl
        .bind(
          call,
          List(
            callable(first, p.Type.IntS32),
            callable(first, p.Type.Float32),
            callable(last, p.Type.Float32)
          )
        )
        .left
        .exists(_.contains("declaration type variable `T` is not bound by the call"))
    )
  }

  test("explicit public types make callable overload checks independent") {
    val t     = p.Type.Var("T")
    val first = p.Sym(List("caller", "first"))
    val last  = p.Sym(List("caller", "last"))
    val decl = p.FunctionDecl(
      p.Sym(List("foo", "zip")),
      List("T"),
      None,
      List(
        p.Arg(p.Named("first", p.Type.Exec(Nil, List(t), t))),
        p.Arg(p.Named("last", p.Type.Exec(Nil, List(t), t)))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    def callable(name: p.Sym, tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("x", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(p.Type.FnRef(first), p.Type.FnRef(last)),
      p.Type.Unit0
    )

    assertEquals(
      decl.bind(
        call,
        List(
          callable(first, p.Type.IntS32),
          callable(first, p.Type.Float32),
          callable(last, p.Type.Float32)
        )
      ),
      Right(InterfaceBinding.Binding(Map("T" -> p.Type.Float32), Map(0 -> first, 1 -> last)))
    )
  }

  test("direct calls reject declarations with explicit captures") {
    val decl = transformDecl.copy(moduleCaptures = List(p.Arg(p.Named("state", p.Type.IntS32))))
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(
        p.Type.Ptr(p.Type.Float32, p.Type.Space.Global),
        p.Type.Ptr(p.Type.Float32, p.Type.Space.Global),
        p.Type.IntS32,
        p.Type.Exec(Nil, List(p.Type.Float32), p.Type.Float32)
      ),
      p.Type.Unit0
    )

    assert(
      decl
        .bind(call, Nil)
        .left
        .exists(_.contains("public declarations with explicit captures cannot be called directly"))
    )
  }

  test("a boundary declaration yields a structural argument plan") {
    import InterfaceBinding.ArgumentKind

    assertEquals(
      transformDecl.classifyArguments,
      Right(
        List(
          ArgumentKind.Buffer(
            p.Arg.Access.Read,
            p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))
          ),
          ArgumentKind.Buffer(
            p.Arg.Access.Write,
            p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))
          ),
          ArgumentKind.ExtentScalar,
          ArgumentKind.Callable(p.Type.Exec(Nil, List(p.Type.Var("T")), p.Type.Var("T")))
        )
      )
    )

    val missingBoundary =
      transformDecl.copy(args = transformDecl.args.updated(0, transformDecl.args.head.copy(boundary = None)))
    assert(missingBoundary.classifyArguments.left.exists(_.exists(_.contains("has no boundary"))))
  }

  test("binding rejects a conflicting element type and callable signature") {
    val decl   = transformDecl
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val i32Ptr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val wrongLambda = p.FunctionDecl(
      p.Sym(List("caller", "wrong")),
      Nil,
      None,
      List(p.Arg(p.Named("x", p.Type.Float32))),
      Nil,
      Nil,
      p.Type.Bool1,
      p.Function.Affinity.Host
    )
    val wrongElement = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(i32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(wrongLambda.name)),
      p.Type.Unit0
    )
    val wrongCallable = wrongElement.copy(args = List(f32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(wrongLambda.name)))

    assert(decl.bind(wrongElement, List(wrongLambda)).isLeft)
    assert(decl.bind(wrongCallable, List(wrongLambda)).isLeft)
  }

  test("binding respects callable type-variable scope and alpha-renaming") {
    val decl = transformDecl.copy(args =
      transformDecl.args.updated(
        3,
        p.Arg(p.Named("op", p.Type.Exec(List("T"), List(p.Type.Var("T")), p.Type.Var("T"))))
      )
    )
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Float32),
      None,
      List(
        f32Ptr,
        f32Ptr,
        p.Type.IntS32,
        p.Type.Exec(List("Element"), List(p.Type.Var("Element")), p.Type.Var("Element"))
      ),
      p.Type.Unit0
    )

    assertEquals(
      decl.bind(call, Nil),
      Right(InterfaceBinding.Binding(Map("T" -> p.Type.Float32), Map.empty))
    )
  }

  test("binding rejects declaration type variables that cannot be inferred") {
    val decl   = transformDecl.copy(tpeVars = List("T", "Unused"))
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val call = p.InvokeSignature(
      decl.name,
      Nil,
      None,
      List(
        f32Ptr,
        f32Ptr,
        p.Type.IntS32,
        p.Type.Exec(Nil, List(p.Type.Float32), p.Type.Float32)
      ),
      p.Type.Unit0
    )

    assert(
      decl
        .bind(call, Nil)
        .left
        .exists(_.contains("declaration type variable `Unused` is not bound by the call"))
    )
  }

  test("binding resolves consistent chained type substitutions") {
    val decl = p.FunctionDecl(
      p.Sym("foo.chain"),
      List("T", "U"),
      None,
      List(p.Arg(p.Named("x", p.Type.Var("T")))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      decl.name,
      List(p.Type.Var("U"), p.Type.Float32),
      None,
      List(p.Type.Float32),
      p.Type.Unit0
    )

    assertEquals(
      decl.bind(call, Nil),
      Right(InterfaceBinding.Binding(Map("T" -> p.Type.Float32, "U" -> p.Type.Float32), Map.empty))
    )
  }

  test("binding resolves a callable returned by reference") {
    val callable = p.FunctionDecl(
      p.Sym("caller.callback"),
      Nil,
      None,
      List(p.Arg(p.Named("x", p.Type.Float32))),
      Nil,
      Nil,
      p.Type.Float32,
      p.Function.Affinity.Host
    )
    val decl = p.FunctionDecl(
      p.Sym("foo.factory"),
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Exec(Nil, List(p.Type.Float32), p.Type.Float32),
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(decl.name, Nil, None, Nil, p.Type.FnRef(callable.name))

    assertEquals(decl.bind(call, List(callable)), Right(InterfaceBinding.Binding(Map.empty, Map.empty)))
  }

  test("binding rejects duplicate binders in an actual callable type") {
    val expected = p.Type.Exec(
      List("A", "B"),
      List(p.Type.Var("A"), p.Type.Var("B")),
      p.Type.Unit0
    )
    val malformed = p.Type.Exec(
      List("Element", "Element"),
      List(p.Type.Var("Element"), p.Type.Var("Element")),
      p.Type.Unit0
    )
    val decl = p.FunctionDecl(
      p.Sym("foo.callable"),
      Nil,
      None,
      List(p.Arg(p.Named("op", expected))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(decl.name, Nil, None, List(malformed), p.Type.Unit0)

    assert(decl.bind(call, Nil).isLeft)
  }

  test("validation rejects malformed callable type-variable binders") {
    val duplicate = transformDecl.copy(args =
      transformDecl.args.updated(
        3,
        p.Arg(
          p.Named("op", p.Type.Exec(List("Element", "Element"), List(p.Type.Var("Element")), p.Type.Var("Element")))
        )
      )
    )
    val empty = transformDecl.copy(args =
      transformDecl.args.updated(
        3,
        p.Arg(p.Named("op", p.Type.Exec(List(" "), List(p.Type.Var(" ")), p.Type.Var(" "))))
      )
    )

    assert(duplicate.validate.exists(_.contains("duplicate callable type variable `Element`")))
    assert(empty.validate.exists(_.contains("callable type variable 0 is empty")))
  }

  test("receiver extents classify referenced arguments") {
    val receiver = p.Arg(
      p.Named("self", p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)),
      boundary = Some(
        p.Arg.Boundary(
          p.Arg.Access.Read,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(0))
        )
      )
    )
    val decl = p.FunctionDecl(
      p.Sym("foo.member"),
      Nil,
      Some(receiver),
      List(p.Arg(p.Named("n", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )

    assertEquals(
      decl.classifyArguments,
      Right(List(InterfaceBinding.ArgumentKind.ExtentScalar))
    )
  }

  test("implementation conformance accepts alpha-renamed direct declarations") {
    val public = transformDecl
    val impl = public.copy(
      name = p.Sym(List("implementation", "transform")),
      tpeVars = List("Element"),
      args = public.args.map(_.modifyAll[p.Type] {
        case p.Type.Var("T") => p.Type.Var("Element")
        case other           => other
      })
    )

    assertEquals(
      impl.conformsTo(public),
      Right(
        InterfaceBinding.ImplementationBinding(
          Map("Element" -> p.Type.Var("T")),
          Map.empty,
          InterfaceBinding.ResultBinding.Direct
        )
      )
    )
  }

  test("implementation conformance keeps public type variables rigid") {
    val public = p.FunctionDecl(
      p.Sym("foo.rigid"),
      List("T"),
      None,
      List(
        p.Arg(p.Named("x", p.Type.Var("T"))),
        p.Arg(p.Named("y", p.Type.Float32))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementation = public.copy(
      name = p.Sym("implementation.rigid"),
      args = List(
        p.Arg(p.Named("x", p.Type.Float32)),
        p.Arg(p.Named("y", p.Type.Var("T")))
      )
    )

    assert(implementation.conformsTo(public).isLeft)
  }

  test("implementation conformance does not capture public callable binders") {
    val public = p.FunctionDecl(
      p.Sym("foo.poly"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named(
            "op",
            p.Type.Exec(
              List("B"),
              List(p.Type.Ptr(p.Type.Var("B"), p.Type.Space.Global)),
              p.Type.Unit0
            )
          )
        )
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementation = public.copy(
      name = p.Sym("implementation.poly"),
      tpeVars = List("E"),
      args = List(
        p.Arg(
          p.Named(
            "op",
            p.Type.Exec(List("A"), List(p.Type.Var("E")), p.Type.Unit0)
          )
        )
      )
    )

    assert(implementation.conformsTo(public).isLeft)
  }

  test("implementation conformance keeps stored public variables rigid") {
    val public = p.FunctionDecl(
      p.Sym("foo.namespaces"),
      List("B"),
      None,
      List(
        p.Arg(p.Named("x", p.Type.Var("B"))),
        p.Arg(p.Named("y", p.Type.Float32)),
        p.Arg(p.Named("z", p.Type.Float32))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val implementation = public.copy(
      name = p.Sym("implementation.namespaces"),
      tpeVars = List("A", "B"),
      args = List(
        p.Arg(p.Named("x", p.Type.Var("A"))),
        p.Arg(p.Named("y", p.Type.Var("A"))),
        p.Arg(p.Named("z", p.Type.Var("B")))
      )
    )

    assert(implementation.conformsTo(public).isLeft)
  }

  test("implementation conformance makes a trailing result slot explicit") {
    val input = transformDecl.args.head.copy(boundary =
      transformDecl.args.head.boundary.map(
        _.copy(extent = p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1)))
      )
    )
    val public = transformDecl.copy(
      name = p.Sym(List("foo", "reduce")),
      args = List(input, p.Arg(p.Named("n", p.Type.IntS32))),
      rtn = p.Type.Var("T")
    )
    val result = p.Arg(
      p.Named("result", p.Type.Ptr(p.Type.Var("Element"), p.Type.Space.Global)),
      boundary = Some(
        p.Arg.Boundary(
          p.Arg.Access.Write,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))
        )
      )
    )
    val implementation = p.FunctionDecl(
      p.Sym(List("implementation", "reduce")),
      List("Element"),
      None,
      List(
        public.args.head.modifyAll[p.Type] {
          case p.Type.Var("T") => p.Type.Var("Element")
          case other           => other
        },
        public.args(1),
        result
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )

    assertEquals(
      implementation.conformsTo(public),
      Right(
        InterfaceBinding.ImplementationBinding(
          Map("Element" -> p.Type.Var("T")),
          Map.empty,
          InterfaceBinding.ResultBinding.TrailingOutput(2)
        )
      )
    )

    val readableResult = implementation.copy(args =
      implementation.args.updated(
        2,
        result.copy(boundary =
          result.boundary.map(
            _.copy(access = p.Arg.Access.ReadWrite)
          )
        )
      )
    )
    assert(readableResult.conformsTo(public).isLeft)
  }

  test("implementation conformance rejects a non-global trailing result") {
    val public = p.FunctionDecl(
      p.Sym("foo.value"),
      Nil,
      None,
      Nil,
      Nil,
      Nil,
      p.Type.Float32,
      p.Function.Affinity.Host
    )
    val output = p.Arg(
      p.Named("result", p.Type.Ptr(p.Type.Float32, p.Type.Space.Constant)),
      boundary = Some(
        p.Arg.Boundary(
          p.Arg.Access.Write,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))
        )
      )
    )
    val implementation = public.copy(
      name = p.Sym("implementation.value"),
      args = List(output),
      rtn = p.Type.Unit0
    )

    assert(implementation.conformsTo(public).isLeft)
  }

  test("package resolution selects an exact conforming implementation by capabilities and layout") {
    val public = transformDecl
    val f32Ptr = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val lambda = p.FunctionDecl(
      p.Sym(List("caller", "lambda")),
      Nil,
      None,
      List(p.Arg(p.Named("x", p.Type.Float32))),
      Nil,
      Nil,
      p.Type.Float32,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      public.name,
      List(p.Type.Float32),
      None,
      List(f32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(lambda.name)),
      p.Type.Unit0
    )
    def implementation(name: String) = public.copy(
      name = p.Sym(List("implementation", name)),
      tpeVars = List("Element"),
      args = public.args.map(_.modifyAll[p.Type] {
        case p.Type.Var("T") => p.Type.Var("Element")
        case other           => other
      })
    )
    val w4 = p.ImplementationCandidate(
      public.name,
      implementation("transform_w4"),
      List("gpu"),
      List(p.TypeSizeConstraint("Element", 4))
    )
    val w8 = p.ImplementationCandidate(
      public.name,
      implementation("transform_w8"),
      List("gpu"),
      List(p.TypeSizeConstraint("Element", 8))
    )
    val index = p.PackageIndex(
      p.InterfaceDef(p.Sym("foo"), List(public)),
      List(w8, w4)
    )
    assertEquals(
      MsgPack.decode[p.PackageIndex](MsgPack.encode(index)),
      Right(index)
    )

    val resolved = index.resolve(
      call,
      List(lambda),
      Set("gpu"),
      Map(p.Type.Float32 -> 4, p.Type.Float64 -> 8)
    )
    assertEquals(resolved.map(_.candidate), Right(w4))

    val unavailable = index.resolve(
      call,
      List(lambda),
      Set.empty,
      Map(p.Type.Float32 -> 4)
    )
    assert(unavailable.left.exists(_.exists(_.contains("requires capability `gpu`"))))

    val ambiguous = index
      .copy(candidates = List(w4, w4.copy(implementation = implementation("transform_w4_alt"))))
      .resolve(
        call,
        List(lambda),
        Set("gpu"),
        Map(p.Type.Float32 -> 4)
      )
    assert(ambiguous.left.exists(_.exists(_.contains("ambiguous"))))

    val ambiguousReversed = index
      .copy(candidates = List(w4.copy(implementation = implementation("transform_w4_alt")), w4))
      .resolve(
        call,
        List(lambda),
        Set("gpu"),
        Map(p.Type.Float32 -> 4)
      )
    assertEquals(ambiguousReversed, ambiguous)

    val wrongSymbol = index.resolve(
      call.copy(name = p.Sym("foo.other")),
      List(lambda),
      Set("gpu"),
      Map(p.Type.Float32 -> 4)
    )
    assert(wrongSymbol.left.exists(_.exists(_.contains("no public declaration"))))
  }

  test("package resolution selects a public overload by call structure") {
    val name = p.Sym(List("foo", "overloaded"))
    def declaration(tpe: p.Type) = p.FunctionDecl(
      name,
      Nil,
      None,
      List(p.Arg(p.Named("x", tpe))),
      Nil,
      Nil,
      tpe,
      p.Function.Affinity.Host
    )
    val f32            = declaration(p.Type.Float32)
    val f64            = declaration(p.Type.Float64)
    val implementation = f32.copy(name = p.Sym(List("implementation", "overloaded_f32")))
    val candidate      = p.ImplementationCandidate(name, implementation, Nil, Nil)
    val index          = p.PackageIndex(p.InterfaceDef(p.Sym("foo"), List(f64, f32)), List(candidate))
    val call           = p.InvokeSignature(name, Nil, None, List(p.Type.Float32), p.Type.Float32)

    assertEquals(
      index.resolve(call, Nil, Set.empty, Map.empty).map(_.publicDecl),
      Right(f32)
    )
    assert(
      index
        .copy(interface = index.interface.copy(decls = List(f32, f32)))
        .resolve(
          call,
          Nil,
          Set.empty,
          Map.empty
        )
        .left
        .exists(_.exists(_.contains("ambiguous public declaration")))
    )
  }

  test("package resolution rejects cyclic call substitutions") {
    val public   = transformDecl
    val variable = p.Type.Var("T")
    val ptr      = p.Type.Ptr(variable, p.Type.Space.Global)
    val call = p.InvokeSignature(
      public.name,
      List(variable),
      None,
      List(ptr, ptr, p.Type.IntS32, p.Type.Exec(Nil, List(variable), variable)),
      p.Type.Unit0
    )
    val implementation = public.copy(
      name = p.Sym("implementation.transform"),
      tpeVars = List("Element"),
      args = public.args.map(_.modifyAll[p.Type] {
        case p.Type.Var("T") => p.Type.Var("Element")
        case other           => other
      })
    )
    val candidate = p.ImplementationCandidate(
      public.name,
      implementation,
      Nil,
      List(p.TypeSizeConstraint("Element", 4))
    )
    val result = p
      .PackageIndex(p.InterfaceDef(p.Sym("foo"), List(public)), List(candidate))
      .resolve(call, Nil, Set.empty, Map.empty)

    assert(result.left.exists(_.exists(_.contains("not concrete"))))
  }
}
