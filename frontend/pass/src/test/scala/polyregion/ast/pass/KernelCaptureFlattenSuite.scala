package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class KernelCaptureFlattenSuite extends munit.FunSuite {

  private val global = p.Type.Space.Global
  private val i32p   = p.Type.Ptr(p.Type.IntS32, global)

  private def host(kernel: p.Function, capture: p.Named, callCapture: Option[p.Term] = None): p.Function = {
    val fired = named("fired", p.Type.Unit0)
    entry(
      body = List(
        p.Stmt.Var(capture, None, isMutable = true),
        p.Stmt.Var(
          fired,
          Some(
            p.Expr
              .Invoke(p.Type.FnRef(kernel.name), Nil, Some(callCapture.getOrElse(selectT(capture))), Nil, p.Type.Unit0)
          ),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      moduleCaptures = Nil,
      termCaptures = Nil
    ).modifyDecl(_.copy(affinity = p.Function.Affinity.Host))
  }

  test("extracts a nested pointer leaf while retaining scalar capture fields") {
    val innerSym = sym("Inner")
    val capSym   = sym("Capture")
    val innerTpe = p.Type.Struct(innerSym, Nil)
    val capTpe   = p.Type.Struct(capSym, Nil)
    val innerDef = p.StructDef(innerSym, Nil, List(named("data", i32p), named("count", p.Type.IntS32)), Nil)
    val capDef   = p.StructDef(capSym, Nil, List(named("inner", innerTpe), named("scalar", p.Type.IntS32)), Nil)
    val self     = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val data     = p.Term.Select(self, List(p.PathStep.Field("inner"), p.PathStep.Field("data")), i32p)
    val scalar   = p.Term.Select(self, List(p.PathStep.Field("scalar")), p.Type.IntS32)
    val kernel = fn(
      "kernel",
      body = List(
        p.Stmt.Var(named("x"), Some(p.Expr.Index(data, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Var(named("s"), Some(p.Expr.Alias(scalar))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", p.Type.Ptr(capTpe, global))

    val in        = program(host(kernel, capture), List(kernel), List(innerDef, capDef))
    val out       = VerifyAnchors(strict = true)(KernelCaptureFlatten(in, NoopLog), NoopLog)
    val piped     = VerifyAnchors(strict = true)(KernelCaptureFlatten(PartialEval()(in, NoopLog), NoopLog), NoopLog)
    val outKernel = out.functions.find(_.name == kernel.name).get
    val outCall   = out.entry.collectWhere[p.Expr] { case i: p.Expr.Invoke => i }.head

    assertEquals(piped.functions.find(_.name == kernel.name).get.args.map(_.named.tpe), List(i32p))
    assertEquals(outKernel.receiver.map(_.named), Some(self))
    assertEquals(outKernel.args.map(_.named.tpe), List(i32p))
    assertEquals(outCall.receiver.map(_.tpe), Some(p.Type.Ptr(capTpe, global)))
    assertEquals(outCall.args.map(_.tpe), List[p.Type](i32p))
    assert(outKernel.collectAll[p.Term].exists {
      case p.Term.Select(root, Nil, `i32p`) => root == outKernel.args.head.named
      case _                                => false
    })
    assert(outKernel.collectAll[p.Term].exists {
      case p.Term.Select(`self`, List(p.PathStep.Field("scalar")), p.Type.IntS32) => true
      case _                                                                      => false
    })
    assertEquals(
      outCall.args.last,
      p.Term.Select(capture, List(p.PathStep.Field("inner"), p.PathStep.Field("data")), i32p)
    )
  }

  test("is available as a pipeline step") {
    val built = PassPipelineParser.parseStep("KernelCaptureFlatten").flatMap(PassRegistry.build)
    assert(built.isRight, built.toString)
  }

  test("extracts pointer leaves from an ordinary capture argument") {
    val capSym = sym("ArgumentCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capPtr = p.Type.Ptr(capTpe, global)
    val capDef = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val capArg = named(p.Conventions.CaptureArg, capPtr)
    val data   = p.Term.Select(capArg, List(p.PathStep.Field("data")), i32p)
    val module = arg("module", p.Type.IntU32)
    val kernel = fn(
      "argumentKernel",
      body = List(
        p.Stmt.Var(named("x"), Some(p.Expr.Index(data, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry,
      moduleCaptures = List(module)
    ).modifyDecl(_.copy(args = List(p.Arg(capArg)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", capPtr)
    val fired   = named("fired", p.Type.Unit0)
    val caller = entry(
      body = List(
        p.Stmt.Var(capture, None, isMutable = true),
        p.Stmt.Var(
          fired,
          Some(
            p.Expr.Invoke(
              p.Type.FnRef(kernel.name),
              Nil,
              None,
              List(p.Term.IntU32Const(7), selectT(capture)),
              p.Type.Unit0
            )
          ),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      moduleCaptures = Nil,
      termCaptures = Nil
    ).modifyDecl(_.copy(affinity = p.Function.Affinity.Host))

    val out       = KernelCaptureFlatten(program(caller, List(kernel), List(capDef)), NoopLog)
    val outKernel = out.functions.head
    val outCall   = out.entry.collectWhere[p.Expr] { case call: p.Expr.Invoke => call }.head

    assertEquals(outKernel.receiver, None)
    assertEquals(outKernel.args.map(_.named.tpe), List[p.Type](capPtr, i32p))
    assertEquals(
      outCall.args,
      List[p.Term](
        p.Term.IntU32Const(7),
        selectT(capture),
        p.Term.Select(capture, List(p.PathStep.Field("data")), i32p)
      )
    )
  }

  test("extracts pointer leaves from a specialised remote launch") {
    val capSym = sym("RemoteCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capPtr = p.Type.Ptr(capTpe, global)
    val capDef = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val capArg = named(p.Conventions.CaptureArg, capPtr)
    val data   = p.Term.Select(capArg, List(p.PathStep.Field("data")), i32p)
    val kernel = fn(
      "remoteKernel",
      body = List(
        p.Stmt.Var(named("x"), Some(p.Expr.Index(data, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry,
      moduleCaptures = List(arg("moduleA", p.Type.IntU32), arg("moduleB", p.Type.IntU64))
    ).modifyDecl(
      _.copy(
        tpeVars = List(p.Type.Var("T")),
        args = List(arg("erased", p.Type.Var("T")), p.Arg(capArg)),
        affinity = p.Function.Affinity.Offload
      )
    )
    val capture = named("capture", capPtr)
    val context = p.Term.NullPtrConst(p.Type.IntU8, global, p.Region.Opaque)
    val one     = p.Term.IntU32Const(1)
    val launch = p.Spec.RemoteLaunch(
      context,
      p.Term.Poison(p.Type.FnRef(kernel.name)),
      List(p.Type.Unit0),
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(p.Term.IntU32Const(7), p.Term.IntU64Const(8), selectT(capture))
    )
    val caller = entry(
      body = List(
        p.Stmt.Var(capture, None, isMutable = true),
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    ).modifyDecl(_.copy(affinity = p.Function.Affinity.Host))

    val out       = KernelCaptureFlatten(Specialisation(program(caller, List(kernel), List(capDef)), NoopLog), NoopLog)
    val outKernel = out.functions.head
    val outLaunch = out.entry.collectWhere[p.Expr] { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }.head

    assertEquals(outKernel.args.map(_.named.tpe), List[p.Type](p.Type.Unit0, capPtr, i32p))
    assertEquals(
      outLaunch.args,
      List[p.Term](
        p.Term.IntU32Const(7),
        p.Term.IntU64Const(8),
        selectT(capture),
        p.Term.Select(capture, List(p.PathStep.Field("data")), i32p)
      )
    )
  }

  test("resolves a remote kernel overload before flattening its capture") {
    val capSym = sym("OverloadedCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capPtr = p.Type.Ptr(capTpe, global)
    val capDef = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val capArg = named(p.Conventions.CaptureArg, capPtr)
    val data   = p.Term.Select(capArg, List(p.PathStep.Field("data")), i32p)
    val selected = fn(
      "overloadedKernel",
      body = List(
        p.Stmt.Var(named("x"), Some(p.Expr.Index(data, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    ).modifyDecl(_.copy(args = List(p.Arg(capArg)), affinity = p.Function.Affinity.Offload))
    val other = fn("overloadedKernel", args = List(arg("value", p.Type.IntS32)))
      .modifyDecl(_.copy(affinity = p.Function.Affinity.Offload))
    val capture = named("capture", capPtr)
    val context = p.Term.NullPtrConst(p.Type.IntU8, global, p.Region.Opaque)
    val one     = p.Term.IntU32Const(1)
    val launch = p.Spec.RemoteLaunch(
      context,
      p.Term.Poison(p.Type.FnRef(selected.name)),
      Nil,
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(selectT(capture))
    )
    val caller = entry(
      body = List(
        p.Stmt.Var(capture, None, isMutable = true),
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    ).modifyDecl(_.copy(affinity = p.Function.Affinity.Host))

    val out = KernelCaptureFlatten(program(caller, List(selected, other), List(capDef)), NoopLog)

    assertEquals(out.functions.map(_.args.map(_.named.tpe)), List(List(capPtr, i32p), List(p.Type.IntS32)))
    assertEquals(
      out.entry.collectWhere[p.Expr] { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }.head.args,
      List[p.Term](selectT(capture), p.Term.Select(capture, List(p.PathStep.Field("data")), i32p))
    )
  }

  test("rewrites updates through an extracted pointer leaf") {
    val capSym              = sym("Capture")
    val capTpe              = p.Type.Struct(capSym, Nil)
    val capDef              = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val self                = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val data: p.Term.Select = p.Term.Select(self, List(p.PathStep.Field("data")), i32p)
    val kernel = fn(
      "updateKernel",
      body = List(
        p.Stmt.Update(data, p.Term.IntS32Const(0), p.Term.IntS32Const(42)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", p.Type.Ptr(capTpe, global))
    val outKernel =
      KernelCaptureFlatten(program(host(kernel, capture), List(kernel), List(capDef)), NoopLog).functions.head

    assert(outKernel.body.exists {
      case p.Stmt.Update(p.Term.Select(root, Nil, `i32p`), _, _) => root == outKernel.args.head.named
      case _                                                     => false
    })
  }

  test("leaves a scalar-only capture unchanged") {
    val capSym = sym("ScalarCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capDef = p.StructDef(capSym, Nil, List(named("value", p.Type.IntS32)), Nil)
    val self   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val kernel =
      fn("scalarKernel", convention = p.CallConvention.OffloadEntry)
        .modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", p.Type.Ptr(capTpe, global))
    val in      = program(host(kernel, capture), List(kernel), List(capDef))

    assertEquals(KernelCaptureFlatten(in, NoopLog), in)
  }

  test("rejects a pointer whose pointee contains another pointer") {
    val nodeSym = sym("Node")
    val capSym  = sym("GraphCapture")
    val nodeTpe = p.Type.Struct(nodeSym, Nil)
    val capTpe  = p.Type.Struct(capSym, Nil)
    val nodePtr = p.Type.Ptr(nodeTpe, global)
    val nodeDef = p.StructDef(nodeSym, Nil, List(named("next", nodePtr)), Nil)
    val capDef  = p.StructDef(capSym, Nil, List(named("head", nodePtr)), Nil)
    val self    = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val head    = p.Term.Select(self, List(p.PathStep.Field("head")), nodePtr)
    val kernel = fn(
      "graphKernel",
      body =
        List(p.Stmt.Var(named("h", nodePtr), Some(p.Expr.Alias(head))), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", p.Type.Ptr(capTpe, global))

    val ex = intercept[RuntimeException] {
      KernelCaptureFlatten(program(host(kernel, capture), List(kernel), List(nodeDef, capDef)), NoopLog)
    }
    assert(ex.getMessage.contains("pointer graph"))
  }

  test("repairs a by-value escape of a pointer-bearing subobject") {
    val innerSym = sym("RepairInner")
    val capSym   = sym("RepairCapture")
    val innerTpe = p.Type.Struct(innerSym, Nil)
    val capTpe   = p.Type.Struct(capSym, Nil)
    val innerDef = p.StructDef(innerSym, Nil, List(named("data", i32p), named("count", p.Type.IntS32)), Nil)
    val capDef   = p.StructDef(capSym, Nil, List(named("inner", innerTpe)), Nil)
    val self     = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val inner    = p.Term.Select(self, List(p.PathStep.Field("inner")), innerTpe)
    val copy     = named("copy", innerTpe)
    val kernel = fn(
      "escapeKernel",
      body = List(p.Stmt.Var(copy, Some(p.Expr.Alias(inner))), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", p.Type.Ptr(capTpe, global))
    val outKernel =
      KernelCaptureFlatten(program(host(kernel, capture), List(kernel), List(innerDef, capDef)), NoopLog).functions.head
    val repair = outKernel.body.collectFirst {
      case p.Stmt.Var(
            n,
            Some(p.Expr.Alias(p.Term.Select(`self`, List(p.PathStep.Field("inner")), `innerTpe`))),
            true
          ) =>
        n
    }.get

    assertEquals(outKernel.args.map(_.named.tpe), List(i32p))
    assert(outKernel.body.exists {
      case p.Stmt.Mut(
            p.Term.Select(`repair`, List(p.PathStep.Field("data")), `i32p`),
            p.Expr.Alias(p.Term.Select(root, Nil, `i32p`))
          ) =>
        root == outKernel.args.head.named
      case _ => false
    })
    assert(outKernel.body.exists {
      case p.Stmt.Var(`copy`, Some(p.Expr.Alias(p.Term.Select(`repair`, Nil, `innerTpe`))), _) => true
      case _                                                                                   => false
    })
  }

  test("repairs an address-taken subobject and binds every pointer leaf below it") {
    val innerSym = sym("PairView")
    val capSym   = sym("ViewCapture")
    val innerTpe = p.Type.Struct(innerSym, Nil)
    val capTpe   = p.Type.Struct(capSym, Nil)
    val innerDef = p.StructDef(
      innerSym,
      Nil,
      List(named("input", i32p), named("output", i32p), named("count", p.Type.IntS32)),
      Nil
    )
    val capDef  = p.StructDef(capSym, Nil, List(named("view", innerTpe)), Nil)
    val self    = named(p.Conventions.ThisReceiver, p.Type.Ptr(capTpe, global))
    val view    = p.Term.Select(self, List(p.PathStep.Field("view")), innerTpe)
    val viewPtr = p.Type.Ptr(innerTpe, global)
    val ref     = named("viewRef", viewPtr)
    val input   = p.Term.Select(ref, List(p.PathStep.Deref, p.PathStep.Field("input")), i32p)
    val direct  = p.Term.Select(self, List(p.PathStep.Field("view"), p.PathStep.Field("input")), i32p)
    val kernel = fn(
      "refEscapeKernel",
      body = List(
        p.Stmt.Var(ref, Some(p.Expr.RefTo(view, None, innerTpe, global, p.Region.Opaque))),
        p.Stmt.Var(named("x"), Some(p.Expr.Index(input, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Var(named("y"), Some(p.Expr.Index(direct, p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture   = named("capture", p.Type.Ptr(capTpe, global))
    val out       = KernelCaptureFlatten(program(host(kernel, capture), List(kernel), List(innerDef, capDef)), NoopLog)
    val outKernel = out.functions.head
    val outCall   = out.entry.collectWhere[p.Expr] { case call: p.Expr.Invoke => call }.head
    val repair = outKernel.body.collectFirst {
      case p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(`self`, List(p.PathStep.Field("view")), `innerTpe`))), true) =>
        n
    }.get

    assertEquals(outKernel.args.map(_.named.tpe), List(i32p, i32p))
    assertEquals(
      outCall.args,
      List(
        p.Term.Select(capture, List(p.PathStep.Field("view"), p.PathStep.Field("input")), i32p),
        p.Term.Select(capture, List(p.PathStep.Field("view"), p.PathStep.Field("output")), i32p)
      )
    )
    assertEquals(
      outKernel.body.count {
        case p.Stmt.Mut(p.Term.Select(`repair`, List(p.PathStep.Field("input" | "output")), `i32p`), _) => true
        case _                                                                                          => false
      },
      2
    )
    assert(outKernel.body.exists {
      case p.Stmt.Var(
            `ref`,
            Some(p.Expr.RefTo(p.Term.Select(`repair`, Nil, `innerTpe`), None, `innerTpe`, _, _)),
            _
          ) =>
        true
      case _ => false
    })
  }

  test("rejects escape of the entire capture pointer") {
    val capSym = sym("IdentityCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capPtr = p.Type.Ptr(capTpe, global)
    val capDef = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val self   = named(p.Conventions.ThisReceiver, capPtr)
    val kernel = fn(
      "captureIdentityKernel",
      body = List(
        p.Stmt.Var(named("captureAlias", capPtr), Some(p.Expr.Alias(p.Term.Select(self, Nil, capPtr)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", capPtr)

    val ex = intercept[RuntimeException] {
      KernelCaptureFlatten(program(host(kernel, capture), List(kernel), List(capDef)), NoopLog)
    }
    assert(ex.getMessage.contains("entire capture pointer"))
  }

  test("rejects a non-select call capture") {
    val capSym = sym("BadCallCapture")
    val capTpe = p.Type.Struct(capSym, Nil)
    val capPtr = p.Type.Ptr(capTpe, global)
    val capDef = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val self   = named(p.Conventions.ThisReceiver, capPtr)
    val data   = p.Term.Select(self, List(p.PathStep.Field("data")), i32p)
    val kernel = fn(
      "badCallKernel",
      body =
        List(p.Stmt.Var(named("p", i32p), Some(p.Expr.Alias(data))), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", capPtr)

    val ex = intercept[RuntimeException] {
      KernelCaptureFlatten(
        program(host(kernel, capture, Some(p.Term.Poison(capPtr))), List(kernel), List(capDef)),
        NoopLog
      )
    }
    assert(ex.getMessage.contains("is not a Select"))
  }

  test("rejects address-taking and mutation of a capture pointer slot") {
    val capSym              = sym("SlotCapture")
    val capTpe              = p.Type.Struct(capSym, Nil)
    val capPtr              = p.Type.Ptr(capTpe, global)
    val capDef              = p.StructDef(capSym, Nil, List(named("data", i32p)), Nil)
    val self                = named(p.Conventions.ThisReceiver, capPtr)
    val data: p.Term.Select = p.Term.Select(self, List(p.PathStep.Field("data")), i32p)
    val ret                 = p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    val addressKernel = fn(
      "addressKernel",
      body = List(
        p.Stmt
          .Var(named("slot", p.Type.Ptr(i32p, global)), Some(p.Expr.RefTo(data, None, i32p, global, p.Region.Opaque))),
        ret
      ),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val capture = named("capture", capPtr)
    val addressEx = intercept[RuntimeException] {
      KernelCaptureFlatten(program(host(addressKernel, capture), List(addressKernel), List(capDef)), NoopLog)
    }
    assert(addressEx.getMessage.contains("address-taking"))

    val mutateKernel = fn(
      "mutateKernel",
      body = List(p.Stmt.Mut(data, p.Expr.Alias(p.Term.NullPtrConst(p.Type.IntS32, global, p.Region.Opaque))), ret),
      convention = p.CallConvention.OffloadEntry
    ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
    val mutationEx = intercept[RuntimeException] {
      KernelCaptureFlatten(program(host(mutateKernel, capture), List(mutateKernel), List(capDef)), NoopLog)
    }
    assert(mutationEx.getMessage.contains("mutation"))
  }

  test("rejects unsupported capture pointer shapes") {
    def reject(member: p.Named, defs: List[p.StructDef] = Nil): String = {
      val capSym = sym(s"Rejected${member.symbol}")
      val capTpe = p.Type.Struct(capSym, Nil)
      val capPtr = p.Type.Ptr(capTpe, global)
      val capDef = p.StructDef(capSym, Nil, List(member), Nil)
      val self   = named(p.Conventions.ThisReceiver, capPtr)
      val field  = p.Term.Select(self, List(p.PathStep.Field(member.symbol)), member.tpe)
      val kernel = fn(
        s"rejected${member.symbol}",
        body = List(
          p.Stmt.Var(named("use", member.tpe), Some(p.Expr.Alias(field))),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        ),
        convention = p.CallConvention.OffloadEntry
      ).modifyDecl(_.copy(receiver = Some(p.Arg(self)), affinity = p.Function.Affinity.Offload))
      val capture = named("capture", capPtr)
      intercept[RuntimeException] {
        KernelCaptureFlatten(program(host(kernel, capture), List(kernel), defs :+ capDef), NoopLog)
      }.getMessage
    }

    val privatePtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    assert(reject(named("privateData", privatePtr)).contains("non-global"))

    val pointers = p.Type.Arr(i32p, 2, global)
    assert(reject(named("pointers", pointers)).contains("pointer-bearing capture array"))

    val recursiveSym = sym("RecursiveValue")
    val recursiveTpe = p.Type.Struct(recursiveSym, Nil)
    val recursiveDef = p.StructDef(recursiveSym, Nil, List(named("self", recursiveTpe), named("data", i32p)), Nil)
    assert(reject(named("recursive", recursiveTpe), List(recursiveDef)).contains("recursive by-value"))

    val unionSym = sym("PointerUnion")
    val unionTpe = p.Type.Struct(unionSym, Nil)
    val unionDef =
      p.StructDef(unionSym, Nil, List(named("data", i32p), named("value", p.Type.IntS64)), Nil, isUnion = true)
    assert(reject(named("union", unionTpe), List(unionDef)).contains("pointer-bearing capture union"))
  }
}
