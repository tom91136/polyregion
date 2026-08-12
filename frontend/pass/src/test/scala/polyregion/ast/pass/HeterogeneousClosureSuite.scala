package polyregion.ast.pass

import polyregion.ast.{MsgPack, PolyAST as p, given}
import PassTest.*

class HeterogeneousClosureSuite extends munit.FunSuite {

  test("heterogeneous orchestration operations round-trip with their complete operands") {
    import p.repr

    val kernel = p.Term.Poison(p.Type.FnRef(p.Sym("example.kernel")))
    val one    = p.Term.IntU32Const(1)
    val zero   = p.Term.IntU32Const(0)
    val bytes  = p.Term.IntU64Const(64)
    val ptr    = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch: p.Spec.RemoteLaunch = p.Spec.RemoteLaunch(
      kernel,
      List(p.Type.Float32),
      one,
      one,
      one,
      one,
      one,
      one,
      zero,
      zero,
      List(ptr)
    )
    val operations: List[p.Spec] = List(
      launch,
      p.Spec.RemoteAlloc(bytes),
      p.Spec.RemoteMemcpy(ptr, ptr, bytes, p.Direction.LocalToRemote),
      p.Spec.RemoteSync(zero),
      p.Spec.RemoteFree(ptr)
    )

    operations.foreach(operation => assertEquals(MsgPack.decode[p.Spec](MsgPack.encode(operation)), Right(operation)))
    assertEquals(
      launch.terms,
      List(kernel, one, one, one, one, one, one, zero, zero, ptr)
    )
    assertEquals(launch.tpeArgs, List(p.Type.Float32))
    assertEquals(launch.tpe, p.Type.Unit0)
    assert(p.Expr.SpecOp(launch).repr.contains("example.kernel"))
    assert(p.Expr.SpecOp(p.Spec.RemoteAlloc(bytes)).repr.contains("remoteAlloc"))
    assert(
      p.Expr
        .SpecOp(p.Spec.RemoteMemcpy(ptr, ptr, bytes, p.Direction.LocalToRemote))
        .repr
        .contains("localToRemote")
    )
    assert(p.Expr.SpecOp(p.Spec.RemoteSync(zero)).repr.contains("remoteSync"))
    assert(p.Expr.SpecOp(p.Spec.RemoteFree(ptr)).repr.contains("remoteFree"))
    assertEquals(p.Direction.LocalToRemote.repr, "localToRemote")
    assertEquals(p.Direction.RemoteToLocal.repr, "remoteToLocal")
    assertEquals(p.Direction.RemoteToRemote.repr, "remoteToRemote")
  }

  test("heterogeneous orchestration operands are verified") {
    def launch(kernel: p.Term, args: List[p.Term] = Nil) = p.Spec.RemoteLaunch(
      kernel,
      Nil,
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(0),
      p.Term.IntU32Const(0),
      args
    )
    val existingKernel = fn("existing.kernel", args = List(arg("value", p.Type.IntU32)))
    val badOps: List[p.Spec] = List(
      p.Spec.RemoteLaunch(
        p.Term.IntU32Const(0),
        Nil,
        p.Term.Bool1Const(false),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(0),
        p.Term.IntU32Const(0),
        Nil
      ),
      launch(p.Term.Poison(p.Type.FnRef(p.Sym("missing.kernel")))),
      launch(p.Term.Poison(p.Type.FnRef(existingKernel.name))),
      p.Spec.RemoteAlloc(p.Term.Bool1Const(false)),
      p.Spec.RemoteFree(p.Term.IntU64Const(0)),
      p.Spec.RemoteMemcpy(
        p.Term.IntU64Const(0),
        p.Term.IntU64Const(0),
        p.Term.Bool1Const(false),
        p.Direction.RemoteToRemote
      ),
      p.Spec.RemoteSync(p.Term.Bool1Const(false))
    )
    val body = badOps.zipWithIndex.map { (op, index) =>
      p.Stmt.Var(p.Named(s"bad$index", op.tpe), Some(p.Expr.SpecOp(op)))
    } :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    val failures =
      Verify(program(entry(body = body), List(existingKernel)), NoopLog, verifyFunction = true).flatMap(_._2)

    assert(failures.exists(_.contains("kernel must be a function reference")), failures.mkString("\n"))
    assert(failures.exists(_.contains("launch references an undefined kernel")), failures.mkString("\n"))
    assert(failures.exists(_.contains("launch does not match any kernel overload")), failures.mkString("\n"))
    assert(failures.exists(_.contains("launch dimension gridX must be U32")), failures.mkString("\n"))
    assert(failures.exists(_.contains("allocation byte count must be U64")), failures.mkString("\n"))
    assert(failures.exists(_.contains("free operand must be a global pointer")), failures.mkString("\n"))
    assert(failures.exists(_.contains("copy destination must be a global pointer")), failures.mkString("\n"))
    assert(failures.exists(_.contains("copy source must be a global pointer")), failures.mkString("\n"))
    assert(failures.exists(_.contains("copy byte count must be U64")), failures.mkString("\n"))
    assert(failures.exists(_.contains("stream handle must be a global pointer, U32, or U64")), failures.mkString("\n"))
  }

  test("launch verification follows the physical capture ABI") {
    val kernel = fn(
      "captured.kernel",
      args = List(arg("value", p.Type.Float32), arg("erased", p.Type.Unit0)),
      moduleCaptures = List(arg("module", p.Type.IntU32)),
      termCaptures = List(arg("term", p.Type.IntU64))
    )
    val launch = p.Spec.RemoteLaunch(
      p.Term.Poison(p.Type.FnRef(kernel.name)),
      Nil,
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(0),
      p.Term.IntU32Const(0),
      List(p.Term.IntU32Const(1), p.Term.IntU64Const(2), p.Term.Float32Const(3))
    )
    val body = List(
      p.Stmt.Var(p.Named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val failures = Verify(program(entry(body = body), List(kernel)), NoopLog, verifyFunction = true).flatMap(_._2)

    assertEquals(failures, Nil)
  }

  test("launch verification erases generic unit parameters after substitution") {
    val kernel = fn("generic.kernel", args = List(arg("erased", p.Type.Var("T"))), tpeVars = List("T"))
    val launch = p.Spec.RemoteLaunch(
      p.Term.Poison(p.Type.FnRef(kernel.name)),
      List(p.Type.Unit0),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(1),
      p.Term.IntU32Const(0),
      p.Term.IntU32Const(0),
      Nil
    )
    val body = List(
      p.Stmt.Var(p.Named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val failures = Verify(program(entry(body = body), List(kernel)), NoopLog, verifyFunction = true).flatMap(_._2)

    assertEquals(failures, Nil)
  }
}
