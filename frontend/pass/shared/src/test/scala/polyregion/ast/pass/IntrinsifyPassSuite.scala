package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class IntrinsifyPassSuite extends munit.FunSuite {

  // Spec: calls into the polyregion.scalalang.intrinsics module should be lowered to the
  // corresponding Math/Spec/Intr op nodes, removing the Invoke from the AST. Calls to other
  // unknown symbols should be left alone.

  private val intrinsicsTpe  = p.Type.Struct(sym("polyregion", "scalalang", "intrinsics$"), Nil)
  private val intrinsicsRecv = selectT(named("intrinsics$", intrinsicsTpe))

  private def call(op: String, args: List[p.Term], rtn: p.Type): p.Expr.Invoke =
    p.Expr.Invoke(sym("polyregion", "scalalang", "intrinsics$", op), Nil, Some(intrinsicsRecv), args, rtn)

  test("intrinsics.sin(x) lowers to a MathOp(Sin)") {
    val xArg = arg("x", p.Type.Float32)
    val e = entry(
      args = List(xArg),
      body = List(p.Stmt.Var(named("r", p.Type.Float32), Some(call("sin", List(selectT(xArg.named)), p.Type.Float32))))
    )
    val out     = IntrinsifyPass(program(e), NoopLog)
    val invokes = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokes.isEmpty, s"sin call should be lowered, got: ${invokes.map(_.repr)}")
    val maths = out.entry.body.flatMap(_.collectWhere[p.Expr] { case m: p.Expr.MathOp => m })
    assert(maths.exists(_.op.isInstanceOf[p.Math.Sin]), s"expected Math.Sin, got: ${maths.map(_.repr)}")
  }

  test("intrinsics.gpuBarrierGlobal() lowers to a SpecOp") {
    val e = entry(body = List(p.Stmt.Var(named("u", p.Type.Unit0), Some(call("gpuBarrierGlobal", Nil, p.Type.Unit0)))))
    val out     = IntrinsifyPass(program(e), NoopLog)
    val invokes = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokes.isEmpty, s"gpuBarrierGlobal should be lowered, got: ${invokes.map(_.repr)}")
    val specs = out.entry.body.flatMap(_.collectWhere[p.Expr] { case s: p.Expr.SpecOp => s })
    assert(specs.nonEmpty, "expected a SpecOp from gpuBarrierGlobal")
  }
}
