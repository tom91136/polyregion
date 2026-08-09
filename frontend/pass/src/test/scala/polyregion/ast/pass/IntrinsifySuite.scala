package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class IntrinsifySuite extends munit.FunSuite {

  // Spec: calls into the polyregion.scalalang.intrinsics module should be lowered to the
  // corresponding Math/Spec/Intr op nodes, removing the Invoke from the AST. Calls to other
  // unknown symbols should be left alone.

  private val intrinsicsTpe  = p.Type.Struct(sym("polyregion", "scalalang", "intrinsics$"), Nil)
  private val intrinsicsRecv = selectT(named("intrinsics$", intrinsicsTpe))

  private def call(op: String, args: List[p.Term], rtn: p.Type): p.Expr.Invoke =
    p.Expr.Invoke(p.Type.FnRef(sym("polyregion", "scalalang", "intrinsics$", op)), Nil, Some(intrinsicsRecv), args, rtn)

  test("intrinsics.sin(x) lowers to a MathOp(Sin)") {
    val xArg = arg("x", p.Type.Float32)
    val e = entry(
      args = List(xArg),
      body = List(p.Stmt.Var(named("r", p.Type.Float32), Some(call("sin", List(selectT(xArg.named)), p.Type.Float32))))
    )
    val out     = Intrinsify(program(e), NoopLog)
    val invokes = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokes.isEmpty, s"sin call should be lowered, got: ${invokes.map(_.repr)}")
    val maths = out.entry.body.flatMap(_.collectWhere[p.Expr] { case m: p.Expr.MathOp => m })
    assert(maths.exists(_.op.isInstanceOf[p.Math.Sin]), s"expected Math.Sin, got: ${maths.map(_.repr)}")
  }

  test("intrinsics.gpuBarrierGlobal() lowers to a SpecOp") {
    val e = entry(body = List(p.Stmt.Var(named("u", p.Type.Unit0), Some(call("gpuBarrierGlobal", Nil, p.Type.Unit0)))))
    val out     = Intrinsify(program(e), NoopLog)
    val invokes = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokes.isEmpty, s"gpuBarrierGlobal should be lowered, got: ${invokes.map(_.repr)}")
    val specs = out.entry.body.flatMap(_.collectWhere[p.Expr] { case s: p.Expr.SpecOp => s })
    assert(specs.nonEmpty, "expected a SpecOp from gpuBarrierGlobal")
  }

  test("promotion statements produced inside try remain inside the try") {
    val x   = arg("x", p.Type.IntS8)
    val y   = arg("y", p.Type.IntS8)
    val sum = named("sum", p.Type.IntS32)
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(sym("scala", "Byte", "+")),
      Nil,
      Some(selectT(x.named)),
      List(selectT(y.named)),
      p.Type.IntS32
    )
    val e = entry(
      args = List(x, y),
      body = List(p.Stmt.Try(List(p.Stmt.Var(sum, Some(invoke))), Nil, Nil))
    )

    val out = Intrinsify(program(e), NoopLog)
    assertEquals(out.entry.body.size, 1)
    val body = out.entry.body.head.asInstanceOf[p.Stmt.Try].body
    assert(body.size > 1, s"expected promotion temporaries in the try body, got ${body.map(_.repr)}")
    assert(body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i }).isEmpty)
  }

  test("promotion temporaries are unique across try blocks") {
    val x = arg("x", p.Type.IntS8)
    val y = arg("y", p.Type.IntS8)
    def sum(name: String) = p.Stmt.Var(
      named(name, p.Type.IntS32),
      Some(
        p.Expr.Invoke(
          p.Type.FnRef(sym("scala", "Byte", "+")),
          Nil,
          Some(selectT(x.named)),
          List(selectT(y.named)),
          p.Type.IntS32
        )
      )
    )
    val e = entry(
      args = List(x, y),
      body = List(
        p.Stmt.Try(
          List(sum("bodySum")),
          List(handler(None, None, List(sum("handlerSum")), None)),
          List(sum("finallySum"))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val out    = Intrinsify(program(e), NoopLog)
    val errors = Verify(out, NoopLog, verifyFunction = true).flatMap(_._2)
    assert(!errors.exists(_.startsWith("Variable intr_")), errors.mkString("\n"))
  }

  test("promotion statements produced in raise cleanup remain in cleanup") {
    val x = arg("x", p.Type.IntS8)
    val y = arg("y", p.Type.IntS8)
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(sym("scala", "Byte", "+")),
      Nil,
      Some(selectT(x.named)),
      List(selectT(y.named)),
      p.Type.IntS32
    )
    val cleanup = p.Stmt.Var(named("sum", p.Type.IntS32), Some(invoke))
    val e       = entry(args = List(x, y), body = List(raise(p.Term.IntS32Const(1), "int", List(cleanup))))

    val out = Intrinsify(program(e), NoopLog).entry.body
    assertEquals(out.size, 1)
    val lowered = out.head.asInstanceOf[p.Stmt.Raise].cleanup
    assert(lowered.size > 1, s"expected promotion temporaries in cleanup, got ${lowered.map(_.repr)}")
    assert(lowered.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i }).isEmpty)
  }
}
