package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class UnitExprElisionPassSuite extends munit.FunSuite {

  // Spec: pure Unit-typed bindings/expressions add no observable behaviour and should be elided.
  // Side-effect-bearing expressions (Invoke, IntrOp/MathOp/SpecOp with effects) of Unit type
  // should be left in place.

  test("pure Unit var binding is elided") {
    val e = entry(body =
      List(p.Stmt.Var(named("u", p.Type.Unit0), Some(p.Expr.Alias(p.Term.Unit0Const))), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
    )
    val out  = UnitExprElisionPass(program(e), NoopLog)
    val vars = out.entry.body.collect { case v: p.Stmt.Var if v.name.tpe == p.Type.Unit0 => v }
    assert(vars.isEmpty, s"pure Unit Stmt.Var should be elided, got: ${vars.map(_.name.symbol)}")
  }

  test("Unit-typed Invoke is preserved (potential side effect)") {
    val invoke = p.Expr.Invoke(sym("doIt"), Nil, None, Nil, p.Type.Unit0)
    val e   = entry(body = List(p.Stmt.Var(named("u", p.Type.Unit0), Some(invoke)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = UnitExprElisionPass(program(e), NoopLog)
    val invokes = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokes.nonEmpty, "Unit-typed Invoke (side-effecting) must not be elided")
  }
}
