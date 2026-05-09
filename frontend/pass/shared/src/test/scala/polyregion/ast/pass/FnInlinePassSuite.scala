package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class FnInlinePassSuite extends munit.FunSuite {

  // Spec: every call from the entry function to a known helper should be inlined into the
  // entry body. After the pass, no Invoke targeting an inlinable helper should remain reachable
  // from entry, and the helper's effect/return should appear in entry's body.

  test("empty entry with no invokes is unchanged") {
    val prog = program(entry())
    val out  = FnInlinePass(prog, NoopLog)
    assertEquals(out.entry.body.collect { case s: p.Stmt.Return => s }.size, 1)
    assertEquals(out.functions, Nil)
  }

  test("entry call to a single-return helper has no Invoke remaining") {
    val xArg = arg("x")
    val helper = fn(
      "helper",
      args = List(xArg),
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(select(xArg.named)))
    )
    val invokeExpr = p.Expr.Invoke(helper.name, Nil, None, List(p.Term.IntS32Const(7)), p.Type.IntS32)
    val e = entry(body = List(p.Stmt.Var(named("r"), Some(invokeExpr)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))

    val out = FnInlinePass(program(e, List(helper)), NoopLog)

    val invokesLeft = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokesLeft.isEmpty, s"expected no Invoke remaining in entry, got: ${invokesLeft.map(_.repr)}")
  }
}
