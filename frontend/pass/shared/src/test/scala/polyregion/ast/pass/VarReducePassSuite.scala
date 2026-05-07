package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class VarReducePassSuite extends munit.FunSuite {

  // Spec: an alias chain `var a = expr; var b = a; ... use(b)` should collapse so all uses of `b`
  // resolve to `a` and the redundant alias `b` disappears (or becomes a no-op). The original
  // value source is preserved. The pass is a fixed-point — running it again is a no-op.

  test("trivial alias is collapsed: var b = a; uses(b) -> uses(a)") {
    val a   = named("a")
    val b   = named("b")
    val ret = p.Stmt.Return(select(b))
    val body = List(
      p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(1)))),
      p.Stmt.Var(b, Some(select(a))),
      ret
    )
    val out = VarReducePass(program(entry(body = body, args = Nil).copy(rtn = p.Type.IntS32)), NoopLog)
    val refsToB = out.entry.body.flatMap(_.collectWhere[p.Term] {
      case s: p.Term.Select if s.root.symbol == b.symbol => s
    })
    assert(refsToB.isEmpty, s"alias `b` should have been collapsed away, but found refs: ${refsToB.map(_.repr)}")
  }

  test("unused alias to a Select is dropped: var b = a; (no uses)") {
    val a = named("a")
    val b = named("b")
    val body = List(
      p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(1)))),
      p.Stmt.Var(b, Some(select(a))),
      p.Stmt.Return(select(a))
    )
    val out      = VarReducePass(program(entry(body = body, args = Nil).copy(rtn = p.Type.IntS32)), NoopLog)
    val varDecls = out.entry.body.collect { case v: p.Stmt.Var => v.name }
    assert(!varDecls.contains(b), s"unused alias `b` should have been dropped, body: ${out.entry.body.map(_.repr)}")
  }

  test("running pass twice is idempotent") {
    val a = named("a")
    val b = named("b")
    val body = List(
      p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(1)))),
      p.Stmt.Var(b, Some(select(a))),
      p.Stmt.Return(select(b))
    )
    val once  = VarReducePass(program(entry(body = body).copy(rtn = p.Type.IntS32)), NoopLog)
    val twice = VarReducePass(once, NoopLog)
    assertEquals(once.entry.body, twice.entry.body)
  }
}
