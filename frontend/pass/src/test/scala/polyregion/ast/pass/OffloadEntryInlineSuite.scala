package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class OffloadEntryInlineSuite extends munit.FunSuite {

  private def call(name: String): p.Expr =
    p.Expr.Invoke(p.Type.FnRef(sym(name)), Nil, None, Nil, p.Type.IntS32)

  private def helper(name: String, value: Int): p.Function =
    fn(
      name,
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(value)))),
      visibility = p.Function.Visibility.Internal
    )

  private def offload(name: String, callee: String): p.Function =
    fn(
      name,
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(call(callee))),
      convention = p.CallConvention.OffloadEntry
    )

  test("each offload entry is inlined from its own reachable call graph") {
    val first  = offload("first", "first.helper")
    val second = offload("second", "second.helper")
    val in = program(
      entry(),
      List(first, second, helper("first.helper", 11), helper("second.helper", 22), helper("orphan", 33))
    )

    val out       = OffloadEntryInline(in, NoopLog)
    val byName    = out.functions.map(f => f.name.fqcn -> f).toMap
    val firstOut  = byName("first")
    val secondOut = byName("second")

    assert(firstOut.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }.isEmpty)
    assert(secondOut.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }.isEmpty)
    assert(firstOut.collectWhere[p.Term] { case p.Term.IntS32Const(value) => value }.contains(11))
    assert(secondOut.collectWhere[p.Term] { case p.Term.IntS32Const(value) => value }.contains(22))
    assertEquals(out.functions.map(_.name.fqcn).toSet, in.functions.map(_.name.fqcn).toSet)
  }
}
