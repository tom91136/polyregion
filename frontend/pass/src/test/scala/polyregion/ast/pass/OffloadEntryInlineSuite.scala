package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
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

    val out       = OffloadEntryInline()(in, NoopLog)
    val byName    = out.functions.map(f => f.name.fqcn -> f).toMap
    val firstOut  = byName("first")
    val secondOut = byName("second")

    assert(firstOut.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }.isEmpty)
    assert(secondOut.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }.isEmpty)
    assert(firstOut.collectWhere[p.Term] { case p.Term.IntS32Const(value) => value }.contains(11))
    assert(secondOut.collectWhere[p.Term] { case p.Term.IntS32Const(value) => value }.contains(22))
    assertEquals(out.functions.map(_.name.fqcn).toSet, in.functions.map(_.name.fqcn).toSet)
  }

  test("a recursive callee is lowered before its offload entry is inlined") {
    val i32                                           = p.Type.IntS32
    def let(name: String, e: p.Expr, t: p.Type = i32) = p.Stmt.Var(named(name, t), Some(e), false)
    def sum(n: p.Term): p.Expr                        = p.Expr.Invoke(p.Type.FnRef(sym("sum")), Nil, None, List(n), i32)
    val recursive = fn(
      "sum",
      List(arg("n")),
      i32,
      List(
        let("done", p.Expr.IntrOp(p.Intr.LogicLte(selectT("n"), p.Term.IntS32Const(0))), p.Type.Bool1),
        p.Stmt.Cond(selectT("done", p.Type.Bool1), List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(0)))), Nil),
        let("m", p.Expr.IntrOp(p.Intr.Sub(selectT("n"), p.Term.IntS32Const(1), i32))),
        let("r", sum(selectT("m"))),
        let("s", p.Expr.IntrOp(p.Intr.Add(selectT("r"), selectT("n"), i32))),
        p.Stmt.Return(p.Expr.Alias(selectT("s")))
      ),
      visibility = p.Function.Visibility.Internal
    )
    val kernel = fn(
      "kernel",
      rtn = i32,
      body = List(p.Stmt.Return(sum(p.Term.IntS32Const(4)))),
      convention = p.CallConvention.OffloadEntry
    )
    val in = program(entry(), List(kernel, recursive))

    val out       = OffloadEntryInline(maxDepth = 8)(in, NoopLog)
    val kernelOut = out.functions.find(_.name == sym("kernel")).get

    assert(kernelOut.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }.isEmpty)
    assert(kernelOut.collectWhere[p.Type] { case p.Type.Arr(_, 9, _) => () }.nonEmpty)
    assert(out.defs.exists(_.members.exists(_.symbol == "#pc")))
    assertEquals(out.functions.find(_.name == sym("sum")), Some(recursive))
    assertEquals(Interpreter.Vm(out).call("kernel", Nil), V.I(10))
  }
}
