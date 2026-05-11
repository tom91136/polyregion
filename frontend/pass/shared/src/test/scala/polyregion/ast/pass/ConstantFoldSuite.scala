package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class ConstantFoldSuite extends munit.FunSuite {

  private def fold(body: List[p.Stmt], rtn: p.Type = p.Type.Unit0): List[p.Stmt] =
    ConstantFold(program(entry(body = body).copy(rtn = rtn)), NoopLog).entry.body

  private def aliasOf(s: p.Stmt): Option[p.Term] = s match {
    case p.Stmt.Var(_, Some(p.Expr.Alias(t)), _) => Some(t)
    case _                                       => None
  }

  test("fold i32(100) - i32(1) -> i32(99)") {
    val v = named("v", p.Type.IntS32)
    val out = fold(
      List(
        p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.Sub(p.Term.IntS32Const(100), p.Term.IntS32Const(1), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS32
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS32Const(99)), actual)
  }

  test("fold mixed binary: (a + b) * c when a,b,c are all const") {
    val a = named("a", p.Type.IntS64)
    val b = named("b", p.Type.IntS64)
    val c = named("c", p.Type.IntS64)
    val r = named("r", p.Type.IntS64)
    val out = fold(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS64Const(2L)))),
        p.Stmt.Var(b, Some(p.Expr.Alias(p.Term.IntS64Const(3L)))),
        p.Stmt.Var(c, Some(p.Expr.Alias(p.Term.IntS64Const(5L)))),
        p.Stmt.Var(r, Some(p.Expr.IntrOp(p.Intr.Mul(p.Term.IntS64Const(0), selectT(c), p.Type.IntS64)))),
        p.Stmt.Return(p.Expr.Alias(selectT(r)))
      ),
      rtn = p.Type.IntS64
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == r => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS64Const(0)), actual)
  }

  test("propagate val const through subsequent IntrOp") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val out = fold(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10)))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(5), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      rtn = p.Type.IntS32
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == b => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS32Const(15)), actual)
  }

  test("integer divide-by-zero is preserved") {
    val v        = named("v", p.Type.IntS32)
    val expected = p.Expr.IntrOp(p.Intr.Div(p.Term.IntS32Const(10), p.Term.IntS32Const(0), p.Type.IntS32))
    val out = fold(
      List(p.Stmt.Var(v, Some(expected)), p.Stmt.Return(p.Expr.Alias(selectT(v)))),
      rtn = p.Type.IntS32
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s.expr }
    assertEquals(Some(Some(expected)), actual)
  }

  test("fold float arithmetic") {
    val v = named("v", p.Type.Float64)
    val out = fold(
      List(
        p.Stmt
          .Var(v, Some(p.Expr.IntrOp(p.Intr.Mul(p.Term.Float64Const(2.5), p.Term.Float64Const(4.0), p.Type.Float64)))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.Float64
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.Float64Const(10.0)), actual)
  }

  test("fold logical comparisons") {
    val v = named("v", p.Type.Bool1)
    val out = fold(
      List(
        p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.LogicLt(p.Term.IntS32Const(3), p.Term.IntS32Const(5))))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.Bool1
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.Bool1Const(true)), actual)
  }

  test("fold cast IntS32 -> IntS64") {
    val v = named("v", p.Type.IntS64)
    val out = fold(
      List(
        p.Stmt.Var(v, Some(p.Expr.Cast(p.Term.IntS32Const(42), p.Type.IntS64))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS64
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS64Const(42L)), actual)
  }

  test("fold cast IntS32 -> Float64") {
    val v = named("v", p.Type.Float64)
    val out = fold(
      List(
        p.Stmt.Var(v, Some(p.Expr.Cast(p.Term.IntS32Const(7), p.Type.Float64))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.Float64
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.Float64Const(7.0)), actual)
  }

  test("Cond(true, t, f) -> emit t") {
    val v = named("v", p.Type.IntS32)
    val out = fold(
      List(
        p.Stmt.Cond(
          p.Term.Bool1Const(true),
          List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(2)))))
        ),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS32
    )
    val conds = out.collect { case c: p.Stmt.Cond => c }
    assert(conds.isEmpty, s"if(true) should have been folded away, got: ${out.map(_.repr).mkString("\n")}")
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS32Const(1)), actual)
  }

  test("Cond(false, t, f) -> emit f") {
    val v = named("v", p.Type.IntS32)
    val out = fold(
      List(
        p.Stmt.Cond(
          p.Term.Bool1Const(false),
          List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(2)))))
        ),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS32
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == v => s }.flatMap(aliasOf)
    assertEquals(Some(p.Term.IntS32Const(2)), actual)
  }

  test("While(false, _) -> drop") {
    val out = fold(
      List(
        p.Stmt.While(p.Term.Bool1Const(false), List(p.Stmt.Break)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val whiles = out.collect { case w: p.Stmt.While => w }
    assert(whiles.isEmpty, s"while(false) should have been dropped, got: ${out.map(_.repr).mkString("\n")}")
  }

  test("ForRange with empty iteration is dropped") {
    val out = fold(
      List(
        p.Stmt.ForRange(
          named("i", p.Type.IntS64),
          p.Term.IntS64Const(10),
          p.Term.IntS64Const(10),
          p.Term.IntS64Const(1),
          List(p.Stmt.Break)
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val fors = out.collect { case f: p.Stmt.ForRange => f }
    assert(fors.isEmpty, s"empty ForRange should have been dropped, got: ${out.map(_.repr).mkString("\n")}")
  }

  test("mutable val is not propagated") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val out = fold(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10))), isMutable = true),
        p.Stmt.Mut(selectT(a), p.Expr.Alias(p.Term.IntS32Const(20))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(5), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      rtn = p.Type.IntS32
    )
    val actual = out.collectFirst { case s: p.Stmt.Var if s.name == b => s }.flatMap(aliasOf)
    assert(actual.isEmpty, s"mutable a should not have been folded into b; got: ${out.map(_.repr).mkString("\n")}")
  }

  test("running pass twice is idempotent") {
    val v = named("v", p.Type.IntS32)
    val body = List(
      p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.Add(p.Term.IntS32Const(1), p.Term.IntS32Const(2), p.Type.IntS32)))),
      p.Stmt.Return(p.Expr.Alias(selectT(v)))
    )
    val once  = ConstantFold(program(entry(body = body).copy(rtn = p.Type.IntS32)), NoopLog)
    val twice = ConstantFold(once, NoopLog)
    assertEquals(once.entry.body, twice.entry.body)
  }
}
