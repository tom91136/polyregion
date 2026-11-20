package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class ZeroAllocSuite extends munit.FunSuite {

  private val i32    = p.Type.IntS32
  private val g      = p.Type.Space.Global
  private val ptrI32 = p.Type.Ptr(i32, g)
  private val pairT  = p.Type.Struct(sym("Pair"), Nil)
  private val pair   = p.StructDef(sym("Pair"), Nil, List(named("a", i32), named("b", ptrI32)), Nil)

  private def run(body: List[p.Stmt], args: List[p.Arg] = Nil): p.Function =
    ZeroAlloc(program(entry(args = args, body = body), Nil, List(pair)), NoopLog).entry.required

  private def whiles(f: p.Function): List[p.Stmt.While] = f.collectWhere[p.Stmt] { case w: p.Stmt.While => w }

  private def updates(f: p.Function): List[p.Stmt.Update] = f.collectWhere[p.Stmt] { case u: p.Stmt.Update => u }

  test("allocation of a scalar is zero filled right after the allocation") {
    val n = arg("n", i32)
    val a = named("a", ptrI32)
    val out = run(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(i32, selectT(n.named), g, p.Region.Opaque))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      List(n)
    )
    assertEquals(whiles(out).size, 1)
    assertEquals(updates(out).map(u => u.lhs.root -> u.value), List(a -> p.Term.IntS32Const(0)))
    assert(out.body.head.isInstanceOf[p.Stmt.Var])
    assertEquals(out.body.indexWhere(_.isInstanceOf[p.Stmt.While]), 3)
  }

  test("allocation of pointers is filled with null pointers") {
    val a = named("a", p.Type.Ptr(ptrI32, g))
    val out = run(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(ptrI32, p.Term.IntS64Const(4), g, p.Region.Opaque))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    assertEquals(updates(out).map(_.value), List(p.Term.NullPtrConst(i32, g, p.Region.Opaque)))
    assertEquals(
      out.collectWhere[p.Term] { case p.Term.Select(r, _, _) if r.symbol.startsWith("_zero_") => r.tpe }.distinct,
      List(p.Type.IntS64, p.Type.Bool1)
    )
  }

  test("allocation of structs is filled with a zeroed struct") {
    val a = named("a", p.Type.Ptr(pairT, g))
    val out = run(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(pairT, p.Term.IntS32Const(2), g, p.Region.Opaque))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val zeroed = out.collectWhere[p.Stmt] {
      case p.Stmt.Mut(p.Term.Select(r, List(p.PathStep.Field(f)), _), p.Expr.Alias(z)) => f -> z
    }
    assertEquals(zeroed, List("a" -> p.Term.IntS32Const(0), "b" -> p.Term.NullPtrConst(i32, g, p.Region.Opaque)))
  }

  test("allocation assigned to an existing pointer is filled through that pointer") {
    val a = named("a", ptrI32)
    val out = run(
      List(
        p.Stmt.Var(a, None),
        p.Stmt.Mut(selectT(a), p.Expr.Alloc(i32, p.Term.IntS32Const(3), g, p.Region.Opaque)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    assertEquals(updates(out).map(_.lhs.root), List(a))
  }

  test("group and stack storage is not zero filled") {
    val a = named("a", p.Type.Ptr(i32, p.Type.Space.Local))
    val out = run(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(i32, p.Term.IntS32Const(4), p.Type.Space.Local, p.Region.Rooted(a)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    assertEquals(whiles(out), Nil)
  }

  test("escape analysis then zero fill leaves only the allocations that are not replaced") {
    val x        = named("x", i32)
    val replaced = named("r", ptrI32)
    val kept     = named("k", ptrI32)
    val n        = arg("n", i32)
    val body = List(
      p.Stmt.Var(replaced, Some(p.Expr.Alloc(i32, p.Term.IntS32Const(2), g, p.Region.Opaque))),
      p.Stmt.Var(kept, Some(p.Expr.Alloc(i32, selectT(n.named), g, p.Region.Opaque))),
      p.Stmt.Update(selectT(kept), selectT(n.named), p.Term.IntS32Const(1)),
      p.Stmt.Var(x, Some(p.Expr.Index(selectT(replaced), p.Term.IntS32Const(1), i32))),
      p.Stmt.Return(p.Expr.Alias(selectT(x)))
    )
    val in  = program(withRtn(entry(args = List(n), body = body), i32), Nil, Nil)
    val out = ZeroAlloc(EscapeAnalysis(refuse = true)(in, NoopLog), NoopLog)
    assertEquals(out.entry.required.collectWhere[p.Expr] { case a: p.Expr.Alloc => a.size }, List(selectT(n.named)))
    assertEquals(whiles(out.entry.required).size, 1)
  }

  private def withRtn(e: p.Function, rtn: p.Type): p.Function = e.copy(decl = e.decl.copy(rtn = rtn))
}
