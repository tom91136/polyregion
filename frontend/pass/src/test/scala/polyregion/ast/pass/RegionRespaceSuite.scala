package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class RegionRespaceSuite extends munit.FunSuite {

  private def ptr(s: p.Type.Space) = p.Type.Ptr(p.Type.IntS32, s)

  private def ptrSpacesOf(f: p.Function, sym: String): Set[p.Type.Space] =
    (f.collectAll[p.Stmt].collect { case p.Stmt.Var(n, _, _) if n.symbol == sym => n.tpe } :::
      f.collectAll[p.Term].collect { case p.Term.Select(n, _, _) if n.symbol == sym => n.tpe }).collect {
      case p.Type.Ptr(_, s) => s
    }.toSet

  private def refToSpaces(f: p.Function): Set[p.Type.Space] =
    f.collectAll[p.Expr].collect { case p.Expr.RefTo(_, _, _, s, _) => s }.toSet

  test("a Global pointer rooted at a Local resource is re-stamped Local (decl, uses, and the RefTo)") {
    val local = named("local", ptr(p.Type.Space.Local))
    val s     = named("s", ptr(p.Type.Space.Global))
    val refTo =
      p.Expr.RefTo(selectT(local), Some(p.Term.IntS64Const(1)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
    val use =
      p.Stmt.Var(named("u", p.Type.IntS32), Some(p.Expr.Index(selectT(s), p.Term.IntS64Const(0), p.Type.IntS32)))
    val e   = entry(body = List(p.Stmt.Var(s, Some(refTo)), use, p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = RegionRespace(program(e), NoopLog)
    assertEquals(ptrSpacesOf(out.entry, "s"), Set[p.Type.Space](p.Type.Space.Local))
    assertEquals(refToSpaces(out.entry), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("a pointer already in its root's space is left untouched") {
    val g = named("g", ptr(p.Type.Space.Global))
    val s = named("s", ptr(p.Type.Space.Global))
    val refTo =
      p.Expr.RefTo(selectT(g), Some(p.Term.IntS64Const(1)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
    val e   = entry(body = List(p.Stmt.Var(s, Some(refTo)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = RegionRespace(program(e), NoopLog)
    assertEquals(ptrSpacesOf(out.entry, "s"), Set[p.Type.Space](p.Type.Space.Global))
  }
}
