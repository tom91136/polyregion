package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class DeadFunctionEliminationSuite extends munit.FunSuite {

  private def callTo(name: String): p.Stmt =
    p.Stmt.Var(named(s"_$name"), Some(p.Expr.Invoke(p.Type.FnRef(sym(name)), Nil, None, Nil, p.Type.IntS32)))

  private def internal(name: String, body: List[p.Stmt] = Nil) =
    fn(
      name,
      body = body ++ List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      visibility = p.Function.Visibility.Internal
    )

  private def exported(name: String, body: List[p.Stmt] = Nil) =
    fn(
      name,
      body = body ++ List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      visibility = p.Function.Visibility.Exported
    )

  private def kept(fns: List[p.Function]): Set[String] =
    DeadFunctionElimination(program(entry(), functions = fns), NoopLog).functions.map(_.name.repr).toSet

  test("function no export reaches is dropped") {
    val out = kept(List(exported("a", List(callTo("shared"))), internal("shared"), internal("orphan")))
    assertEquals(out, Set("a", "shared"))
  }

  test("callee shared by two exports survives, and each export alone keeps it") {
    val all = List(
      exported("a", List(callTo("shared"))),
      exported("b", List(callTo("shared"))),
      internal("shared"),
      internal("orphan")
    )
    assertEquals(kept(all), Set("a", "b", "shared"))

    val onlyA = all.map(f => if (f.name.repr == "b") f.copy(visibility = p.Function.Visibility.Internal) else f)
    assertEquals(kept(onlyA), Set("a", "shared"))
  }

  test("transitive chain is followed") {
    val out = kept(
      List(
        exported("a", List(callTo("mid"))),
        internal("mid", List(callTo("leaf"))),
        internal("leaf"),
        internal("orphan")
      )
    )
    assertEquals(out, Set("a", "mid", "leaf"))
  }

  test("recursive cycle terminates") {
    val out = kept(List(exported("a", List(callTo("b"))), internal("b", List(callTo("a"))), internal("orphan")))
    assertEquals(out, Set("a", "b"))
  }

  test("every overload of a reached name survives") {
    val all = List(
      exported("a", List(callTo("shared"))),
      internal("shared").modifyDecl(_.copy(args = List(arg("x", p.Type.IntS32)))),
      internal("shared").modifyDecl(_.copy(args = List(arg("x", p.Type.Float32)))),
      internal("orphan")
    )
    val out = DeadFunctionElimination(program(entry(), functions = all), NoopLog).functions
    assertEquals(out.count(_.name.repr == "shared"), 2)
    assert(!out.exists(_.name.repr == "orphan"))
  }

  test("no exports keeps nothing") {
    assertEquals(kept(List(internal("a"), internal("b"))), Set.empty[String])
  }
}
