package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class SpecialisationPassSuite extends munit.FunSuite {

  // Spec: each generic function (one with non-empty tpeVars) should be cloned per unique
  // applied type at the call sites; the original generic should be replaced/removed; remaining
  // call sites should reference the specialised functions with concrete types and no tpeArgs.

  test("non-generic program is unchanged") {
    val helper = fn("h", args = List(arg("a")), rtn = p.Type.IntS32, body = List(p.Stmt.Return(select("a"))))
    val prog   = program(entry(), List(helper))
    val out    = SpecialisationPass(prog, NoopLog)
    assertEquals(out.functions.map(_.name), List(helper.name))
  }

  test("generic function called with one concrete type produces a specialised function") {
    val tArg = arg("a", p.Type.Var("T"))
    val generic = fn(
      "id",
      args = List(tArg),
      rtn = p.Type.Var("T"),
      body = List(p.Stmt.Return(select(tArg.named))),
      tpeVars = List("T")
    )
    val callSite =
      p.Expr.Invoke(generic.name, List(p.Type.IntS32), None, List(p.Term.IntS32Const(1)), p.Type.IntS32)
    val e   = entry(body = List(p.Stmt.Var(named("r"), Some(callSite)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = SpecialisationPass(program(e, List(generic)), NoopLog)

    val genericLeft = out.functions.exists(f => f.name == generic.name && f.tpeVars.nonEmpty)
    assert(!genericLeft, s"generic should be removed; remaining: ${out.functions.map(f => f.name -> f.tpeVars)}")

    val specialised = out.functions.filter(_.tpeVars.isEmpty)
    assert(specialised.nonEmpty, "expected at least one specialised function with no tpeVars")
  }
}
