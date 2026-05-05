package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class VerifyPassSuite extends munit.FunSuite {

  // Spec: VerifyPass collects per-function error messages. A well-formed program has no errors;
  // a program with an undeclared identifier reference has at least one error against the function
  // that contains the bad reference.

  test("well-formed entry yields no errors") {
    val a    = arg("a")
    val e    = entry(args = List(a), body = List(p.Stmt.Return(p.Expr.Unit0Const))).copy(rtn = p.Type.Unit0)
    val errs = VerifyPass(program(e), NoopLog, verifyFunction = true)
    assert(errs.forall(_._2.isEmpty), s"expected no errors, got: $errs")
  }

  test("reference to an undeclared name produces an error against that function") {
    val undeclared = named("ghost")
    val e    = entry(body = List(p.Stmt.Var(named("r"), Some(select(undeclared))), p.Stmt.Return(p.Expr.Unit0Const)))
    val errs = VerifyPass(program(e), NoopLog, verifyFunction = true)
    val entryErrs = errs.collectFirst { case (f, es) if f.name == e.name => es }.getOrElse(Nil)
    assert(entryErrs.nonEmpty, s"expected at least one error for entry, got: $errs")
  }
}
