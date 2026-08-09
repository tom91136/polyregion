package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class VerifySuite extends munit.FunSuite {

  // Spec: Verify collects per-function error messages. A well-formed program has no errors;
  // a program with an undeclared identifier reference has at least one error against the function
  // that contains the bad reference.

  test("well-formed entry yields no errors") {
    val a = arg("a")
    val e = entry(args = List(a), body = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))).copy(rtn = p.Type.Unit0)
    val errs = Verify(program(e), NoopLog, verifyFunction = true)
    assert(errs.forall(_._2.isEmpty), s"expected no errors, got: $errs")
  }

  test("handler source identity is non-empty") {
    val invalid = p.Handler(Some(p.ExceptionKind(p.Type.IntS32, "  ")), None, Nil)
    val errs    = errors(List(p.Stmt.Try(Nil, List(invalid), Nil)))
    assert(errs.exists(_.contains("source identity is empty")), errs.mkString("\n"))
  }

  test("handler binders agree with the caught type") {
    val catchAllBinder = handler(None, Some(named("all")), Nil, None)
    val wrongBinder = handler(
      Some(p.Type.IntS32),
      Some(named("caught", p.Type.IntS64)),
      Nil,
      Some("int")
    )
    val errs = errors(List(p.Stmt.Try(Nil, List(catchAllBinder, wrongBinder), Nil)))
    assert(errs.exists(_.contains("catch-all handler cannot bind")), errs.mkString("\n"))
    assert(errs.exists(_.contains("binder type")), errs.mkString("\n"))
  }

  test("raise descriptor agrees with the raised value") {
    val invalid = p.Stmt.Raise(
      p.Term.IntS32Const(1),
      p.ExceptionKind(p.Type.IntS64, "int"),
      Nil
    )
    val errs = errors(List(invalid))
    assert(errs.exists(_.contains("does not match value type")), errs.mkString("\n"))
  }

  test("reference to an undeclared name produces an error against that function") {
    val undeclared = named("ghost")
    val e = entry(body =
      List(p.Stmt.Var(named("r"), Some(select(undeclared))), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
    )
    val errs      = Verify(program(e), NoopLog, verifyFunction = true)
    val entryErrs = errs.collectFirst { case (f, es) if f.name == e.name => es }.getOrElse(Nil)
    assert(entryErrs.nonEmpty, s"expected at least one error for entry, got: $errs")
  }

  private def errors(body: List[p.Stmt]): List[String] =
    Verify(
      program(entry(body = body :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))),
      NoopLog,
      verifyFunction = true
    )
      .flatMap(_._2)

  test("a try-body local is not visible in a handler") {
    val local = named("tryLocal")
    val errs = errors(
      List(
        p.Stmt.Try(
          List(p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          List(handler(None, None, List(p.Stmt.Var(named("use"), Some(select(local)))), None)),
          Nil
        )
      )
    )
    assert(errs.exists(_.contains("unseen variable tryLocal")), errs.mkString("\n"))
  }

  test("a handler binder is visible only in its own handler") {
    val caught = named("caught")
    val errs = errors(
      List(
        p.Stmt.Try(
          Nil,
          List(
            handler(Some(p.Type.IntS32), Some(caught), Nil, Some("int")),
            handler(None, None, List(p.Stmt.Var(named("use"), Some(select(caught)))), None)
          ),
          Nil
        )
      )
    )
    assert(errs.exists(_.contains("unseen variable caught")), errs.mkString("\n"))
  }

  test("a try-body local is not visible in finally") {
    val local = named("tryLocal")
    val errs = errors(
      List(
        p.Stmt.Try(
          List(p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          Nil,
          List(p.Stmt.Var(named("use"), Some(select(local))))
        )
      )
    )
    assert(errs.exists(_.contains("unseen variable tryLocal")), errs.mkString("\n"))
  }

  test("a raise cleanup local is not visible after the raise") {
    val local = named("cleanupLocal")
    val errs = errors(
      List(
        raise(
          p.Term.IntS32Const(1),
          "int",
          List(p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(1)))))
        ),
        p.Stmt.Var(named("use"), Some(select(local)))
      )
    )
    assert(errs.exists(_.contains("unseen variable cleanupLocal")), errs.mkString("\n"))
  }
}
