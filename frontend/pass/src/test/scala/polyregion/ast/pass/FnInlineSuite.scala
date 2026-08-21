package polyregion.ast.pass

import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class FnInlineSuite extends munit.FunSuite {

  // Spec: every call from the entry function to a known helper should be inlined into the
  // entry body. After the pass, no Invoke targeting an inlinable helper should remain reachable
  // from entry, and the helper's effect/return should appear in entry's body.

  test("empty entry with no invokes is unchanged") {
    val prog = program(entry())
    val out  = FnInline(prog, NoopLog)
    assertEquals(out.entry.body.collect { case s: p.Stmt.Return => s }.size, 1)
    assertEquals(out.functions, Nil)
  }

  test("entry call to a single-return helper has no Invoke remaining") {
    val xArg = arg("x")
    val helper = fn(
      "helper",
      args = List(xArg),
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(select(xArg.named)))
    )
    val invokeExpr = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(p.Term.IntS32Const(7)), p.Type.IntS32)
    val e = entry(body = List(p.Stmt.Var(named("r"), Some(invokeExpr)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))

    val out = FnInline(program(e, List(helper)), NoopLog)

    val invokesLeft = out.entry.body.flatMap(_.collectWhere[p.Expr] { case i: p.Expr.Invoke => i })
    assert(invokesLeft.isEmpty, s"expected no Invoke remaining in entry, got: ${invokesLeft.map(_.repr)}")
  }

  test("generic member calls take inline type arguments from pointer receivers") {
    val boxName   = sym("Box")
    val boxT      = p.Type.Struct(boxName, List(p.Type.Var("T")))
    val boxInt    = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxTPtr   = p.Type.Ptr(boxT, p.Type.Space.Private)
    val boxIntPtr = p.Type.Ptr(boxInt, p.Type.Space.Private)
    val helper = fn(
      "Box.apply",
      args = List(arg("value", p.Type.Var("T"))),
      rtn = p.Type.Var("T"),
      body = List(p.Stmt.Return(select("value", p.Type.Var("T")))),
      tpeVars = List("T")
    ).modifyDecl(_.copy(receiver = Some(arg("self", boxTPtr))))
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(helper.name),
      Nil,
      Some(p.Term.Poison(boxIntPtr)),
      List(p.Term.IntS32Const(7)),
      p.Type.IntS32
    )
    val in = program(
      entry(body = List(p.Stmt.Var(named("result"), Some(invoke)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))),
      List(helper)
    )

    val out = FnInline(in, NoopLog)

    assertEquals(out.entry.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }, Nil)
    assertEquals(out.entry.collectWhere[p.Term] { case x: p.Term.IntS32Const => x.value }, List(7))
  }

  test("pointer receiver arguments combine with compiler-supplied method type arguments") {
    val boxName   = sym("Box")
    val boxT      = p.Type.Struct(boxName, List(p.Type.Var("T")))
    val boxInt    = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxTPtr   = p.Type.Ptr(boxT, p.Type.Space.Private)
    val boxIntPtr = p.Type.Ptr(boxInt, p.Type.Space.Private)
    val helper = fn(
      "Box.convert",
      args = List(arg("value", p.Type.Var("U"))),
      rtn = p.Type.Var("U"),
      body = List(p.Stmt.Return(select("value", p.Type.Var("U")))),
      tpeVars = List("T", "U")
    ).modifyDecl(_.copy(receiver = Some(arg("self", boxTPtr))))
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(helper.name),
      List(p.Type.Float32),
      Some(p.Term.Poison(boxIntPtr)),
      List(p.Term.Float32Const(3.5f)),
      p.Type.Float32
    )
    val in = program(entry(body = List(p.Stmt.Return(invoke))), List(helper))

    val out = FnInline(in, NoopLog)

    assertEquals(out.entry.collectWhere[p.Expr] { case x: p.Expr.Invoke => x }, Nil)
    assertEquals(out.entry.collectWhere[p.Term] { case x: p.Term.Float32Const => x.value }, List(3.5f))
  }

  test("recursive helpers fail promptly with a deterministic diagnostic") {
    val helperName = sym("recursive.helper")
    val recursive  = p.Expr.Invoke(p.Type.FnRef(helperName), Nil, None, Nil, p.Type.Unit0)
    val helper     = fn(helperName.repr, body = List(p.Stmt.Return(recursive)))
    val invoke     = p.Expr.Invoke(p.Type.FnRef(helperName), Nil, None, Nil, p.Type.Unit0)

    val error = intercept[IllegalStateException] {
      FnInline(program(entry(body = List(p.Stmt.Return(invoke))), List(helper)), NoopLog)
    }

    assert(error.getMessage.contains("recursive call cycle"))
  }

  test("same-name overload delegation is not mistaken for recursion") {
    val name = sym("delegating.overload")
    val leaf = fn(
      name.repr,
      args = List(arg("value", p.Type.Float32)),
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(7))))
    )
    val delegating = fn(
      name.repr,
      args = List(arg("value", p.Type.IntS32)),
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(
            p.Type.FnRef(name),
            Nil,
            None,
            List(p.Term.Float32Const(1.0f)),
            p.Type.IntS32
          )
        )
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(name), Nil, None, List(p.Term.IntS32Const(1)), p.Type.IntS32)

    val out =
      FnInline(program(entry(body = List(p.Stmt.Var(named("result"), Some(invoke)))), List(leaf, delegating)), NoopLog)

    assertEquals(out.entry.collectAll[p.Expr].collect { case call: p.Expr.Invoke => call }, Nil)
    assertEquals(out.entry.collectAll[p.Term].collect { case value: p.Term.IntS32Const => value.value }, List(7))
  }

  test("mutable module captures use a fresh local for each inline") {
    val capture = arg("capture", p.Type.IntS32)
    val helper = fn(
      "mutable.capture",
      moduleCaptures = List(capture),
      body = List(
        p.Stmt.Mut(selectT(capture.named), p.Expr.Alias(p.Term.IntS32Const(2))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(selectT(capture.named)), p.Type.Unit0)
    val in = program(
      entry(
        moduleCaptures = List(capture),
        body = List(
          p.Stmt.Var(named("first", p.Type.Unit0), Some(invoke)),
          p.Stmt.Var(named("second", p.Type.Unit0), Some(invoke)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      List(helper)
    )

    val out    = FnInline(in, NoopLog)
    val locals = out.entry.body.collect { case p.Stmt.Var(name, _, true) => name }
    val roots  = out.entry.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(root, _, _), _) => root }
    assertEquals(locals.distinct.size, 2)
    assertEquals(roots.toSet, locals.toSet)
    assert(!locals.contains(capture.named))
  }

  test("constant return analysis preserves branch scope") {
    val local = named("branch.local", p.Type.IntS32)
    val helper = fn(
      "constant.scope",
      body = List(
        p.Stmt.Cond(
          p.Term.Bool1Const(true),
          List(p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          Nil
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)

    val out = FnInline(program(entry(body = List(p.Stmt.Return(invoke))), List(helper)), NoopLog)

    assertEquals(out.entry.body.collect { case cond: p.Stmt.Cond => cond }.size, 1)
  }

  test("a branch-local return value is rebound inside its scope") {
    val local = named("branch.result", p.Type.IntS32)
    val helper = fn(
      "branch.return",
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Cond(
          p.Term.Bool1Const(true),
          List(
            p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(1)))),
            p.Stmt.Return(select(local))
          ),
          Nil
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(2)))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.IntS32)
    val in = program(
      entry(
        body = List(
          p.Stmt.Var(named("result"), Some(invoke)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      List(helper)
    )

    val out = FnInline(in, NoopLog)

    assertEquals(Verify(out, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }

  test("inlining preserves type variables shadowed by callable binders") {
    val callable = p.Type.Exec(List("T"), List(p.Type.Var("T")), p.Type.Var("T"))
    val helper = fn(
      "callable.shadow",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T"),
      body = List(
        p.Stmt.Var(named("callback", callable), Some(p.Expr.Alias(p.Term.Poison(callable)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(helper.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val out = FnInline(program(entry(body = List(p.Stmt.Return(invoke))), List(helper)), NoopLog)

    assertEquals(out.entry.collectAll[p.Type].collect { case exec: p.Type.Exec => exec }.distinct, List(callable))
  }

  test("inlining drops a poison return after nested constant branches return") {
    val helper = fn(
      "helper",
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Cond(
          p.Term.Bool1Const(true),
          List(
            p.Stmt.Cond(
              p.Term.Bool1Const(true),
              List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(7)))),
              Nil
            )
          ),
          Nil
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Poison(p.Type.IntS32)))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.IntS32)
    val in = program(
      entry(body = List(p.Stmt.Var(named("result"), Some(invoke)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))),
      List(helper)
    )

    val out = FnInline(in, NoopLog)
    assertEquals(out.entry.collectWhere[p.Term] { case x: p.Term.Poison => x }, Nil)
    assertEquals(out.entry.collectWhere[p.Term] { case x: p.Term.IntS32Const => x.value }, List(7))
  }

  private def runTryReturn(helperBody: List[p.Stmt]): (Long, Long) = {
    val i32  = p.Type.IntS32
    val g    = p.Type.Space.Global
    val out  = named("out", p.Type.Ptr(i32, g))
    val hOut = arg("hOut", out.tpe)
    val helper = fn(
      "helper",
      args = List(hOut),
      rtn = i32,
      body = helperBody.map(_.modifyAll[p.Term] {
        case p.Term.Select(root, steps, tpe) if root.symbol == out.symbol =>
          p.Term.Select(hOut.named, steps, tpe)
        case x => x
      })
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(selectT(out)), i32)
    val result = named("result", i32)
    val e = entry(
      args = List(p.Arg(out)),
      body = List(
        p.Stmt.Var(result, Some(invoke)),
        p.Stmt.Update(selectT(out), p.Term.IntS32Const(1), selectT(result)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    def run(prog: p.Program): Long = {
      val vm   = Interpreter.Vm(prog)
      val cell = vm.alloc(8L)
      vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(cell)))
      def intAt(addr: Long) = vm.load(addr, i32) match { case V.I(v) => v; case _ => Long.MinValue }
      intAt(cell) * 100 + intAt(cell + 4L)
    }
    val in = program(e, List(helper))
    run(in) -> run(FnInline(in, NoopLog))
  }

  test("inlining preserves return from a try body and runs finally") {
    val out = named("out", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val body = List(
      p.Stmt.Try(
        List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(3)))),
        Nil,
        List(p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), p.Term.IntS32Const(7)))
      ),
      p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(9)))
    )
    val (direct, inlined) = runTryReturn(body)
    assertEquals(direct, 703L)
    assertEquals(inlined, direct)
  }

  test("inlining preserves return from a handler and runs finally") {
    val out = named("out", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val body = List(
      p.Stmt.Try(
        List(raise(p.Term.IntS32Const(1), "int")),
        List(
          handler(Some(p.Type.IntS32), None, List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(3)))), Some("int"))
        ),
        List(p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), p.Term.IntS32Const(7)))
      ),
      p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(9)))
    )
    val (direct, inlined) = runTryReturn(body)
    assertEquals(direct, 703L)
    assertEquals(inlined, direct)
  }

  test("inlining preserves a return from finally overriding a pending return") {
    val body = List(
      p.Stmt.Try(
        List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(3)))),
        Nil,
        List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(5))))
      ),
      p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(9)))
    )
    val (direct, inlined) = runTryReturn(body)
    assertEquals(direct, 5L)
    assertEquals(inlined, direct)
  }

  test("a pending inlined return does not suppress a non-returning finalizer path") {
    val out  = named("out", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val cond = named("cond", p.Type.Bool1)
    val body = List(
      p.Stmt.Try(
        List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(3)))),
        Nil,
        List(
          p.Stmt.Var(cond, Some(p.Expr.Alias(p.Term.Bool1Const(false)))),
          p.Stmt.Cond(
            selectT(cond),
            List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(5)))),
            Nil
          ),
          p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), p.Term.IntS32Const(7))
        )
      ),
      p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(9)))
    )
    val (direct, inlined) = runTryReturn(body)
    assertEquals(direct, 703L)
    assertEquals(inlined, direct)
  }

  test("a sole handler return remains inside the handler scope after inlining") {
    val i32    = p.Type.IntS32
    val caught = named("caught", i32)
    val helper = fn(
      "helper",
      rtn = i32,
      body = List(
        p.Stmt.Try(
          List(raise(p.Term.IntS32Const(3), "int")),
          List(handler(Some(i32), Some(caught), List(p.Stmt.Return(p.Expr.Alias(selectT(caught)))), Some("int"))),
          Nil
        )
      )
    )
    val result = named("result", i32)
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, i32)
    val in = program(
      entry(body = List(p.Stmt.Var(result, Some(invoke)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))),
      List(helper)
    )

    val out    = FnInline(in, NoopLog)
    val errors = Verify(out, NoopLog, verifyFunction = true).flatMap(_._2)
    assertEquals(errors, Nil)
  }

  test("inlining alpha-renames a range induction variable with its uses") {
    val i = named("i", p.Type.IntS32)
    val helper = fn(
      "helper",
      body = List(
        p.Stmt.ForRange(
          i,
          p.Term.IntS32Const(0),
          p.Term.IntS32Const(1),
          p.Term.IntS32Const(1),
          List(p.Stmt.Var(named("x"), Some(p.Expr.Alias(selectT(i)))))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
    val in = program(
      entry(
        body = List(
          p.Stmt.Var(named("r", p.Type.Unit0), Some(invoke)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      List(helper)
    )

    val out    = FnInline(in, NoopLog)
    val errors = Verify(out, NoopLog, verifyFunction = true).flatMap(_._2)
    assertEquals(errors, Nil)
    val loop  = out.entry.collectFirst_[p.Stmt] { case x: p.Stmt.ForRange => x }.getOrElse(fail("missing inlined loop"))
    val roots = loop.body.flatMap(_.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root })
    assertNotEquals(loop.induction.symbol, i.symbol)
    assert(roots.contains(loop.induction), s"expected ${loop.induction.symbol} in ${roots.map(_.symbol)}")
    assert(!roots.contains(i), s"stale induction variable ${i.symbol} remains")
  }

  test("inlining binds a mutable by-value parameter to a local") {
    val value = arg("value", p.Type.IntS32)
    val helper = fn(
      "helper",
      args = List(value),
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Mut(selectT(value.named), p.Expr.Alias(p.Term.IntS32Const(2))),
        p.Stmt.Return(p.Expr.Alias(selectT(value.named)))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(p.Term.IntS32Const(1)), p.Type.IntS32)
    val in = program(
      entry(body = List(p.Stmt.Var(named("result"), Some(invoke)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))),
      List(helper)
    )

    val out = FnInline(in, NoopLog)
    assertEquals(Verify(out, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }

  test("inlining preserves an early return from a loop") {
    val i32   = p.Type.IntS32
    val limit = arg("limit", i32)
    val i     = named("i", i32)
    val helper = fn(
      "helper",
      args = List(limit),
      rtn = i32,
      body = List(
        p.Stmt.ForRange(
          i,
          p.Term.IntS32Const(0),
          p.Term.Select(limit.named, Nil, i32),
          p.Term.IntS32Const(1),
          List(
            p.Stmt.Var(
              named("found", p.Type.Bool1),
              Some(p.Expr.IntrOp(p.Intr.LogicEq(p.Term.Select(i, Nil, i32), p.Term.IntS32Const(2))))
            ),
            p.Stmt.Cond(
              p.Term.Select(named("found", p.Type.Bool1), Nil, p.Type.Bool1),
              List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(7)))),
              Nil
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(9)))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(p.Term.IntS32Const(5)), i32)
    val result = named("result", i32)
    val out    = arg("out", p.Type.Ptr(i32, p.Type.Space.Global))
    val in = program(
      entry(
        args = List(out),
        body = List(
          p.Stmt.Var(result, Some(invoke)),
          p.Stmt.Update(
            p.Term.Select(out.named, Nil, out.named.tpe),
            p.Term.IntS32Const(0),
            p.Term.Select(result, Nil, i32)
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      List(helper)
    )

    def run(prog: p.Program): Long = {
      val vm   = Interpreter.Vm(prog)
      val cell = vm.allocOf(i32, 1)
      vm.call(p.Conventions.EntryName, List(out.named.tpe -> V.I(cell)))
      vm.load(cell, i32) match { case V.I(value) => value; case _ => Long.MinValue }
    }

    assertEquals(run(in), 7L)
    assertEquals(run(FnInline(in, NoopLog)), 7L)
  }

  test("inlining alpha-renames locals inside raise cleanup") {
    val local = named("cleanup", p.Type.IntS32)
    val helper = fn(
      "helper",
      body = List(
        raise(
          p.Term.IntS32Const(1),
          "int",
          List(
            p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS32Const(2))), isMutable = true),
            p.Stmt.Mut(selectT(local), p.Expr.Alias(p.Term.IntS32Const(3)))
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
    val in = program(
      entry(
        body = List(
          p.Stmt.Var(named("a", p.Type.Unit0), Some(invoke)),
          p.Stmt.Var(named("b", p.Type.Unit0), Some(invoke)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      List(helper)
    )

    val out     = FnInline(in, NoopLog)
    val raises  = out.entry.collectWhere[p.Stmt] { case x: p.Stmt.Raise => x }
    val locals  = raises.flatMap(_.cleanup.collect { case p.Stmt.Var(name, _, _) => name })
    val mutated = raises.flatMap(_.cleanup.collect { case p.Stmt.Mut(p.Term.Select(root, _, _), _) => root })
    assertEquals(raises.size, 2)
    assertEquals(locals.distinct.size, 2)
    assertEquals(mutated, locals)
    assert(locals.forall(_.symbol != local.symbol))
    assertEquals(Verify(out, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }
}
