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
