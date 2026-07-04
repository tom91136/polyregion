package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import polyregion.ast.Traversal.*

class RecursionLowerSuite extends munit.FunSuite {
  import PassTest.*

  private val i32  = p.Type.IntS32
  private val bool = p.Type.Bool1

  private def vlet(n: String, e: p.Expr, t: p.Type = i32, mut: Boolean = false) =
    p.Stmt.Var(named(n, t), Some(e), mut)
  private def ret(t: p.Term)            = p.Stmt.Return(p.Expr.Alias(t))
  private def lt(a: p.Term, b: p.Term)  = p.Expr.IntrOp(p.Intr.LogicLt(a, b))
  private def lte(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.LogicLte(a, b))
  private def eq(a: p.Term, b: p.Term)  = p.Expr.IntrOp(p.Intr.LogicEq(a, b))
  private def sub(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.Sub(a, b, i32))
  private def add(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.Add(a, b, i32))
  private def mul(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.Mul(a, b, i32))

  private def hasFrame(out: p.Program): Boolean = out.defs.exists(_.members.exists(_.symbol == "#pc"))
  // the generic path traps on stack overflow with a `'assert`; the StructuredExit pass lowers this
  private def hasOverflowGuard(out: p.Program): Boolean =
    out.functions.exists(_.body.collectWhere[p.Expr] { case p.Expr.SpecOp(_: p.Spec.Assert) => () }.nonEmpty)
  private def call(name: String, args: List[p.Term], rtn: p.Type = i32) =
    p.Expr.Invoke(sym(name), Nil, None, args, rtn)
  private def i(v: Int): p.Term = p.Term.IntS32Const(v)
  private def alias(t: p.Term)  = p.Expr.Alias(t)
  private def boolc(b: Boolean) = p.Term.Bool1Const(b)

  private def selfCalls(f: p.Function): Int =
    f.body.collectWhere[p.Expr] { case ivk: p.Expr.Invoke if ivk.name == f.name => () }.size

  // fib(n) = n<2 ? n : fib(n-1)+fib(n-2), in the ANF shape the frontend emits
  private def fib: p.Function =
    fn(
      "fib",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lt(selectT("n", i32), i(2)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(selectT("n", i32))),
          List(
            vlet("a", sub(selectT("n", i32), i(1))),
            vlet("t0", call("fib", List(selectT("a", i32)))),
            vlet("b", sub(selectT("n", i32), i(2))),
            vlet("t1", call("fib", List(selectT("b", i32)))),
            vlet("s", add(selectT("t0", i32), selectT("t1", i32))),
            ret(selectT("s", i32))
          )
        )
      )
    )

  // ackermann: nested calls + a tail recursive call + a value saved across a call
  private def ack: p.Function =
    fn(
      "ack",
      List(arg("m", i32), arg("n", i32)),
      i32,
      List(
        vlet("mz", eq(selectT("m", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("mz", bool),
          List(vlet("np1", add(selectT("n", i32), i(1))), ret(selectT("np1", i32))),
          List(
            vlet("nz", eq(selectT("n", i32), i(0)), bool),
            p.Stmt.Cond(
              selectT("nz", bool),
              List(
                vlet("m1", sub(selectT("m", i32), i(1))),
                p.Stmt.Return(call("ack", List(selectT("m1", i32), i(1))))
              ),
              List(
                vlet("n1", sub(selectT("n", i32), i(1))),
                vlet("inner", call("ack", List(selectT("m", i32), selectT("n1", i32)))),
                vlet("m1b", sub(selectT("m", i32), i(1))),
                vlet("r", call("ack", List(selectT("m1b", i32), selectT("inner", i32)))),
                ret(selectT("r", i32))
              )
            )
          )
        )
      )
    )

  // tail recursion: fact(n, acc) = n<=1 ? acc : fact(n-1, acc*n)
  private def fact: p.Function =
    fn(
      "fact",
      List(arg("n", i32), arg("acc", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(1)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(selectT("acc", i32))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            vlet("acc1", mul(selectT("acc", i32), selectT("n", i32))),
            p.Stmt.Return(call("fact", List(selectT("n1", i32), selectT("acc1", i32))))
          )
        )
      )
    )

  // one tail self-call (return g(t)) and one non-tail self-call (t = g(n-1)) -> not TCO-eligible
  private def gmix: p.Function =
    fn(
      "g",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(i(0))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            vlet("t", call("g", List(selectT("n1", i32)))),
            p.Stmt.Return(call("g", List(selectT("t", i32))))
          )
        )
      )
    )

  // tail recursion whose tail-call args swap two params: a naive in-order reassign clobbers, so this only
  // works if the transform snapshots args to temps first. swap_count(a,b,n) = n even ? a : b
  private def swapCount: p.Function =
    fn(
      "swap_count",
      List(arg("a", i32), arg("b", i32), arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(selectT("a", i32))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            p.Stmt.Return(call("swap_count", List(selectT("b", i32), selectT("a", i32), selectT("n1", i32))))
          )
        )
      )
    )

  // two distinct tail calls in different branches
  private def ping: p.Function =
    fn(
      "ping",
      List(arg("n", i32), arg("x", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(selectT("x", i32))),
          List(
            vlet("c2", lte(selectT("n", i32), i(5)), bool),
            vlet("n1", sub(selectT("n", i32), i(1))),
            p.Stmt.Cond(
              selectT("c2", bool),
              List(
                vlet("x1", add(selectT("x", i32), i(1))),
                p.Stmt.Return(call("ping", List(selectT("n1", i32), selectT("x1", i32))))
              ),
              List(
                vlet("x2", add(selectT("x", i32), i(2))),
                p.Stmt.Return(call("ping", List(selectT("n1", i32), selectT("x2", i32))))
              )
            )
          )
        )
      )
    )

  // linear (single) non-tail recursion: sum(n) = n + sum(n-1)
  private def sum: p.Function =
    fn(
      "sum",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(i(0))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            vlet("t", call("sum", List(selectT("n1", i32)))),
            vlet("r", add(selectT("n", i32), selectT("t", i32))),
            ret(selectT("r", i32))
          )
        )
      )
    )

  // a recursive call inside a for-loop: lsum(n) = sum_{i<n} lsum(i) + n = 2^n - 1
  private def loopSum: p.Function =
    fn(
      "lsum",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(i(0))),
          List(
            p.Stmt.Var(named("s", i32), Some(alias(i(0))), isMutable = true),
            p.Stmt.ForRange(
              named("i", i32),
              i(0),
              selectT("n", i32),
              i(1),
              List(
                vlet("t", call("lsum", List(selectT("i", i32)))),
                p.Stmt.Mut(selectT("s", i32), add(selectT("s", i32), selectT("t", i32)))
              )
            ),
            vlet("r", add(selectT("s", i32), selectT("n", i32))),
            ret(selectT("r", i32))
          )
        )
      )
    )

  // mutual recursion: neither function self-calls, so a self-call-only detector would miss it
  private def isEven: p.Function =
    fn(
      "is_even",
      List(arg("n", i32)),
      bool,
      List(
        vlet("c", eq(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(boolc(true))),
          List(vlet("n1", sub(selectT("n", i32), i(1))), p.Stmt.Return(call("is_odd", List(selectT("n1", i32)), bool)))
        )
      )
    )
  private def isOdd: p.Function =
    fn(
      "is_odd",
      List(arg("n", i32)),
      bool,
      List(
        vlet("c", eq(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(boolc(false))),
          List(vlet("n1", sub(selectT("n", i32), i(1))), p.Stmt.Return(call("is_even", List(selectT("n1", i32)), bool)))
        )
      )
    )

  // non-tail mutual recursion: mf(n)=n+mg(n-1), mg(n)=mf(n-1)+1 (calls bound to temps -> generic stack)
  private def mf: p.Function =
    fn(
      "mf",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(i(0))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            vlet("t", call("mg", List(selectT("n1", i32)))),
            vlet("r", add(selectT("n", i32), selectT("t", i32))),
            ret(selectT("r", i32))
          )
        )
      )
    )
  private def mg: p.Function =
    fn(
      "mg",
      List(arg("n", i32)),
      i32,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(ret(i(0))),
          List(
            vlet("n1", sub(selectT("n", i32), i(1))),
            vlet("t", call("mf", List(selectT("n1", i32)))),
            vlet("r", add(selectT("t", i32), i(1))),
            ret(selectT("r", i32))
          )
        )
      )
    )

  private val i32p = p.Type.Ptr(i32, p.Type.Space.Global)
  // void (Unit0) recursion with side effects through a pointer param: fillrec(out, n) sets out[n-1]=n, recurse
  private def fillrec: p.Function =
    fn(
      "fillrec",
      List(arg("out", i32p), arg("n", i32)),
      p.Type.Unit0,
      List(
        vlet("c", lte(selectT("n", i32), i(0)), bool),
        p.Stmt.Cond(
          selectT("c", bool),
          List(p.Stmt.Return(alias(p.Term.Unit0Const))),
          List(
            vlet("idx", sub(selectT("n", i32), i(1))),
            p.Stmt.Update(selectT("out", i32p), selectT("idx", i32), selectT("n", i32)),
            vlet("n1", sub(selectT("n", i32), i(1))),
            p.Stmt.Var(
              named("_u", p.Type.Unit0),
              Some(call("fillrec", List(selectT("out", i32p), selectT("n1", i32)), p.Type.Unit0))
            ),
            p.Stmt.Return(alias(p.Term.Unit0Const))
          )
        )
      )
    )

  private def lower(prog: p.Program): p.Program = RecursionLower()(prog, PassTest.NoopLog)

  test("fib lowers to a non-recursive driver and matches the recursive oracle") {
    val in      = program(entry(), List(fib))
    val out     = lower(in)
    val lowered = out.functions.find(_.name == sym("fib")).get
    assertEquals(selfCalls(fib), 2)
    assertEquals(selfCalls(lowered), 0, "no self-Invoke should remain after lowering")
    assert(out.defs.exists(_.members.exists(_.symbol == "#pc")), "a frame struct should be synthesised")

    for (n <- 0 to 15) {
      val oracle = Interpreter.Vm(in).call("fib", List(i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("fib", List(i32 -> V.I(n)))
      assertEquals(got, oracle, s"fib($n)")
    }
    assertEquals(Interpreter.Vm(out).call("fib", List(i32 -> V.I(10))), V.I(55))
  }

  test("ackermann (nested + tail calls) lowers and matches the oracle") {
    val in  = program(entry(), List(ack))
    val out = lower(in)
    assertEquals(selfCalls(out.functions.find(_.name == sym("ack")).get), 0)

    for {
      m <- 0 to 3
      n <- 0 to 4
    } {
      val oracle = Interpreter.Vm(in).call("ack", List(i32 -> V.I(m), i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("ack", List(i32 -> V.I(m), i32 -> V.I(n)))
      assertEquals(got, oracle, s"ack($m,$n)")
    }
  }

  test("tail-recursive factorial takes the TCO path (a loop, no frame struct)") {
    val in  = program(entry(), List(fact))
    val out = lower(in)
    assertEquals(selfCalls(out.functions.find(_.name == sym("fact")).get), 0)
    assert(!hasFrame(out), "TCO must not synthesise a frame struct - that is how we know the fast path ran")

    for ((n, expected) <- List(1 -> 1, 5 -> 120, 10 -> 3628800)) {
      val oracle = Interpreter.Vm(in).call("fact", List(i32 -> V.I(n), i32 -> V.I(1)))
      val got    = Interpreter.Vm(out).call("fact", List(i32 -> V.I(n), i32 -> V.I(1)))
      assertEquals(got, oracle, s"fact($n)")
      assertEquals(got, V.I(expected.toLong))
    }
  }

  test("a non-tail self-call forces the generic stack path (frame struct present)") {
    val in  = program(entry(), List(gmix))
    val out = lower(in)
    assertEquals(selfCalls(out.functions.find(_.name == sym("g")).get), 0)
    assert(hasFrame(out), "a function with any non-tail self-call must use the generic path")

    for (n <- 0 to 6) {
      val oracle = Interpreter.Vm(in).call("g", List(i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("g", List(i32 -> V.I(n)))
      assertEquals(got, oracle, s"g($n)")
    }
  }

  test("fib (non-tail binary recursion) takes the generic path (frame struct present)") {
    assert(hasFrame(lower(program(entry(), List(fib)))), "fib must use the generic stack path")
  }

  // pins the assert-pass integration point: the generic path must emit an overflow trap to lower
  test("the generic path emits a stack-overflow guard ('assert) but the TCO path does not") {
    assert(hasOverflowGuard(lower(program(entry(), List(fib)))), "generic stack path must guard against overflow")
    assert(!hasOverflowGuard(lower(program(entry(), List(fact)))), "the TCO path is a bounded loop, no overflow guard")
  }

  test("tail call that swaps params is correct (snapshot, not in-order clobber)") {
    val in  = program(entry(), List(swapCount))
    val out = lower(in)
    assert(!hasFrame(out), "swap_count is pure tail recursion -> TCO")
    for (n <- 0 to 5) {
      val oracle = Interpreter.Vm(in).call("swap_count", List(i32 -> V.I(10), i32 -> V.I(20), i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("swap_count", List(i32 -> V.I(10), i32 -> V.I(20), i32 -> V.I(n)))
      assertEquals(got, oracle, s"swap_count(10,20,$n)")
      assertEquals(got, V.I(if (n % 2 == 0) 10L else 20L)) // a clobbered reassign would give 20 for every n
    }
  }

  test("branching tail calls take the TCO path") {
    val in  = program(entry(), List(ping))
    val out = lower(in)
    assert(!hasFrame(out), "every self-call is a tail return -> TCO even across branches")
    for (n <- 0 to 9) {
      val oracle = Interpreter.Vm(in).call("ping", List(i32 -> V.I(n), i32 -> V.I(0)))
      val got    = Interpreter.Vm(out).call("ping", List(i32 -> V.I(n), i32 -> V.I(0)))
      assertEquals(got, oracle, s"ping($n,0)")
    }
  }

  test("linear (single) non-tail recursion takes the generic path") {
    val in  = program(entry(), List(sum))
    val out = lower(in)
    assert(hasFrame(out), "a non-tail recursive call -> generic stack path")
    for (n <- 0 to 12) {
      val oracle = Interpreter.Vm(in).call("sum", List(i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("sum", List(i32 -> V.I(n)))
      assertEquals(got, oracle, s"sum($n)")
      assertEquals(got, V.I(n.toLong * (n + 1) / 2))
    }
  }

  test("a recursive call inside a for-loop is lowered (generic stack) and matches the oracle") {
    val in  = program(entry(), List(loopSum))
    val out = lower(in)
    assertEquals(selfCalls(out.functions.find(_.name == sym("lsum")).get), 0)
    assert(hasFrame(out), "a recursive call inside a loop needs the explicit stack")
    for (n <- 0 to 8) {
      val oracle = Interpreter.Vm(in).call("lsum", List(i32 -> V.I(n)))
      val got    = Interpreter.Vm(out).call("lsum", List(i32 -> V.I(n)))
      assertEquals(got, oracle, s"lsum($n)")
      assertEquals(got, V.I((1L << n) - 1))
    }
  }

  test("tail mutual recursion (is_even/is_odd) lowers to TCO and matches the oracle") {
    val in  = program(entry(), List(isEven, isOdd))
    val out = lower(in)
    assert(!hasFrame(out), "tail-mutual recursion merges to a tail driver -> TCO, no frame struct")
    assertEquals(out.functions.count(_.name.repr.startsWith("_scc")), 1, "one merged driver for the cluster")
    for (n <- 0 to 8) {
      val gotE = Interpreter.Vm(out).call("is_even", List(i32 -> V.I(n)))
      val gotO = Interpreter.Vm(out).call("is_odd", List(i32 -> V.I(n)))
      assertEquals(gotE, Interpreter.Vm(in).call("is_even", List(i32 -> V.I(n))), s"is_even($n)")
      assertEquals(gotO, Interpreter.Vm(in).call("is_odd", List(i32 -> V.I(n))), s"is_odd($n)")
      assertEquals(gotE, V.I(if (n % 2 == 0) 1L else 0L))
    }
  }

  test("non-tail mutual recursion (mf/mg) takes the generic stack path and matches the oracle") {
    val in  = program(entry(), List(mf, mg))
    val out = lower(in)
    assert(hasFrame(out), "non-tail mutual recursion needs the explicit stack")
    for (n <- 0 to 8) {
      assertEquals(
        Interpreter.Vm(out).call("mf", List(i32 -> V.I(n))),
        Interpreter.Vm(in).call("mf", List(i32 -> V.I(n))),
        s"mf($n)"
      )
      assertEquals(
        Interpreter.Vm(out).call("mg", List(i32 -> V.I(n))),
        Interpreter.Vm(in).call("mg", List(i32 -> V.I(n))),
        s"mg($n)"
      )
    }
  }

  test("void (Unit0) recursion fills an array via a pointer param (generic path)") {
    val in  = program(entry(), List(fillrec))
    val out = lower(in)
    assert(hasFrame(out), "non-tail void recursion -> generic stack path")
    def run(prog: p.Program): List[Long] = {
      val vm  = Interpreter.Vm(prog)
      val arr = vm.allocOf(i32, 5)
      (0 until 5).foreach(k => vm.store(arr + k * 4, V.I(0), i32))
      vm.call("fillrec", List(i32p -> V.I(arr), i32 -> V.I(5)))
      (0 until 5).toList.map(k => vm.load(arr + k * 4, i32) match { case V.I(x) => x; case _ => -1L })
    }
    assertEquals(run(in), List(1L, 2L, 3L, 4L, 5L))
    assertEquals(run(out), run(in))
  }

  test("non-recursive programs pass through untouched") {
    val sq = fn("sq", List(arg("x", i32)), i32, List(ret(selectT("x", i32))))
    val in = program(entry(), List(sq))
    assertEquals(lower(in), in)
  }
}
