package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import PassTest.*

class StructuredExitSuite extends munit.FunSuite {

  private val i32   = p.Type.IntS32
  private val g     = p.Type.Space.Global
  private val errT  = p.Type.Ptr(p.Type.IntS8, g)
  private val limit = p.Conventions.assertMessageLimit

  private def assertStmt(code: Int = p.Enums.AssertCode.Assert.value, msg: String = "x"): p.Stmt =
    p.Stmt.Var(
      named("_a", p.Type.Unit0),
      Some(p.Expr.SpecOp(p.Spec.Assert(p.Term.IntU32Const(code), p.Term.StringConst(msg))))
    )

  private def ret = p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))

  private def lower(args: List[p.Arg], body: List[p.Stmt]): p.Program =
    StructuredExit(program(entry(args = args, body = body)), NoopLog)

  private def byteAt(vm: Interpreter.Vm, addr: Long): Int = vm.load(addr, p.Type.IntU8) match {
    case V.I(v) => (v & 0xff).toInt
    case _      => 0
  }
  private def i32At(vm: Interpreter.Vm, addr: Long): Long = vm.load(addr, i32) match {
    case V.I(v) => v
    case _      => 0
  }

  private def decode(vm: Interpreter.Vm, err: Long): (Int, String) = {
    val code = i32At(vm, err).toInt
    val msg  = Iterator.from(4).take(limit).map(o => byteAt(vm, err + o)).takeWhile(_ != 0).map(_.toChar).mkString
    (code, msg)
  }

  test("an entry that never asserts is unchanged") {
    val in = program(entry(body = List(ret)))
    assertEquals(StructuredExit(in, NoopLog), in)
  }

  test("an asserting entry gains a leading error-buffer arg and a drain return") {
    val out = lower(Nil, List(assertStmt(), ret))
    assertEquals(out.entry.args.headOption.map(_.named.tpe), Some(errT)) // the leading Ptr<i8> the dispatch binds
    assert(out.entry.body.lastOption.exists(_.isInstanceOf[p.Stmt.Return]), "drains to a return")
  }

  test("round-trip: the 4cc code (little-endian) and the message land in the error buffer") {
    val code = p.Enums.AssertCode.Assert.value
    val out  = lower(Nil, List(assertStmt(code, "out of bounds"), ret))
    val vm   = Interpreter.Vm(out)
    val err  = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err)))
    assertEquals(decode(vm, err), (code, "out of bounds"))
  }

  test("a statement after an assert is fenced") {
    val arr = named("arr", p.Type.Ptr(i32, g))
    val out = lower(
      List(p.Arg(arr)),
      List(
        p.Stmt.Update(selectT(arr), p.Term.IntU32Const(0), p.Term.IntS32Const(1)),
        assertStmt(),
        p.Stmt.Update(selectT(arr), p.Term.IntU32Const(0), p.Term.IntS32Const(2)),
        ret
      )
    )
    val vm  = Interpreter.Vm(out)
    val a   = vm.alloc(4L)
    val err = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err), p.Type.Ptr(i32, g) -> V.I(a)))
    assertEquals(i32At(vm, a), 1L) // the post-assert write never happened
  }

  test("a loop holding an assert drains: the asserting iteration commits, later ones do not run") {
    val arr = named("arr", p.Type.Ptr(i32, g))
    val i   = named("i", i32)
    val t   = named("t", i32)
    val c   = named("c", p.Type.Bool1)
    val loopBody = List(
      p.Stmt.Var(t, Some(p.Expr.IntrOp(p.Intr.Add(selectT(i), p.Term.IntS32Const(1), i32)))),
      p.Stmt.Update(selectT(arr), selectT(i), selectT(t)),
      p.Stmt.Var(c, Some(p.Expr.IntrOp(p.Intr.LogicEq(selectT(i), p.Term.IntS32Const(1))))),
      p.Stmt.Cond(selectT(c), List(assertStmt()), Nil)
    )
    val out = lower(
      List(p.Arg(arr)),
      List(p.Stmt.ForRange(i, p.Term.IntS32Const(0), p.Term.IntS32Const(4), p.Term.IntS32Const(1), loopBody), ret)
    )
    val vm  = Interpreter.Vm(out)
    val a   = vm.alloc(4L * 4)
    val err = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err), p.Type.Ptr(i32, g) -> V.I(a)))
    assertEquals((0 until 4).map(k => i32At(vm, a + 4L * k)).toList, List(1L, 2L, 0L, 0L))
  }

  test("an asserting lane and a non-asserting lane execute the same collective barriers") {
    val flag    = named("flag", p.Type.Bool1)
    val barrier = p.Stmt.Var(named("_b", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), isMutable = false)
    val out     = lower(List(p.Arg(flag)), List(p.Stmt.Cond(selectT(flag), List(assertStmt()), Nil), barrier, ret))
    def barriersWhen(asserts: Boolean): Long = {
      val vm  = Interpreter.Vm(out)
      val err = vm.alloc(4L + limit)
      vm.call(p.Conventions.EntryName, List(errT -> V.I(err), p.Type.Bool1 -> V.I(if (asserts) 1L else 0L)))
      vm.barrierCount
    }
    assertEquals(barriersWhen(asserts = true), barriersWhen(asserts = false))
  }

  test("draining a barrier-free loop still reaches a trailing collective barrier on every lane") {
    val arr      = named("arr", p.Type.Ptr(i32, g))
    val assertAt = named("assertAt", i32)
    val i        = named("i", i32)
    val t        = named("t", i32)
    val c        = named("c", p.Type.Bool1)
    val loopBody = List(
      p.Stmt.Var(t, Some(p.Expr.IntrOp(p.Intr.Add(selectT(i), p.Term.IntS32Const(1), i32)))),
      p.Stmt.Update(selectT(arr), selectT(i), selectT(t)),
      p.Stmt.Var(c, Some(p.Expr.IntrOp(p.Intr.LogicEq(selectT(i), selectT(assertAt))))),
      p.Stmt.Cond(selectT(c), List(assertStmt()), Nil)
    )
    val barrier = p.Stmt.Var(named("_b", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), isMutable = false)
    val out = lower(
      List(p.Arg(arr), p.Arg(assertAt)),
      List(
        p.Stmt.ForRange(i, p.Term.IntS32Const(0), p.Term.IntS32Const(4), p.Term.IntS32Const(1), loopBody),
        barrier,
        ret
      )
    )
    def barriersWhen(at: Long): Long = {
      val vm  = Interpreter.Vm(out)
      val a   = vm.alloc(4L * 4)
      val err = vm.alloc(4L + limit)
      vm.call(p.Conventions.EntryName, List(errT -> V.I(err), p.Type.Ptr(i32, g) -> V.I(a), i32 -> V.I(at)))
      vm.barrierCount
    }
    assertEquals(barriersWhen(at = 1), barriersWhen(at = -1))
  }

  test("a barrier region's setup runs on an asserting lane too") {
    val out     = named("out", p.Type.Ptr(i32, g))
    val flag    = named("flag", p.Type.Bool1)
    val base    = named("base", i32)
    val j       = named("j", i32)
    val barrier = p.Stmt.Var(named("_b", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), isMutable = false)
    val prog = lower(
      List(p.Arg(out), p.Arg(flag)),
      List(
        p.Stmt.Var(base, Some(p.Expr.Alias(p.Term.IntS32Const(0))), isMutable = true),
        p.Stmt.Cond(selectT(flag), List(assertStmt()), Nil),
        p.Stmt.Mut(selectT(base), p.Expr.Alias(p.Term.IntS32Const(42))),
        p.Stmt.ForRange(
          j,
          p.Term.IntS32Const(0),
          p.Term.IntS32Const(1),
          p.Term.IntS32Const(1),
          List(barrier, p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), selectT(base)))
        ),
        ret
      )
    )
    val vm  = Interpreter.Vm(prog)
    val o   = vm.alloc(4L)
    val err = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err), p.Type.Ptr(i32, g) -> V.I(o), p.Type.Bool1 -> V.I(1L)))
    assertEquals(i32At(vm, o), 42L) // the asserting lane set base before the barrier region read it
  }

  test("an assert inside a barrier-bearing loop is rejected") {
    val i       = named("i", i32)
    val barrier = p.Stmt.Var(named("_b", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), isMutable = false)
    val loop = p.Stmt.ForRange(
      i,
      p.Term.IntS32Const(0),
      p.Term.IntS32Const(4),
      p.Term.IntS32Const(1),
      List(barrier, assertStmt())
    )
    intercept[RuntimeException](StructuredExit(program(entry(body = List(loop, ret))), NoopLog))
  }
}
