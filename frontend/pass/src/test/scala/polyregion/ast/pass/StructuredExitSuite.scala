package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import PassTest.*

class StructuredExitSuite extends munit.FunSuite {

  private val i32    = p.Type.IntS32
  private val g      = p.Type.Space.Global
  private val errT   = p.Type.Ptr(p.Type.IntS8, g)
  private val limit  = p.Conventions.assertMessageLimit
  private val outArg = named("out", p.Type.Ptr(i32, g))

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
    val bytes = Iterator
      .from(4)
      .take(limit)
      .map(o => byteAt(vm, err + o))
      .takeWhile(_ != 0)
      .map(_.toByte)
      .toArray
    val msg = String(bytes, java.nio.charset.StandardCharsets.UTF_8)
    (code, msg)
  }

  private def runOutBody(body: List[p.Stmt]): (Long, (Int, String)) = {
    val lowered = lower(List(p.Arg(outArg)), body :+ ret)
    val vm      = Interpreter.Vm(lowered)
    val out     = vm.alloc(4L)
    val err     = vm.alloc(4L + limit)
    val args = lowered.entry.args.map { arg =>
      arg.named.symbol match {
        case p.Conventions.ErrorArg            => errT       -> V.I(err)
        case symbol if symbol == outArg.symbol => outArg.tpe -> V.I(out)
        case symbol                            => fail(s"unexpected lowered argument $symbol")
      }
    }
    vm.call(p.Conventions.EntryName, args)
    i32At(vm, out) -> decode(vm, err)
  }

  private def runOut(stmt: p.Stmt): (Long, (Int, String)) = runOutBody(List(stmt))

  private def append(digit: Int, suffix: String): List[p.Stmt] = {
    val current = named(s"current$suffix", i32)
    val shifted = named(s"shifted$suffix", i32)
    val next    = named(s"next$suffix", i32)
    List(
      p.Stmt.Var(current, Some(p.Expr.Index(selectT(outArg), p.Term.IntS32Const(0), i32))),
      p.Stmt.Var(shifted, Some(p.Expr.IntrOp(p.Intr.Mul(selectT(current), p.Term.IntS32Const(10), i32)))),
      p.Stmt.Var(next, Some(p.Expr.IntrOp(p.Intr.Add(selectT(shifted), p.Term.IntS32Const(digit), i32)))),
      p.Stmt.Update(selectT(outArg), p.Term.IntS32Const(0), selectT(next))
    )
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

  test("literal messages are encoded as UTF-8") {
    val code = p.Enums.AssertCode.Assert.value
    val out  = lower(Nil, List(assertStmt(code, "λ failed"), ret))
    val vm   = Interpreter.Vm(out)
    val err  = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err)))
    assertEquals(decode(vm, err), (code, "λ failed"))
  }

  test("lowering is deterministic across repeated pass runs") {
    val in = program(entry(body = List(assertStmt(), ret)))
    assertEquals(StructuredExit(in, NoopLog), StructuredExit(in, NoopLog))
  }

  test("an assertion exposed by helper inlining changes the entry ABI") {
    val helper = fn("helper", body = List(assertStmt(msg = "from helper"), ret))
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
    val in = program(
      entry(body = List(p.Stmt.Var(named("result", p.Type.Unit0), Some(invoke)), ret)),
      functions = List(helper)
    )

    val out = StructuredExit(FnInline(in, NoopLog), NoopLog)

    assertEquals(out.functions, Nil)
    assertEquals(out.entry.args.headOption.map(_.named.symbol), Some(p.Conventions.ErrorArg))
    assertEquals(out.entry.collectWhere[p.Expr] { case p.Expr.SpecOp(_: p.Spec.Assert) => () }, Nil)

    val vm  = Interpreter.Vm(out)
    val err = vm.alloc(4L + limit)
    vm.call(p.Conventions.EntryName, List(errT -> V.I(err)))
    assertEquals(decode(vm, err), (p.Enums.AssertCode.Assert.value, "from helper"))
  }

  test("an assertion after a handled raise is not caught by an enclosing catch-all") {
    val inner = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int")),
      List(handler(Some(i32), None, Nil, Some("int"))),
      Nil
    )
    val outer = p.Stmt.Try(
      List(inner, assertStmt(msg = "still an assert")),
      List(
        handler(
          None,
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(1))),
          None
        )
      ),
      Nil
    )
    assertEquals(runOut(outer), 0L -> (p.Enums.AssertCode.Assert.value, "still an assert"))
  }

  test("finally runs after a normally completing try body") {
    val in = p.Stmt.Try(
      List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(1))),
      Nil,
      List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(2)))
    )
    assertEquals(runOut(in)._1, 2L)
  }

  test("finally runs after an assertion without requiring exception-tag state") {
    val in = p.Stmt.Try(
      List(assertStmt()),
      Nil,
      List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(2)))
    )
    assertEquals(runOut(in), 2L -> (p.Enums.AssertCode.Assert.value, "x"))
  }

  test("finally runs before return") {
    val out = named("out", p.Type.Ptr(i32, g))
    val in = program(
      entry(
        args = List(p.Arg(out)),
        body = List(
          p.Stmt.Try(
            List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
            Nil,
            List(p.Stmt.Update(selectT(out), p.Term.IntU32Const(0), p.Term.IntS32Const(7)))
          )
        )
      )
    )
    val lowered = StructuredExit(in, NoopLog)
    val vm      = Interpreter.Vm(lowered)
    val cell    = vm.alloc(4L)
    vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(cell)))
    assertEquals(i32At(vm, cell), 7L)
  }

  test("finally runs before break") {
    val i = named("i", i32)
    val in = p.Stmt.ForRange(
      i,
      p.Term.IntS32Const(0),
      p.Term.IntS32Const(3),
      p.Term.IntS32Const(1),
      List(
        p.Stmt.Try(
          List(p.Stmt.Break),
          Nil,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(7)))
        )
      )
    )
    assertEquals(runOut(in)._1, 7L)
  }

  test("finally runs before continue") {
    val i     = named("i", i32)
    val count = named("count", i32)
    val next  = named("next", i32)
    val in = p.Stmt.ForRange(
      i,
      p.Term.IntS32Const(0),
      p.Term.IntS32Const(3),
      p.Term.IntS32Const(1),
      List(
        p.Stmt.Try(
          List(p.Stmt.Cont),
          Nil,
          List(
            p.Stmt.Var(count, Some(p.Expr.Index(selectT(outArg), p.Term.IntU32Const(0), i32))),
            p.Stmt.Var(next, Some(p.Expr.IntrOp(p.Intr.Add(selectT(count), p.Term.IntS32Const(1), i32)))),
            p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), selectT(next))
          )
        )
      )
    )
    val lowered = lower(List(p.Arg(outArg)), List(in, ret))
    val vm      = Interpreter.Vm(lowered)
    val cell    = vm.alloc(4L)
    vm.store(cell, V.I(0), i32)
    val err = vm.alloc(4L + limit)
    val args = lowered.entry.args.map { arg =>
      if (arg.named.symbol == p.Conventions.ErrorArg) errT -> V.I(err) else outArg.tpe -> V.I(cell)
    }
    vm.call(p.Conventions.EntryName, args)
    assertEquals(i32At(vm, cell), 3L)
  }

  test("nested finalizers run from the inside out before return") {
    val out = named("out", p.Type.Ptr(i32, g))
    val one = named("one", i32)
    val ten = named("ten", i32)
    val in = program(
      entry(
        args = List(p.Arg(out)),
        body = List(
          p.Stmt.Try(
            List(
              p.Stmt.Try(
                List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
                Nil,
                List(p.Stmt.Update(selectT(out), p.Term.IntU32Const(0), p.Term.IntS32Const(1)))
              )
            ),
            Nil,
            List(
              p.Stmt.Var(one, Some(p.Expr.Index(selectT(out), p.Term.IntU32Const(0), i32))),
              p.Stmt.Var(ten, Some(p.Expr.IntrOp(p.Intr.Mul(selectT(one), p.Term.IntS32Const(10), i32)))),
              p.Stmt.Update(selectT(out), p.Term.IntU32Const(0), selectT(ten))
            )
          )
        )
      )
    )
    val lowered = StructuredExit(in, NoopLog)
    val vm      = Interpreter.Vm(lowered)
    val cell    = vm.alloc(4L)
    vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(cell)))
    assertEquals(i32At(vm, cell), 10L)
    assertEquals(Verify(lowered, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }

  test("finally runs after a handled raise") {
    val in = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int")),
      List(
        handler(
          Some(i32),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(1))),
          Some("int")
        )
      ),
      List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(2)))
    )
    assertEquals(runOut(in), 2L -> (0, ""))
  }

  test("finally runs before an unhandled raise reaches an outer handler") {
    val inner = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int")),
      Nil,
      List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(1)))
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(
        handler(
          Some(i32),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(2))),
          Some("int")
        )
      ),
      Nil
    )
    assertEquals(runOut(outer), 2L -> (0, ""))
  }

  test("a raise from finally replaces the pending raise") {
    val inner = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int")),
      Nil,
      List(raise(p.Term.IntS64Const(9), "long"))
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(
        handler(
          Some(i32),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(1))),
          Some("int")
        ),
        handler(
          Some(p.Type.IntS64),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), p.Term.IntS32Const(2))),
          Some("long")
        )
      ),
      Nil
    )
    assertEquals(runOut(outer), 2L -> (0, ""))
  }

  test("a raise from finally cleans the pending exception before its replacement") {
    val inner = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int", append(1, "Original"))),
      Nil,
      List(raise(p.Term.IntS64Const(9), "long", append(2, "Replacement")))
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(handler(Some(p.Type.IntS64), None, Nil, Some("long"))),
      Nil
    )
    assertEquals(runOut(outer), 12L -> (0, ""))
  }

  test("an abrupt finally cleans the pending exception") {
    val cleanup = append(1, "Pending")
    val returning = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int", cleanup)),
      Nil,
      List(ret)
    )
    assertEquals(runOut(returning), 1L -> (0, ""))

    val breaking = p.Stmt.While(
      p.Term.Bool1Const(true),
      List(
        p.Stmt.Try(
          List(raise(p.Term.IntS32Const(7), "int", cleanup)),
          Nil,
          List(p.Stmt.Break)
        )
      )
    )
    assertEquals(runOut(breaking), 1L -> (0, ""))

    val more = named("more", p.Type.Bool1)
    val continuing = List(
      p.Stmt.Var(more, Some(p.Expr.Alias(p.Term.Bool1Const(true))), isMutable = true),
      p.Stmt.While(
        selectT(more),
        List(
          p.Stmt.Try(
            List(raise(p.Term.IntS32Const(7), "int", cleanup)),
            Nil,
            List(
              p.Stmt.Mut(selectT(more), p.Expr.Alias(p.Term.Bool1Const(false))),
              p.Stmt.Cont
            )
          )
        )
      )
    )
    assertEquals(runOutBody(continuing), 1L -> (0, ""))
  }

  test("a throwing finally copied for an abrupt exit runs once") {
    val inner = p.Stmt.Try(
      List(ret),
      Nil,
      append(1, "Finally") :+ raise(p.Term.IntS64Const(9), "long")
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(handler(Some(p.Type.IntS64), None, Nil, Some("long"))),
      Nil
    )
    assertEquals(runOut(outer), 1L -> (0, ""))
  }

  test("a throwing finally bypasses its associated handlers") {
    val inner = p.Stmt.Try(
      List(ret),
      List(
        handler(
          Some(p.Type.IntS64),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntS32Const(0), p.Term.IntS32Const(9))),
          Some("long")
        )
      ),
      List(raise(p.Term.IntS64Const(7), "long"))
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(
        handler(
          Some(p.Type.IntS64),
          None,
          List(p.Stmt.Update(selectT(outArg), p.Term.IntS32Const(0), p.Term.IntS32Const(1))),
          Some("long")
        )
      ),
      Nil
    )
    assertEquals(runOut(outer), 1L -> (0, ""))
  }

  test("nested abrupt finally cleans pending exceptions inside out") {
    val nested = p.Stmt.Try(
      List(raise(p.Term.IntS64Const(9), "long", append(2, "Inner"))),
      Nil,
      List(ret)
    )
    val outer = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(7), "int", append(1, "Outer"))),
      Nil,
      List(nested)
    )
    assertEquals(runOut(outer), 21L -> (0, ""))
  }

  test("nested handled raises run independent cleanup finalizers") {
    val inner = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(9), "int", append(7, "Inner"))),
      List(handler(Some(i32), None, Nil, Some("int"))),
      Nil
    )
    val outer = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(3), "int", append(7, "Outer"))),
      List(handler(Some(i32), None, List(inner), Some("int"))),
      Nil
    )
    assertEquals(runOut(outer), 77L -> (0, ""))
  }

  test("finally symbols avoid declarations nested in raise cleanup") {
    val cleanupLocal = named("#finally1_done", p.Type.Bool1)
    val inner = p.Stmt.Try(
      List(ret),
      Nil,
      List(
        raise(
          p.Term.IntS64Const(9),
          "long",
          List(p.Stmt.Var(cleanupLocal, Some(p.Expr.Alias(p.Term.Bool1Const(false)))))
        )
      )
    )
    val outer = p.Stmt.Try(
      List(inner),
      List(handler(Some(p.Type.IntS64), None, Nil, Some("long"))),
      Nil
    )
    val hidden = lower(List(p.Arg(outArg)), List(outer, ret)).entry.collectWhere[p.Stmt] {
      case p.Stmt.Var(n, _, _) if n.symbol.startsWith("#finally") => n.symbol
    }
    assert(!hidden.contains(cleanupLocal.symbol), s"cleanup declaration was not alpha-renamed: $hidden")
  }

  test("a handled raise in finally cannot overwrite the pending exception payload") {
    val cleanup = p.Stmt.Try(
      List(raise(p.Term.IntS32Const(99), "int")),
      List(handler(Some(i32), None, Nil, Some("int"))),
      Nil
    )
    val inner  = p.Stmt.Try(List(raise(p.Term.IntS32Const(7), "int")), Nil, List(cleanup))
    val binder = named("caught", i32)
    val outer = p.Stmt.Try(
      List(inner),
      List(
        handler(
          Some(i32),
          Some(binder),
          List(p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), selectT(binder))),
          Some("int")
        )
      ),
      Nil
    )
    assertEquals(runOut(outer), 7L -> (0, ""))
  }

  test("a handled raise in finally cannot overwrite the pending exception message") {
    val whatT = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Private)
    val what  = named(p.Conventions.ExceptionWhat, whatT)
    def message(value: String) =
      p.Stmt.Mut(selectT(what), p.Expr.Alias(p.Term.StringConst(value)))

    val cleanup = p.Stmt.Try(
      List(message("inner"), raise(p.Term.IntS64Const(9), "long")),
      List(handler(Some(p.Type.IntS64), None, Nil, Some("long"))),
      Nil
    )
    val inner = p.Stmt.Try(
      List(message("outer"), raise(p.Term.IntS32Const(7), "int")),
      Nil,
      List(cleanup)
    )
    val first = named("first", p.Type.IntS8)
    val outer = p.Stmt.Try(
      List(inner),
      List(
        handler(
          Some(i32),
          None,
          List(
            p.Stmt.Var(first, Some(p.Expr.Index(selectT(what), p.Term.IntU32Const(0), p.Type.IntS8))),
            p.Stmt.Update(selectT(outArg), p.Term.IntU32Const(0), selectT(first))
          ),
          Some("int")
        )
      ),
      Nil
    )
    val lowered = lower(List(p.Arg(outArg)), List(outer, ret))
    assertEquals(Verify(lowered, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
    assertEquals(runOut(outer), 111L -> (0, ""))
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
      p.Stmt.Cond(selectT(c), List(assertStmt()), Nil),
      p.Stmt.Update(selectT(arr), selectT(i), selectT(t))
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
    val drainBreaks = out.collectWhere[p.Stmt] {
      case c @ p.Stmt.Cond(p.Term.Select(root, Nil, _), List(p.Stmt.Break), Nil)
          if root.symbol == p.Conventions.AssertedFlag =>
        c
    }
    assertEquals(drainBreaks.size, 2) // loop entry plus the post-assert tail fence
  }

  test("a handled raise inside a loop reaches its handler and continues the loop") {
    val i = named("i", i32)
    val handled = p.Stmt.Try(
      List(raise(selectT(i), "int")),
      List(handler(Some(i32), None, append(1, "Handled"), Some("int"))),
      Nil
    )
    val loop = p.Stmt.ForRange(
      i,
      p.Term.IntS32Const(0),
      p.Term.IntS32Const(3),
      p.Term.IntS32Const(1),
      handled :: append(2, "Tail")
    )
    assertEquals(runOut(loop), 121212L -> (0, ""))
  }

  test("a raise escaping a loop breaks directly at the raise site") {
    val i = named("i", i32)
    val loop = p.Stmt.ForRange(
      i,
      p.Term.IntS32Const(0),
      p.Term.IntS32Const(3),
      p.Term.IntS32Const(1),
      raise(selectT(i), "int") :: append(9, "Unreachable")
    )
    val handled = p.Stmt.Try(
      List(loop),
      List(handler(Some(i32), None, append(1, "Handled"), Some("int"))),
      Nil
    )
    val out    = lower(List(p.Arg(outArg)), List(handled, ret))
    val breaks = out.collectWhere[p.Stmt] { case p.Stmt.Break => () }
    assertEquals(breaks.size, 3) // loop entry, raise site, and the generic tail fence
    val vm   = Interpreter.Vm(out)
    val cell = vm.alloc(4L)
    vm.call(p.Conventions.EntryName, List(outArg.tpe -> V.I(cell)))
    assertEquals(i32At(vm, cell), 1L)
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
