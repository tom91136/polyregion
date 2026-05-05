package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class DeadArgEliminationPassSuite extends munit.FunSuite {

  test("entry function: unused arg is dropped") {
    val used   = arg("x")
    val unused = arg("y")
    val e      = entry(args = List(used, unused), body = List(p.Stmt.Return(select(used.named))))
    val out    = DeadArgEliminationPass(program(e), NoopLog)
    assertEquals(out.entry.args.map(_.named.symbol), List("x"))
  }

  test("entry function: referenced arg is preserved") {
    val a    = arg("a")
    val b    = arg("b")
    val body = List(p.Stmt.Var(named("tmp"), Some(select(a.named))), p.Stmt.Return(select(b.named)))
    val out  = DeadArgEliminationPass(program(entry(args = List(a, b), body = body)), NoopLog)
    assertEquals(out.entry.args.map(_.named.symbol).toSet, Set("a", "b"))
  }

  test("non-entry function: receiver and args are not swept (calling-convention stable)") {
    val recv   = arg("self")
    val unused = arg("p")
    val helper = fn("h", args = List(unused)).copy(receiver = Some(recv))
    val out    = DeadArgEliminationPass(program(entry(), List(helper)), NoopLog)
    val h      = out.functions.head
    assertEquals(h.args.map(_.named.symbol), List("p"))
    assertEquals(h.receiver.map(_.named.symbol), Some("self"))
  }
}
