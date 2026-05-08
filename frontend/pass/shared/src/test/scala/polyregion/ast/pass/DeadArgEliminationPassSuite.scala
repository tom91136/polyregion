package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class DeadArgEliminationPassSuite extends munit.FunSuite {

  test("entry function: positional args are preserved") {
    val used   = arg("x")
    val unused = arg("y")
    val e      = entry(args = List(used, unused), body = List(p.Stmt.Return(select(used.named))))
    val out    = DeadArgEliminationPass(program(e), NoopLog)
    assertEquals(out.entry.args.map(_.named.symbol), List("x", "y"))
  }

  test("entry function: unused module captures are still pruned") {
    val used   = arg("a")
    val unused = arg("u")
    val e      = entry(args = Nil, moduleCaptures = List(used, unused), body = List(p.Stmt.Return(select(used.named))))
    val out    = DeadArgEliminationPass(program(e), NoopLog)
    assertEquals(out.entry.moduleCaptures.map(_.named.symbol), List("a"))
  }

  test("non-entry function: receiver and args are not swept") {
    val recv   = arg("self")
    val unused = arg("p")
    val helper = fn("h", args = List(unused)).copy(receiver = Some(recv))
    val out    = DeadArgEliminationPass(program(entry(), List(helper)), NoopLog)
    val h      = out.functions.head
    assertEquals(h.args.map(_.named.symbol), List("p"))
    assertEquals(h.receiver.map(_.named.symbol), Some("self"))
  }
}
