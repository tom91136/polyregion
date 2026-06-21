package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class DeadArgEliminationSuite extends munit.FunSuite {

  test("entry function: positional args are preserved") {
    val used   = arg("x")
    val unused = arg("y")
    val e      = entry(args = List(used, unused), body = List(p.Stmt.Return(select(used.named))))
    val out    = DeadArgElimination(program(e), NoopLog)
    assertEquals(out.entry.args.map(_.named.symbol), List("x", "y"))
  }

  // the entry's params are the kernel ABI the JVM macro packs before this pass runs; pruning them
  // would desync host mirroring from the kernel
  test("entry function: module captures are kept even when unused") {
    val used   = arg("a")
    val unused = arg("u")
    val e      = entry(args = Nil, moduleCaptures = List(used, unused), body = List(p.Stmt.Return(select(used.named))))
    val out    = DeadArgElimination(program(e), NoopLog)
    assertEquals(out.entry.moduleCaptures.map(_.named.symbol), List("a", "u"))
  }

  test("non-entry function: receiver and args are not swept") {
    val recv   = arg("self")
    val unused = arg("p")
    val helper = fn("h", args = List(unused)).copy(receiver = Some(recv))
    val out    = DeadArgElimination(program(entry(), List(helper)), NoopLog)
    val h      = out.functions.head
    assertEquals(h.args.map(_.named.symbol), List("p"))
    assertEquals(h.receiver.map(_.named.symbol), Some("self"))
  }
}
