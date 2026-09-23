package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, given}
import PassTest.*

import java.lang.management.ManagementFactory

class AddressRefinementAllocationSuite extends munit.FunSuite {

  private val bean = ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]

  override def beforeAll(): Unit = {
    assert(bean.isThreadAllocatedMemorySupported)
    bean.setThreadAllocatedMemoryEnabled(true)
    assert(bean.isThreadAllocatedMemoryEnabled)
  }

  private def allocated(iterations: Int)(run: () => Long): (Long, Long) = {
    run()
    val before    = bean.getCurrentThreadAllocatedBytes
    var checksum  = 0L
    var iteration = 0
    while (iteration < iterations) {
      checksum += run()
      iteration += 1
    }
    checksum -> (bean.getCurrentThreadAllocatedBytes - before)
  }

  private def input(size: Int): (p.Program, p.Function) = {
    val captureType = p.Type.Struct(sym("Capture"), Nil)
    val pointerType = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val captures    = List.tabulate(size)(index => named(s"capture$index", captureType))
    val body = captures.zipWithIndex.flatMap { case (capture, index) =>
      val pointer = named(s"pointer$index", pointerType)
      val value   = named(s"value$index", p.Type.IntS32)
      List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.Select(capture, List(p.PathStep.Field("data")), pointerType))),
          isMutable = false
        ),
        p.Stmt.Var(
          value,
          Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        )
      )
    }
    val function = entry(
      args = captures.map(p.Arg(_)),
      body = body :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    program(
      function,
      defs = List(p.StructDef(sym("Capture"), Nil, List(named("data", pointerType)), Nil))
    ) -> function
  }

  test("slot lookup allocation remains linear across aggregate roots") {
    val (smallProgram, smallEntry) = input(192)
    val (largeProgram, largeEntry) = input(384)
    def run(program: p.Program, entry: p.Function): Long = {
      val solution = AddressRefinement.solve(program, entry)
      assertEquals(solution.diagnostics, Nil)
      solution.bindings.size.toLong + solution.slots.size
    }

    var warmup = 0
    while (warmup < 8) {
      run(smallProgram, smallEntry)
      run(largeProgram, largeEntry)
      warmup += 1
    }
    val samples = List.fill(5) {
      allocated(iterations = 2)(() => run(smallProgram, smallEntry)) ->
        allocated(iterations = 2)(() => run(largeProgram, largeEntry))
    }
    def median(values: List[Long]): Long = values.sorted.apply(values.size / 2)
    val smallChecksum                    = samples.head._1._1
    val largeChecksum                    = samples.head._2._1
    val smallBytes                       = median(samples.map(_._1._2))
    val largeBytes                       = median(samples.map(_._2._2))

    assertEquals(largeChecksum, smallChecksum * 2)
    assert(largeBytes < smallBytes * 3, clues(smallBytes, largeBytes))
  }
}
