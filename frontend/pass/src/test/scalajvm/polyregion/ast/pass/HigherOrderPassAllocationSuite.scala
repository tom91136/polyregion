package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, given}
import polyregion.ast.Traversal.*
import PassTest.*

import java.lang.management.ManagementFactory

class HigherOrderPassAllocationSuite extends munit.FunSuite {

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

  private def allocationGrowth(small: () => Long, large: () => Long): (Long, Long, Long, Long) = {
    var warmup = 0
    while (warmup < 10) {
      small()
      large()
      warmup += 1
    }
    val samples = List.fill(5)(allocated(iterations = 3)(small) -> allocated(iterations = 3)(large))
    def median(values: List[Long]): Long = values.sorted.apply(values.size / 2)
    (
      samples.head._1._1,
      median(samples.map(_._1._2)),
      samples.head._2._1,
      median(samples.map(_._2._2))
    )
  }

  test("disabled specialisation logging avoids diagnostic allocation") {
    final class SinkLog(override val enabled: Boolean) extends Log {
      var calls                                         = 0
      def info(message: String, details: String*): Unit = calls += 1
      def subLog(name: String): Log                     = this
    }

    val genericType = p.Type.Var("T")
    val helpers = List.tabulate(128) { index =>
      val value = arg("value", genericType)
      fn(
        s"vendor.${"component" * 32}.$index",
        args = List(value),
        rtn = genericType,
        body = List(p.Stmt.Return(select(value.named))),
        tpeVars = List("T"),
        visibility = p.Function.Visibility.Internal
      )
    }
    val calls = helpers.zipWithIndex.map { case (helper, index) =>
      p.Stmt.Var(
        named(s"result$index"),
        Some(
          p.Expr.Invoke(
            p.Type.FnRef(helper.name),
            List(p.Type.IntS32),
            None,
            List(p.Term.IntS32Const(index)),
            p.Type.IntS32
          )
        ),
        isMutable = false
      )
    }
    val input = program(entry(body = calls :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))), helpers)

    assert(!PluginEntry.NoopLog.enabled)
    assert(!NoopLog.enabled)

    val disabled = SinkLog(enabled = false)
    val enabled  = SinkLog(enabled = true)
    var warmup   = 0
    while (warmup < 10) {
      Specialisation(input, disabled)
      Specialisation(input, enabled)
      warmup += 1
    }
    disabled.calls = 0
    enabled.calls = 0
    val samples = List.fill(5) {
      allocated(iterations = 3)(() => Specialisation(input, disabled).functions.size.toLong) ->
        allocated(iterations = 3)(() => Specialisation(input, enabled).functions.size.toLong)
    }
    def median(values: List[Long]): Long = values.sorted.apply(values.size / 2)
    val disabledChecksum                 = samples.head._1._1
    val enabledChecksum                  = samples.head._2._1
    val disabledBytes                    = median(samples.map(_._1._2))
    val enabledBytes                     = median(samples.map(_._2._2))

    assertEquals(disabledChecksum, enabledChecksum)
    assertEquals(disabled.calls, 0)
    assert(enabled.calls > 0)
    assert(enabledBytes - disabledBytes > 512 * 1024, clues(disabledBytes, enabledBytes))
  }

  test("flat-block inlining accumulation avoids quadratic allocation growth") {
    def input(size: Int): p.Program = {
      val statements = List.tabulate(size) { index =>
        p.Stmt.Var(
          named(s"value$index"),
          Some(p.Expr.Alias(p.Term.IntS32Const(index))),
          isMutable = false
        )
      }
      val helper = fn(
        "large.helper",
        body = statements :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)),
        visibility = p.Function.Visibility.Internal
      )
      val call = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
      program(
        entry(
          body = List(
            p.Stmt.Var(named("call", p.Type.Unit0), Some(call), isMutable = false),
            p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
          )
        ),
        List(helper)
      )
    }

    val small = input(512)
    val large = input(1024)
    val (smallChecksum, smallBytes, largeChecksum, largeBytes) = allocationGrowth(
      () => FnInline(small, NoopLog).entry.get.body.size.toLong,
      () => FnInline(large, NoopLog).entry.get.body.size.toLong
    )

    assertEquals(largeChecksum - smallChecksum, 512L * 3)
    assert(largeBytes < smallBytes * 3, clues(smallBytes, largeBytes))
  }

  test("remote launch scalar-cast accumulation avoids quadratic allocation growth") {
    def input(size: Int): p.Program = {
      val kernel = fn(
        "cast.kernel",
        args = List.tabulate(size)(index => arg(s"value$index", p.Type.IntS64)),
        tpeVars = List("Unused")
      )
      val one = p.Term.IntU32Const(1)
      val launch = p.Spec.RemoteLaunch(
        p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque),
        p.Term.Poison(p.Type.FnRef(kernel.name)),
        List(p.Type.IntS32),
        one,
        one,
        one,
        one,
        one,
        one,
        p.Term.IntU32Const(0),
        List.tabulate(size)(p.Term.IntS32Const.apply)
      )
      program(entry(body = List(p.Stmt.Return(p.Expr.SpecOp(launch)))), List(kernel))
    }

    val small = input(128)
    val large = input(256)
    val (smallChecksum, smallBytes, largeChecksum, largeBytes) = allocationGrowth(
      () => Specialisation(small, NoopLog).entry.get.body.size.toLong,
      () => Specialisation(large, NoopLog).entry.get.body.size.toLong
    )

    assertEquals(largeChecksum - smallChecksum, 128L * 3)
    assert(largeBytes < smallBytes * 3, clues(smallBytes, largeBytes))
  }

  test("remote launch declaration discovery avoids quadratic nesting growth") {
    def input(depth: Int): p.Program = {
      val kernel = fn(
        "nested.cast.kernel",
        args = List(arg("value", p.Type.IntS64)),
        tpeVars = List("Unused")
      )
      val one = p.Term.IntU32Const(1)
      val launch = p.Expr.SpecOp(
        p.Spec.RemoteLaunch(
          p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque),
          p.Term.Poison(p.Type.FnRef(kernel.name)),
          List(p.Type.IntS32),
          one,
          one,
          one,
          one,
          one,
          one,
          p.Term.IntU32Const(0),
          List(p.Term.IntS32Const(1))
        )
      )
      val nested = (0 until depth).foldLeft(List[p.Stmt](p.Stmt.Return(launch))) { (body, index) =>
        List(
          p.Stmt.Var(named(s"nested$index"), Some(p.Expr.Alias(p.Term.IntS32Const(index))), isMutable = false),
          p.Stmt.Cond(p.Term.Bool1Const(true), body, Nil)
        )
      }
      program(entry(body = nested), List(kernel))
    }

    def declarations(program: p.Program): Long =
      Specialisation(program, NoopLog).entry.get.collectWhere[p.Stmt] { case _: p.Stmt.Var => 1L }.sum

    val small = input(128)
    val large = input(256)
    val (smallChecksum, smallBytes, largeChecksum, largeBytes) = allocationGrowth(
      () => declarations(small),
      () => declarations(large)
    )

    assert(largeChecksum > smallChecksum)
    assert(largeBytes < smallBytes * 3, clues(smallBytes, largeBytes))
  }
}
