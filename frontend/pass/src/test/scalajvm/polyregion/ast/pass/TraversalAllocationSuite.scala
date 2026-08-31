package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, given}

import java.lang.management.ManagementFactory

class TraversalAllocationSuite extends munit.FunSuite {

  test("visitAll scans a large program without materialising traversal results") {
    val bean = ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assert(bean.isThreadAllocatedMemorySupported)
    bean.setThreadAllocatedMemoryEnabled(true)
    assert(bean.isThreadAllocatedMemoryEnabled)

    val unit = p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    val functions = List.tabulate(2000) { functionIndex =>
      val body = List.tabulate(12) { referenceIndex =>
        val resultType = p.Type.Struct(p.Sym(s"Record${referenceIndex % 31}"), List(p.Type.IntS32))
        p.Stmt.Var(
          p.Named(s"value$referenceIndex", resultType),
          Some(
            p.Expr.Invoke(
              p.Type.FnRef(p.Sym(s"callee${(functionIndex + referenceIndex) % 257}")),
              Nil,
              None,
              Nil,
              resultType
            )
          ),
          isMutable = false
        )
      }
      p.Function(
        p.FunctionDecl(
          p.Sym(s"function$functionIndex"),
          Nil,
          None,
          Nil,
          Nil,
          Nil,
          p.Type.Unit0,
          p.Function.Affinity.Offload
        ),
        body :+ unit,
        p.Function.Visibility.Internal,
        p.Function.FpMode.Relaxed,
        p.CallConvention.RegularCall
      )
    }

    def collect(): Long = functions.foldLeft(0L) { (count, function) =>
      count + function
        .collectWhere[p.Type] {
          case _: p.Type.FnRef  => 1L
          case _: p.Type.Struct => 1L
        }
        .sum
    }

    def visit(): Long = {
      var count = 0L
      functions.foreach(_.visitAll[p.Type] {
        case _: p.Type.FnRef  => count += 1
        case _: p.Type.Struct => count += 1
        case _                => ()
      })
      count
    }

    def allocated(run: () => Long): (Long, Long) = {
      run()
      val before    = bean.getCurrentThreadAllocatedBytes
      var checksum  = 0L
      var iteration = 0
      while (iteration < 10) {
        checksum += run()
        iteration += 1
      }
      checksum -> (bean.getCurrentThreadAllocatedBytes - before)
    }

    val (collectChecksum, collectBytes) = allocated(() => collect())
    val (visitChecksum, visitBytes)     = allocated(() => visit())

    assertEquals(visitChecksum, collectChecksum)
    assert(visitBytes * 2 < collectBytes, clues(visitBytes, collectBytes))
  }
}
