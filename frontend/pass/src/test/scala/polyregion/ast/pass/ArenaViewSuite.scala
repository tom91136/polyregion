package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class ArenaViewSuite extends munit.FunSuite {

  private val nodeSym = p.Sym("Node")
  private val iterSym = p.Sym("Iter")
  private val capSym  = p.Sym("Cap")

  private val nodeTpe = p.Type.Struct(nodeSym, Nil)
  private val iterTpe = p.Type.Struct(iterSym, Nil)
  private val capTpe  = p.Type.Struct(capSym, Nil)

  private val defs = List(
    p.StructDef(nodeSym, Nil, List(named("val", p.Type.IntS32)), Nil),
    p.StructDef(iterSym, Nil, List(named("ptr", p.Type.Ptr(nodeTpe, p.Type.Space.Global))), Nil),
    p.StructDef(capSym, Nil, Nil, Nil)
  )

  // a stack-local iterator (not reachable from the capture arg) whose node pointer is chased into arena
  // memory: `p = &itVal; p->ptr->val = 42` - a mutation crossing from a real local pointer into the arena
  private def buildEntry(): p.Function = {
    val capArg = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val itVal  = named("itVal", iterTpe)
    val pp     = named("p", p.Type.Ptr(iterTpe, p.Type.Space.Global))
    entry(
      args = List(capArg),
      body = List(
        p.Stmt.Var(itVal, None, isMutable = true),
        p.Stmt.Var(
          pp,
          Some(p.Expr.RefTo(selectT(itVal), None, iterTpe, p.Type.Space.Global, p.Region.Opaque)),
          isMutable = false
        ),
        p.Stmt.Mut(
          p.Term.Select(pp, List(p.PathStep.Field("ptr"), p.PathStep.Field("val")), p.Type.IntS32),
          p.Expr.Alias(p.Term.IntS32Const(42))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
  }

  test("mutation through a stack-local iterator's node pointer resolves to an arena store, not a stale field select") {
    val program = p.Program(buildEntry(), Nil, defs)
    val result  = ArenaView(program, NoopLog)
    // no surviving select may reach through a field ArenaView retyped to i64
    val staleFieldSelects = result.entry.collectAll[p.Term].collect {
      case s @ p.Term.Select(_, steps, _) if steps.size >= 2 && steps.contains(p.PathStep.Field("val")) => s
    }
    assertEquals(staleFieldSelects, Nil, result.entry.body.map(_.repr).mkString("\n"))
  }
}
