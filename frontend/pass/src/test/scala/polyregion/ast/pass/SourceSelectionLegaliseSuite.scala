package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class SourceSelectionLegaliseSuite extends munit.FunSuite {

  test("same-width field views become explicit bit casts for reads and writes") {
    val symbol                = sym("Bits")
    val tpe                   = p.Type.Struct(symbol, Nil)
    val definition            = p.StructDef(symbol, Nil, List(named("value", p.Type.IntS32)), Nil)
    val bits                  = named("bits", tpe)
    val viewed: p.Term.Select = p.Term.Select(bits, List(p.PathStep.Field("value")), p.Type.Float32)
    val kernel = entry(
      args = List(p.Arg(bits)),
      body = List(
        p.Stmt.Var(named("result", p.Type.Float32), Some(p.Expr.Alias(viewed))),
        p.Stmt.Mut(viewed, p.Expr.Alias(p.Term.Float32Const(1))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = SourceSelectionLegalise(program(kernel, defs = List(definition)), NoopLog).entry.required
    val bitCasts = out.collectAll[p.Expr].count(_.isInstanceOf[p.Expr.BitCast])
    assertEquals(bitCasts, 2, clues(out.body))
    val selected = out.collectAll[p.Term].collect { case select: p.Term.Select if select.steps.nonEmpty => select.tpe }
    assert(selected.forall(_ == p.Type.IntS32), clues(selected))
  }
}
