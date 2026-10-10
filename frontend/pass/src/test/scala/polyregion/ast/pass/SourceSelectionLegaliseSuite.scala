package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class SourceSelectionLegaliseSuite extends munit.FunSuite {

  test("erased callable members retain their logical function reference") {
    val callable                = sym("callable")
    val ownerSym                = sym("CallableOwner")
    val ownerTpe: p.Type.Struct = p.Type.Struct(ownerSym, Nil)
    val functionTpe             = p.Type.FnRef(callable)
    val owner                   = named("owner", ownerTpe)
    val selected                = p.Term.Select(owner, List(p.PathStep.Field("function")), functionTpe)
    val local                   = named("local", functionTpe)
    val kernel = entry(
      args = List(p.Arg(owner)),
      body = List(
        p.Stmt.Var(local, Some(p.Expr.Alias(selected))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val logical = program(
      kernel,
      defs = List(p.StructDef(ownerSym, Nil, List(named("function", functionTpe)), Nil))
    )
    val stored = SourceStorageLegalise()(logical, NoopLog)

    assertEquals(stored.defs.head.members.head.tpe, p.Type.IntU8)
    val out = SourceSelectionLegalise(stored, NoopLog)
    assert(out.entry.required.collectAll[p.Term].contains(selected), clues(out.entry.required.body))
  }

  test("erased callable member writes remain logical") {
    val callable                = sym("callable")
    val ownerSym                = sym("CallableOwner")
    val ownerTpe: p.Type.Struct = p.Type.Struct(ownerSym, Nil)
    val functionTpe             = p.Type.FnRef(callable)
    val owner                   = named("owner", ownerTpe)
    val selected: p.Term.Select = p.Term.Select(owner, List(p.PathStep.Field("function")), functionTpe)
    val kernel = entry(
      args = List(p.Arg(owner)),
      body = List(
        p.Stmt.Mut(selected, p.Expr.Alias(p.Term.Poison(functionTpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val logical = program(
      kernel,
      defs = List(p.StructDef(ownerSym, Nil, List(named("function", functionTpe)), Nil))
    )
    val stored = SourceStorageLegalise()(logical, NoopLog)

    val out       = SourceSelectionLegalise(stored, NoopLog)
    val mutations = out.entry.required.collectAll[p.Stmt].collect { case mutation: p.Stmt.Mut => mutation }
    assertEquals(mutations.map(_.name.tpe), List(functionTpe))
  }

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

  test("pointer views that differ only in address space become explicit pointer casts") {
    val symbol                = sym("Holder")
    val tpe                   = p.Type.Struct(symbol, Nil)
    val stored                = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    val viewedTpe             = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val definition            = p.StructDef(symbol, Nil, List(named("pointer", stored)), Nil)
    val holder                = named("holder", tpe)
    val viewed: p.Term.Select = p.Term.Select(holder, List(p.PathStep.Field("pointer")), viewedTpe)
    val kernel = entry(
      args = List(p.Arg(holder)),
      body = List(
        p.Stmt.Var(named("result", viewedTpe), Some(p.Expr.Alias(viewed))),
        p.Stmt.Mut(viewed, p.Expr.Alias(p.Term.NullPtrConst(p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out   = SourceSelectionLegalise(program(kernel, defs = List(definition)), NoopLog).entry.required
    val casts = out.collectAll[p.Expr].collect { case p.Expr.Cast(_, to: p.Type.Ptr) => to }
    assertEquals(casts.toSet, Set(viewedTpe, stored), clues(out.body))
  }

  test("array views that differ only in address space read the same storage") {
    val symbol                = sym("Holder")
    val tpe                   = p.Type.Struct(symbol, Nil)
    val stored                = p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global)
    val viewedTpe             = p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Private)
    val definition            = p.StructDef(symbol, Nil, List(named("values", stored)), Nil)
    val holder                = named("holder", tpe)
    val viewed: p.Term.Select = p.Term.Select(holder, List(p.PathStep.Field("values")), viewedTpe)
    val kernel = entry(
      args = List(p.Arg(holder)),
      body = List(
        p.Stmt.Var(named("result", viewedTpe), Some(p.Expr.Alias(viewed))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourceSelectionLegalise(program(kernel, defs = List(definition)), NoopLog).entry.required
    assert(
      !out.collectAll[p.Expr].exists(e => e.isInstanceOf[p.Expr.Cast] || e.isInstanceOf[p.Expr.BitCast]),
      clues(out.body)
    )
  }
}
