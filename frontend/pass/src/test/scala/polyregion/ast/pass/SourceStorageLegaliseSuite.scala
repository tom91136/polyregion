package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class SourceStorageLegaliseSuite extends munit.FunSuite {

  test("callable storage, empty bases, and address-taken values are represented before rendering") {
    val emptySym             = sym("Empty")
    val ownerSym             = sym("Owner")
    val empty: p.Type.Struct = p.Type.Struct(emptySym, Nil)
    val owner: p.Type.Struct = p.Type.Struct(ownerSym, Nil)
    val emptyDef             = p.StructDef(emptySym, Nil, Nil, Nil)
    val ownerDef = p.StructDef(
      ownerSym,
      Nil,
      List(named("#base_Empty", empty), named("fn", p.Type.FnRef(sym("f")))),
      List(empty)
    )
    val value  = named("value", owner)
    val base   = named("base", p.Type.Ptr(empty, p.Type.Space.Private))
    val scalar = named("scalar", p.Type.Ptr(p.Type.IntU32, p.Type.Space.Private))
    val kernel = entry(
      body = List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(
          base,
          Some(
            p.Expr.RefTo(
              p.Term.Select(value, List(p.PathStep.Field("#base_Empty")), empty),
              None,
              empty,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Var(
          scalar,
          Some(p.Expr.RefTo(p.Term.IntU32Const(1), None, p.Type.IntU32, p.Type.Space.Private, p.Region.Opaque))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourceStorageLegalise()(program(kernel, defs = List(emptyDef, ownerDef)), NoopLog)
    assertEquals(
      out.defs.find(_.name == ownerSym).getOrElse(fail("missing Owner")).members,
      List(named("fn", p.Type.IntU8))
    )
    assert(out.entry.required.collectAll[p.Expr].exists(_.isInstanceOf[p.Expr.Cast]))
    assert(out.entry.required.body.exists {
      case p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.IntU32Const(1))), true) =>
        name.symbol.startsWith("#source_storage_")
      case _ => false
    })
  }

  test("addresses use the selected lvalue's storage space") {
    val value   = named("value", p.Type.IntU32)
    val pointer = named("pointer", p.Type.Ptr(p.Type.IntU32, p.Type.Space.Private))
    val kernel = entry(
      body = List(
        p.Stmt.Var(value, Some(p.Expr.Alias(p.Term.IntU32Const(1))), isMutable = true),
        p.Stmt.Var(
          pointer,
          Some(
            p.Expr.RefTo(
              p.Term.Select(value, Nil, value.tpe),
              None,
              value.tpe,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourceStorageLegalise()(program(kernel), NoopLog)
    assert(out.entry.required.body.exists {
      case p.Stmt.Var(
            `pointer`,
            Some(p.Expr.RefTo(p.Term.Select(`value`, Nil, _), None, p.Type.IntU32, p.Type.Space.Private, region)),
            _
          ) =>
        region == p.Region.Rooted(value)
      case _ => false
    })
  }
}
