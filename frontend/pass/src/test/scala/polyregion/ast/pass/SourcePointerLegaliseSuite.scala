package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class SourcePointerLegaliseSuite extends munit.FunSuite {

  private val globalPtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)

  test("an immediately-read cross-space pointer merge becomes a scalar merge") {
    val input        = named("input", globalPtr)
    val value        = named("value")
    val merged       = named("merged", globalPtr)
    val result       = named("result")
    val assignGlobal = p.Stmt.Mut(selectT(merged), p.Expr.Alias(selectT(input)))
    val assignPrivate = p.Stmt.Mut(
      selectT(merged),
      p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Private, p.Region.Opaque)
    )
    val kernel = entry(
      args = List(p.Arg(input)),
      body = List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(merged, None, isMutable = true),
        p.Stmt.Cond(p.Term.Bool1Const(true), List(assignGlobal), List(assignPrivate)),
        p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(merged, List(p.PathStep.Deref), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourcePointerLegalise()(program(kernel), NoopLog).entry.required

    val mergedTypes = out
      .collectAll[p.Stmt]
      .collect {
        case p.Stmt.Var(name, _, _) if name.symbol == merged.symbol => name.tpe
      }
      .toSet
    assertEquals(mergedTypes, Set[p.Type](p.Type.IntS32))
    assertEquals(
      out
        .collectAll[p.Stmt]
        .collect {
          case p.Stmt.Mut(p.Term.Select(root, Nil, p.Type.IntS32), expression) if root.symbol == merged.symbol =>
            expression
        }
        .size,
      2
    )
  }

  test("concrete aggregate objects receive address-space-specialised representations") {
    val boxSym     = sym("Box")
    val box        = p.Type.Struct(boxSym, Nil)
    val boxDef     = p.StructDef(boxSym, Nil, List(named("ptr", globalPtr)), Nil)
    val input      = named("input", globalPtr)
    val value      = named("value")
    val globalBox  = named("globalBox", box)
    val privateBox = named("privateBox", box)
    def member(owner: p.Named): p.Term.Select =
      p.Term.Select(owner, List(p.PathStep.Field("ptr")), globalPtr)
    val poison = p.Expr.Alias(p.Term.Poison(box))
    val kernel = entry(
      args = List(p.Arg(input)),
      body = List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(globalBox, Some(poison), isMutable = true),
        p.Stmt.Mut(member(globalBox), p.Expr.Alias(selectT(input))),
        p.Stmt.Var(privateBox, Some(poison), isMutable = true),
        p.Stmt.Mut(
          member(privateBox),
          p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Private, p.Region.Opaque)
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourcePointerLegalise()(program(kernel, defs = List(boxDef)), NoopLog)
    val privateClone = out.defs
      .find(_.members.exists {
        case p.Named("ptr", p.Type.Ptr(_, p.Type.Space.Private), _) => true
        case _                                                      => false
      })
      .getOrElse(fail("missing private Box specialisation"))
    val privateType = out.entry.required.collectAll[p.Stmt].collectFirst {
      case p.Stmt.Var(name, _, _) if name.symbol == privateBox.symbol => name.tpe
    }
    assertEquals(privateType, Some(p.Type.Struct(privateClone.name, Nil)))
  }

  test("taking the address of a pointer preserves its pointee space and uses private slot storage") {
    val local   = named("local", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local))
    val stale   = p.Type.Ptr(p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global), p.Type.Space.Global)
    val address = named("address", stale)
    val kernel = entry(
      body = List(
        p.Stmt.Var(local, None, isMutable = true),
        p.Stmt.Var(
          address,
          Some(
            p.Expr.RefTo(
              selectT(local),
              None,
              p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global),
              p.Type.Space.Global,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = SourcePointerLegalise()(program(kernel), NoopLog).entry.required
    val expected = p.Type.Ptr(p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local), p.Type.Space.Private)
    assertEquals(
      out.collectAll[p.Stmt].collectFirst { case p.Stmt.Var(name, _, _) if name.symbol == address.symbol => name.tpe },
      Some(expected)
    )
    assertEquals(
      out.collectAll[p.Expr].collectFirst { case ref: p.Expr.RefTo => ref.tpe },
      Some(expected)
    )
  }

  test("taking an indexed address preserves the element type") {
    val input   = named("input", globalPtr)
    val element = named("element", globalPtr)
    val kernel = entry(
      args = List(p.Arg(input)),
      body = List(
        p.Stmt.Var(
          element,
          Some(
            p.Expr.RefTo(
              selectT(input),
              Some(p.Term.IntS32Const(1)),
              p.Type.IntS32,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourcePointerLegalise()(program(kernel), NoopLog).entry.required
    assertEquals(
      out.collectAll[p.Expr].collectFirst { case ref: p.Expr.RefTo => ref.tpe },
      Some(globalPtr)
    )
  }

  test("pointer casts retain the refined source address space") {
    val value  = named("value")
    val source = named("source", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private))
    val casted = named("casted", globalPtr)
    val result = named("result")
    val kernel = entry(
      body = List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(
          source,
          Some(p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Private, p.Region.Opaque))
        ),
        p.Stmt.Var(casted, Some(p.Expr.Cast(selectT(source), globalPtr))),
        p.Stmt.Var(result, Some(p.Expr.Index(selectT(casted), p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = SourcePointerLegalise()(program(kernel), NoopLog).entry.required
    val localPtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    assertEquals(
      out.collectAll[p.Stmt].collectFirst { case p.Stmt.Var(name, _, _) if name.symbol == casted.symbol => name.tpe },
      Some(localPtr)
    )
    assertEquals(
      out.collectAll[p.Expr].collectFirst { case p.Expr.Cast(_, as) => as },
      Some(localPtr)
    )
  }

  test("pointer refinement crosses erased casts and delayed assignment") {
    val privatePtr = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    val erasedPtr  = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val value      = named("value")
    val source     = named("source", privatePtr)
    val erased     = named("erased", erasedPtr)
    val casted     = named("casted", globalPtr)
    val alias      = named("alias", globalPtr)
    val result     = named("result")
    val kernel = entry(
      body = List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(
          source,
          Some(p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Private, p.Region.Opaque))
        ),
        p.Stmt.Var(casted, None, isMutable = true),
        p.Stmt.Var(erased, Some(p.Expr.Cast(selectT(source), erasedPtr))),
        p.Stmt.Mut(selectT(casted), p.Expr.Cast(selectT(erased), globalPtr)),
        p.Stmt.Var(alias, Some(p.Expr.Alias(selectT(casted)))),
        p.Stmt.Var(result, Some(p.Expr.Index(selectT(alias), p.Term.IntS32Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = SourcePointerLegalise()(program(kernel), NoopLog).entry.required
    val pointerTypes = out.collectAll[p.Stmt].collect {
      case p.Stmt.Var(name, _, _) if Set(source.symbol, erased.symbol, casted.symbol, alias.symbol)(name.symbol) =>
        AddressRefinement.spaceOf(name.tpe)
    }
    assert(pointerTypes.forall(_.contains(p.Type.Space.Private)), clues(pointerTypes))
  }
}
