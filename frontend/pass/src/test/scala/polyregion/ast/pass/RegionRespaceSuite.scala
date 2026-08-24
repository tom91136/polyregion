package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class RegionRespaceSuite extends munit.FunSuite {

  private def ptr(s: p.Type.Space) = p.Type.Ptr(p.Type.IntS32, s)

  private def ptrSpacesOf(f: p.Function, sym: String): Set[p.Type.Space] =
    (f.collectAll[p.Stmt].collect { case p.Stmt.Var(n, _, _) if n.symbol == sym => n.tpe } :::
      f.collectAll[p.Term].collect { case p.Term.Select(n, _, _) if n.symbol == sym => n.tpe }).collect {
      case p.Type.Ptr(_, s) => s
    }.toSet
  private def ptrSpacesOf(f: Option[p.Function], sym: String): Set[p.Type.Space] = ptrSpacesOf(f.required, sym)

  private def refToSpaces(f: p.Function): Set[p.Type.Space] =
    f.collectAll[p.Expr].collect { case p.Expr.RefTo(_, _, _, s, _) => s }.toSet
  private def refToSpaces(f: Option[p.Function]): Set[p.Type.Space] = refToSpaces(f.required)

  test("a Global pointer rooted at a Local resource is re-stamped Local (decl, uses, and the RefTo)") {
    val local = named("local", ptr(p.Type.Space.Local))
    val s     = named("s", ptr(p.Type.Space.Global))
    val refTo =
      p.Expr.RefTo(selectT(local), Some(p.Term.IntS64Const(1)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
    val use =
      p.Stmt.Var(named("u", p.Type.IntS32), Some(p.Expr.Index(selectT(s), p.Term.IntS64Const(0), p.Type.IntS32)))
    val e   = entry(body = List(p.Stmt.Var(s, Some(refTo)), use, p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = RegionRespace(program(e), NoopLog)
    assertEquals(ptrSpacesOf(out.entry, "s"), Set[p.Type.Space](p.Type.Space.Local))
    assertEquals(refToSpaces(out.entry), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("a pointer already in its root's space is left untouched") {
    val g = named("g", ptr(p.Type.Space.Global))
    val s = named("s", ptr(p.Type.Space.Global))
    val refTo =
      p.Expr.RefTo(selectT(g), Some(p.Term.IntS64Const(1)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
    val e   = entry(body = List(p.Stmt.Var(s, Some(refTo)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = RegionRespace(program(e), NoopLog)
    assertEquals(ptrSpacesOf(out.entry, "s"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("default provenance keeps first reassignment semantics for pointer arguments") {
    val left  = named("left", ptr(p.Type.Space.Global))
    val right = named("right", ptr(p.Type.Space.Global))
    val e = entry(
      args = List(p.Arg(left), p.Arg(right)),
      body = List(
        p.Stmt.Mut(selectT(left), p.Expr.Alias(selectT(right))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val derived = Provenance.derivedIn(e)

    assertEquals(derived.get(left), Some(p.Region.Rooted(right)))
  }

  test("address-spaced array roots join conservatively") {
    val local  = named("local", p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Local))
    val global = named("global", p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global))

    assertEquals(
      Provenance.joinRegions(p.Region.Rooted(local), p.Region.Rooted(global)),
      p.Region.Opaque
    )
  }

  test("a stale term type still resolves the declaration's local provenance by symbol") {
    val local = named("local", ptr(p.Type.Space.Local))
    val stale = named("local", ptr(p.Type.Space.Global))
    val s     = named("s", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(
          s,
          Some(
            p.Expr
              .RefTo(selectT(stale), Some(p.Term.IntS64Const(1)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "s"), Set[p.Type.Space](p.Type.Space.Local))
    assertEquals(refToSpaces(out.entry), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("pointer provenance survives integer address arithmetic") {
    val local    = named("local", ptr(p.Type.Space.Local))
    val address  = named("address", p.Type.IntU64)
    val advanced = named("advanced", p.Type.IntU64)
    val result   = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(address, Some(p.Expr.Cast(selectT(local), p.Type.IntU64))),
        p.Stmt.Var(
          advanced,
          Some(p.Expr.IntrOp(p.Intr.Add(selectT(address), p.Term.IntU64Const(4), p.Type.IntU64)))
        ),
        p.Stmt.Var(result, Some(p.Expr.Cast(selectT(advanced), result.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("destructive integer arithmetic does not preserve pointer provenance") {
    val local   = named("local", ptr(p.Type.Space.Local))
    val address = named("address", p.Type.IntU64)
    val zero    = named("zero", p.Type.IntU64)
    val result  = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(address, Some(p.Expr.Cast(selectT(local), p.Type.IntU64))),
        p.Stmt.Var(zero, Some(p.Expr.IntrOp(p.Intr.Mul(selectT(address), p.Term.IntU64Const(0), p.Type.IntU64)))),
        p.Stmt.Var(result, Some(p.Expr.Cast(selectT(zero), result.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("narrow integer casts do not preserve pointer provenance") {
    val local     = named("local", ptr(p.Type.Space.Local))
    val truncated = named("truncated", p.Type.IntU32)
    val result    = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(truncated, Some(p.Expr.Cast(selectT(local), p.Type.IntU32))),
        p.Stmt.Var(result, Some(p.Expr.Cast(selectT(truncated), result.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("destructive reassignment clears encoded pointer provenance") {
    val local   = named("local", ptr(p.Type.Space.Local))
    val address = named("address", p.Type.IntU64)
    val result  = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(address, Some(p.Expr.Cast(selectT(local), p.Type.IntU64)), isMutable = true),
        p.Stmt.Mut(
          selectT(address),
          p.Expr.IntrOp(p.Intr.Mul(selectT(address), p.Term.IntU64Const(0), p.Type.IntU64))
        ),
        p.Stmt.Var(result, Some(p.Expr.Cast(selectT(address), result.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("a stale pointer argument term resolves its canonical address space") {
    val local  = named("local", ptr(p.Type.Space.Local))
    val stale  = named("local", ptr(p.Type.Space.Global))
    val result = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      args = List(p.Arg(local)),
      body = List(
        p.Stmt.Var(result, Some(p.Expr.Cast(selectT(stale), result.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("a declaration-only pointer takes provenance from its first assignment") {
    val local  = named("local", ptr(p.Type.Space.Local))
    val slot   = named("slot", ptr(p.Type.Space.Global))
    val result = named("result", ptr(p.Type.Space.Global))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(slot, None, isMutable = true),
        p.Stmt.Mut(selectT(slot), p.Expr.Alias(selectT(local))),
        p.Stmt.Var(result, Some(p.Expr.Alias(selectT(slot)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "slot"), Set[p.Type.Space](p.Type.Space.Local))
    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("pointer provenance preserves a selected field's address space") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holder    = named("holder", holderTpe)
    val middle    = named("middle", holderTpe)
    val copied    = named("copied", holderTpe)
    val local     = named("local", localPtr)
    val base      = named("base", globalPtr)
    val end       = named("end", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(
          p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr),
          p.Expr.Alias(selectT(local))
        ),
        p.Stmt.Var(middle, Some(p.Expr.Alias(selectT(holder)))),
        p.Stmt.Var(copied, Some(p.Expr.Alias(selectT(middle)))),
        p.Stmt.Var(base, Some(p.Expr.Alias(p.Term.Select(copied, List(p.PathStep.Field("data")), globalPtr)))),
        p.Stmt.Var(
          end,
          Some(
            p.Expr
              .RefTo(selectT(base), Some(p.Term.IntS64Const(4)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "base"), Set[p.Type.Space](p.Type.Space.Local))
    assertEquals(ptrSpacesOf(out.entry, "end"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("a pointer cast honours a patched aggregate field's provenance") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holder    = named("holder", holderTpe)
    val copied    = named("copied", holderTpe)
    val local     = named("local", localPtr)
    val casted    = named("casted", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(
          p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr),
          p.Expr.Alias(selectT(local))
        ),
        p.Stmt.Var(copied, Some(p.Expr.Alias(selectT(holder)))),
        p.Stmt.Var(
          casted,
          Some(p.Expr.Cast(p.Term.Select(copied, List(p.PathStep.Field("data")), globalPtr), globalPtr))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "casted"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("copying a nested aggregate preserves its pointer-field provenance") {
    val innerSym                = sym("Inner")
    val outerSym                = sym("Outer")
    val localPtr                = ptr(p.Type.Space.Local)
    val globalPtr               = ptr(p.Type.Space.Global)
    val innerTpe: p.Type.Struct = p.Type.Struct(innerSym, Nil)
    val outerTpe                = p.Type.Struct(outerSym, Nil)
    val inner                   = named("inner", innerTpe)
    val outer                   = named("outer", outerTpe)
    val local                   = named("local", localPtr)
    val result                  = named("result", globalPtr)
    val innerDef                = p.StructDef(innerSym, Nil, List(named("data", globalPtr)), Nil)
    val outerDef                = p.StructDef(outerSym, Nil, List(named("inner", innerTpe)), List(innerTpe))
    val e = entry(
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(p.Term.Select(inner, List(p.PathStep.Field("data")), globalPtr), p.Expr.Alias(selectT(local))),
        p.Stmt.Mut(p.Term.Select(outer, List(p.PathStep.Field("inner")), innerTpe), p.Expr.Alias(selectT(inner))),
        p.Stmt.Var(
          result,
          Some(
            p.Expr.Alias(
              p.Term.Select(outer, List(p.PathStep.Field("inner"), p.PathStep.Field("data")), globalPtr)
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(innerDef, outerDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Local))
  }

  test("aggregate loads do not confuse storage and pointer-field provenance") {
    val holderSym = sym("Holder")
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val storage   = named("storage", p.Type.Ptr(holderTpe, p.Type.Space.Local))
    val copied    = named("copied", holderTpe)
    val result    = named("result", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val e = entry(
      args = List(p.Arg(storage)),
      body = List(
        p.Stmt.Var(
          copied,
          Some(p.Expr.Alias(p.Term.Select(storage, List(p.PathStep.Deref), holderTpe)))
        ),
        p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(copied, List(p.PathStep.Field("data")), globalPtr)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("a later aggregate mutation does not flow backward through an earlier copy") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holder    = named("holder", holderTpe)
    val copied    = named("copied", holderTpe)
    val local     = named("local", localPtr)
    val result    = named("result", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val e = entry(
      args = List(p.Arg(holder)),
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Var(copied, Some(p.Expr.Alias(selectT(holder))), isMutable = true),
        p.Stmt.Mut(
          p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr),
          p.Expr.Alias(selectT(local))
        ),
        p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(copied, List(p.PathStep.Field("data")), globalPtr)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("a one-sided conditional field write is conservatively opaque") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holder    = named("holder", holderTpe)
    val condition = named("condition", p.Type.Bool1)
    val local     = named("local", localPtr)
    val result    = named("result", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val writeLocal = p.Stmt.Mut(
      p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr),
      p.Expr.Alias(selectT(local))
    )
    val e = entry(
      args = List(p.Arg(holder), p.Arg(condition)),
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Cond(selectT(condition), List(writeLocal), Nil),
        p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("a zero-trip loop field write is conservatively opaque") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holder    = named("holder", holderTpe)
    val local     = named("local", localPtr)
    val result    = named("result", globalPtr)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)
    val e = entry(
      args = List(p.Arg(holder)),
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.While(
          p.Term.Bool1Const(false),
          List(
            p.Stmt.Mut(
              p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr),
              p.Expr.Alias(selectT(local))
            )
          )
        ),
        p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(holder, List(p.PathStep.Field("data")), globalPtr)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(holderDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("recursive aggregate copies have a finite slot domain") {
    val nodeSym                = sym("Node")
    val localPtr               = ptr(p.Type.Space.Local)
    val globalPtr              = ptr(p.Type.Space.Global)
    val nodeTpe: p.Type.Struct = p.Type.Struct(nodeSym, Nil)
    val node                   = named("node", nodeTpe)
    val local                  = named("local", localPtr)
    val result                 = named("result", globalPtr)
    val nodeDef = p.StructDef(
      nodeSym,
      Nil,
      List(named("next", nodeTpe), named("data", globalPtr)),
      List(nodeTpe)
    )
    val e = entry(
      args = List(p.Arg(node)),
      body = List(
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(
          p.Term.Select(node, List(p.PathStep.Field("data")), globalPtr),
          p.Expr.Alias(selectT(local))
        ),
        p.Stmt.While(
          p.Term.Bool1Const(false),
          List(
            p.Stmt.Mut(
              p.Term.Select(node, List(p.PathStep.Field("next")), nodeTpe),
              p.Expr.Alias(selectT(node))
            )
          )
        ),
        p.Stmt.Var(
          result,
          Some(
            p.Expr.Alias(
              p.Term.Select(node, List(p.PathStep.Field("next"), p.PathStep.Field("data")), globalPtr)
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = RegionRespace(program(e, defs = List(nodeDef)), NoopLog)

    assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
  }

  test("conditional aggregate copies conservatively join pointer-field provenance") {
    val holderSym = sym("Holder")
    val localPtr  = ptr(p.Type.Space.Local)
    val globalPtr = ptr(p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", globalPtr)), Nil)

    def build(reverse: Boolean): p.Function = {
      val local       = named("local", localPtr)
      val global      = named("global", globalPtr)
      val left        = named("left", holderTpe)
      val right       = named("right", holderTpe)
      val selected    = named("selected", holderTpe)
      val result      = named("result", globalPtr)
      val chooseLeft  = p.Stmt.Mut(selectT(selected), p.Expr.Alias(selectT(left)))
      val chooseRight = p.Stmt.Mut(selectT(selected), p.Expr.Alias(selectT(right)))
      entry(
        args = List(p.Arg(global)),
        body = List(
          p.Stmt.Var(
            local,
            Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(8), p.Type.Space.Local, p.Region.Rooted(local)))
          ),
          p.Stmt.Mut(
            p.Term.Select(left, List(p.PathStep.Field("data")), globalPtr),
            p.Expr.Alias(selectT(local))
          ),
          p.Stmt.Mut(
            p.Term.Select(right, List(p.PathStep.Field("data")), globalPtr),
            p.Expr.Alias(selectT(global))
          ),
          p.Stmt.Cond(
            p.Term.Bool1Const(true),
            List(if (reverse) chooseRight else chooseLeft),
            List(if (reverse) chooseLeft else chooseRight)
          ),
          p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Select(selected, List(p.PathStep.Field("data")), globalPtr)))),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      )
    }

    List(false, true).foreach { reverse =>
      val input   = build(reverse)
      val derived = Provenance.derivedIn(input, trackSlots = true)
      val result  = input.body.collectFirst { case p.Stmt.Var(n, _, _) if n.symbol == "result" => n }.get
      assertEquals(Provenance.at(derived, selectT(result)), p.Region.Opaque)

      val out = RegionRespace(program(input, defs = List(holderDef)), NoopLog)
      assertEquals(ptrSpacesOf(out.entry, "result"), Set[p.Type.Space](p.Type.Space.Global))
    }
  }
}
