package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class ArenaLowerSuite extends munit.FunSuite {

  test("arena atomic and volatile operations rebase offset pointers") {
    val capSym  = sym("Cap")
    val ptrTpe  = p.Type.Ptr(p.Type.IntU32, p.Type.Space.Global)
    val capTpe  = p.Type.Struct(capSym, Nil)
    val cap     = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val pointer = named("pointer", ptrTpe)
    val loaded  = named("loaded", p.Type.IntU32)
    val stored  = named("stored", p.Type.Unit0)
    val swapped = named("swapped", p.Type.IntU32)
    val capDef  = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          isMutable = false
        ),
        p.Stmt.Var(
          loaded,
          Some(p.Expr.SpecOp(p.Spec.GpuVolatileLoad(p.Term.Select(pointer, Nil, ptrTpe), p.Type.IntU32))),
          isMutable = false
        ),
        p.Stmt.Var(
          stored,
          Some(p.Expr.SpecOp(p.Spec.GpuVolatileStore(p.Term.Select(pointer, Nil, ptrTpe), p.Term.IntU32Const(7)))),
          isMutable = false
        ),
        p.Stmt.Var(
          swapped,
          Some(
            p.Expr.SpecOp(
              p.Spec.GpuAtomicRMW(
                p.AtomicOp.Xchg,
                p.Term.Select(pointer, Nil, ptrTpe),
                p.Term.IntU32Const(9),
                p.MemScope.Device,
                p.MemOrder.Relaxed,
                p.Type.IntU32
              )
            )
          ),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry
    val pointers = out.collectAll[p.Expr].collect {
      case p.Expr.SpecOp(x: p.Spec.GpuAtomicRMW)     => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileLoad)  => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileStore) => x.ptr
    }
    assertEquals(pointers.size, 3)
    assert(pointers.forall {
      case p.Term.Select(root, Nil, `ptrTpe`) => root.symbol.startsWith("#ab")
      case _                                  => false
    })
  }

  test("flat arena lowering exposes a byte arena at the function boundary") {
    val capSym    = sym("Cap")
    val capStruct = p.Type.Struct(capSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capStruct, p.Type.Space.Global))
    val arena     = named("#arena_base", BytePtr)
    val self      = named("self", cap.tpe)
    val capDef    = p.StructDef(capSym, Nil, List(named("x")), Nil)
    val readX =
      p.Stmt.Var(named("out"), Some(p.Expr.Alias(p.Term.Select(self, List(p.PathStep.Field("x")), p.Type.IntS32))))
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(self, Some(p.Expr.Alias(p.Term.Select(cap, Nil, cap.tpe))), isMutable = false),
        readX,
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assertEquals(out.args.map(_.named), List(arena))
    assertEquals(
      out.body.headOption,
      Some(p.Stmt.Var(cap, Some(p.Expr.Cast(p.Term.Select(arena, Nil, BytePtr), cap.tpe)), isMutable = false))
    )
    assertEquals(
      out.body.collectFirst { case p.Stmt.Var(n, Some(expr), _) if n.symbol == "out" => expr },
      Some(p.Expr.Alias(p.Term.Select(self, List(p.PathStep.Field("x")), p.Type.IntS32)))
    )
  }

  test("flat arena lowering clears an element boundary after erasing the capture pointee") {
    val capSym    = sym("Cap")
    val capStruct = p.Type.Struct(capSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capStruct, p.Type.Space.Global))
    val boundary = p.Arg.Boundary(
      p.Arg.Access.Read,
      p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))
    )
    val capDef = p.StructDef(capSym, Nil, List(named("x")), Nil)
    val e      = entry(args = List(p.Arg(cap, boundary = Some(boundary))))

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assertEquals(out.args.head.boundary, None)
  }

  test("address of arena-rooted inline array is kept as an arena offset") {
    val capSym  = sym("Cap")
    val dataTpe = p.Type.Arr(p.Type.IntS8, 4, p.Type.Space.Global)
    val capTpe  = p.Type.Struct(capSym, Nil)
    val cap     = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val ptr     = named("p", p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global))
    val ch      = named("ch", p.Type.IntS8)
    val capDef  = p.StructDef(capSym, Nil, List(named("data", dataTpe)), Nil)
    val data    = p.Term.Select(cap, List(p.PathStep.Field("data")), dataTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          ptr,
          Some(
            p.Expr.RefTo(data, Some(p.Term.IntS32Const(0)), p.Type.IntS8, p.Type.Space.Global, p.Region.Rooted(cap))
          ),
          isMutable = false
        ),
        p.Stmt.Var(
          ch,
          Some(p.Expr.Index(p.Term.Select(ptr, Nil, ptr.tpe), p.Term.IntS64Const(0), p.Type.IntS8)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Var(_, Some(p.Expr.IntrOp(p.Intr.Sub(_, _, p.Type.IntU64))), _) => true
      case _                                                                      => false
    })
    assert(out.body.collectFirst {
      case p.Stmt.Var(n, Some(p.Expr.Cast(_, p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global))), _)
          if n.symbol == ptr.symbol =>
        ()
    }.nonEmpty)
    assert(out.body.collectFirst { case p.Stmt.Var(`ch`, Some(e), _) => e }.exists {
      case p.Expr.Index(p.Term.Select(root, Nil, _), _, p.Type.IntS8) => root.symbol.startsWith("#ab")
      case _                                                          => false
    })
  }

  test("direct index of arena-rooted inline array remains a rooted array access") {
    val capSym    = sym("Cap")
    val bufferSym = sym("Buffer")
    val dataTpe   = p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global)
    val bufferTpe = p.Type.Struct(bufferSym, Nil)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val bufferPtr = named("buffer", p.Type.Ptr(bufferTpe, p.Type.Space.Global))
    val ch        = named("ch", p.Type.IntS32)
    val capDef    = p.StructDef(capSym, Nil, List(named("buffer", bufferTpe)), Nil)
    val bufferDef = p.StructDef(bufferSym, Nil, List(named("slots", dataTpe), named("head")), Nil, isUnion = true)
    val buffer    = p.Term.Select(cap, List(p.PathStep.Field("buffer")), bufferTpe)
    val slots     = p.Term.Select(bufferPtr, List(p.PathStep.Field("slots")), dataTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          bufferPtr,
          Some(p.Expr.RefTo(buffer, None, bufferTpe, p.Type.Space.Global, p.Region.Rooted(cap))),
          isMutable = false
        ),
        p.Stmt.Var(ch, Some(p.Expr.Index(slots, p.Term.IntS64Const(2), p.Type.IntS32)), isMutable = false),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, bufferDef)), NoopLog).entry

    assert(out.body.collectFirst { case p.Stmt.Var(`ch`, Some(e), _) => e }.exists {
      case p.Expr.Index(p.Term.Select(root, List(p.PathStep.Field("slots")), `dataTpe`), _, p.Type.IntS32) =>
        root.symbol.startsWith("#ab")
      case _ => false
    })
  }

  test("address of arena offset pointer element remains an arena offset") {
    val capSym = sym("Cap")
    val ptrTpe = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)
    val capTpe = p.Type.Struct(capSym, Nil)
    val cap    = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val p0     = named("p", ptrTpe)
    val p2     = named("p2", ptrTpe)
    val ch     = named("ch", p.Type.IntS8)
    val capDef = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val data   = p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(p0, Some(p.Expr.Alias(data)), isMutable = false),
        p.Stmt.Var(
          p2,
          Some(
            p.Expr.RefTo(
              p.Term.Select(p0, Nil, ptrTpe),
              Some(p.Term.IntS64Const(2)),
              p.Type.IntS8,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          isMutable = false
        ),
        p.Stmt.Var(
          ch,
          Some(p.Expr.Index(p.Term.Select(p2, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS8)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Var(_, Some(p.Expr.IntrOp(p.Intr.Add(_, _, p.Type.IntU64))), _) => true
      case _                                                                      => false
    })
    assert(out.body.collectFirst {
      case p.Stmt.Var(n, Some(p.Expr.Cast(_, p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global))), _)
          if n.symbol == p2.symbol =>
        ()
    }.nonEmpty)
    assert(out.body.collectFirst { case p.Stmt.Var(`ch`, Some(e), _) => e }.exists {
      case p.Expr.Index(p.Term.Select(root, Nil, _), _, p.Type.IntS8) => root.symbol.startsWith("#ab")
      case _                                                          => false
    })
  }

  test("pointer loaded through an arena offset remains an arena offset") {
    val capSym    = sym("Cap")
    val ptrTpe    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val ptrPtrTpe = p.Type.Ptr(ptrTpe, p.Type.Space.Global)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val outer     = named("outer", ptrPtrTpe)
    val inner     = named("inner", ptrTpe)
    val value     = named("value", p.Type.IntS32)
    val capDef    = p.StructDef(capSym, Nil, List(named("data", ptrPtrTpe)), Nil)
    val data      = p.Term.Select(cap, List(p.PathStep.Field("data")), ptrPtrTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(outer, Some(p.Expr.Alias(data)), isMutable = false),
        p.Stmt.Var(
          inner,
          Some(p.Expr.Index(p.Term.Select(outer, Nil, ptrPtrTpe), p.Term.IntS64Const(0), ptrTpe)),
          isMutable = false
        ),
        p.Stmt.Var(
          value,
          Some(p.Expr.Index(p.Term.Select(inner, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assert(
      out.body.collectFirst { case p.Stmt.Var(`value`, Some(expr), _) => expr }.exists {
        case p.Expr.Index(p.Term.Select(root, Nil, _), _, p.Type.IntS32) => root.symbol.startsWith("#ab")
        case _                                                           => false
      },
      out.repr
    )
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef))), Nil)
  }

  test("arena offset stored through a local aggregate alias remains an arena offset") {
    val capSym    = sym("Cap")
    val holderSym = sym("Holder")
    val ptrTpe    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val holder    = named("holder", holderTpe)
    val holderPtr = named("holderPtr", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val loaded    = named("loaded", ptrTpe)
    val value     = named("value", p.Type.IntS32)
    val capDef    = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", ptrTpe)), Nil)
    val data      = p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)
    val slot =
      p.Term.Select(holderPtr, List(p.PathStep.Field("data")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Var(
          holderPtr,
          Some(
            p.Expr.RefTo(
              p.Term.Select(holder, Nil, holderTpe),
              None,
              holderTpe,
              p.Type.Space.Private,
              p.Region.Rooted(holder)
            )
          ),
          isMutable = false
        ),
        p.Stmt.Mut(slot, p.Expr.Alias(data)),
        p.Stmt.Var(
          loaded,
          Some(p.Expr.Cast(p.Term.Select(holder, List(p.PathStep.Field("data")), ptrTpe), ptrTpe)),
          isMutable = false
        ),
        p.Stmt.Var(
          value,
          Some(p.Expr.Index(p.Term.Select(loaded, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, holderDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Mut(`slot`, p.Expr.Alias(p.Term.Select(root, Nil, _))) => root.symbol.startsWith("#ab")
      case _                                                             => false
    })
    assert(out.body.collectFirst { case p.Stmt.Var(`value`, Some(expr), _) => expr }.exists {
      case p.Expr.Index(p.Term.Select(`loaded`, Nil, _), _, p.Type.IntS32) => true
      case _                                                               => false
    })
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef, holderDef))), Nil)
  }

  test("arena pointer fields stay real across aggregate copies and mutations") {
    val capSym    = sym("Cap")
    val holderSym = sym("Holder")
    val nodeSym   = sym("Node")
    val nodeTpe   = p.Type.Struct(nodeSym, Nil)
    val ptrTpe    = p.Type.Ptr(nodeTpe, p.Type.Space.Global)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val holder    = named("holder", holderTpe)
    val holderPtr = named("holderPtr", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val copied    = named("copied", holderTpe)
    val loaded    = named("loaded", ptrTpe)
    val value     = named("value", p.Type.IntS32)
    val capDef    = p.StructDef(capSym, Nil, List(named("head", ptrTpe)), Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("node", ptrTpe)), Nil)
    val nodeDef = p.StructDef(
      nodeSym,
      Nil,
      List(named("next", ptrTpe), named("value", p.Type.IntS32)),
      Nil
    )
    val holderSlot = p.Term.Select(holderPtr, List(p.PathStep.Field("node")), ptrTpe).asInstanceOf[p.Term.Select]
    val copiedSlot = p.Term.Select(copied, List(p.PathStep.Field("node")), ptrTpe).asInstanceOf[p.Term.Select]
    val next = p.Term.Select(
      copied,
      List(p.PathStep.Field("node"), p.PathStep.Field("next")),
      ptrTpe
    )
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Var(
          holderPtr,
          Some(
            p.Expr.RefTo(
              p.Term.Select(holder, Nil, holderTpe),
              None,
              holderTpe,
              p.Type.Space.Private,
              p.Region.Rooted(holder)
            )
          ),
          isMutable = false
        ),
        p.Stmt.Mut(
          holderSlot,
          p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("head")), ptrTpe))
        ),
        p.Stmt.Var(copied, Some(p.Expr.Alias(p.Term.Select(holder, Nil, holderTpe))), isMutable = true),
        p.Stmt.Mut(copiedSlot, p.Expr.Alias(next)),
        p.Stmt.Var(loaded, Some(p.Expr.Cast(copiedSlot, ptrTpe)), isMutable = false),
        p.Stmt.Var(
          value,
          Some(
            p.Expr.Alias(
              p.Term.Select(loaded, List(p.PathStep.Field("value")), p.Type.IntS32)
            )
          ),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, holderDef, nodeDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Mut(`copiedSlot`, p.Expr.Alias(p.Term.Select(root, Nil, _))) => root.symbol.startsWith("#ab")
      case _                                                                   => false
    })
    assert(out.body.collectFirst { case p.Stmt.Var(`value`, Some(expr), _) => expr }.exists {
      case p.Expr.Alias(p.Term.Select(`loaded`, List(p.PathStep.Field("value")), p.Type.IntS32)) => true
      case _                                                                                     => false
    })
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef, holderDef, nodeDef))), Nil)
  }

  test("arena offset loaded through a private pointer slot is rebased") {
    val capSym      = sym("Cap")
    val closureSym  = sym("Closure")
    val ptrTpe      = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val ptrSlotTpe  = p.Type.Ptr(ptrTpe, p.Type.Space.Private)
    val capTpe      = p.Type.Struct(capSym, Nil)
    val closureTpe  = p.Type.Struct(closureSym, Nil)
    val cap         = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val offset      = named("offset", ptrTpe)
    val closure     = named("closure", closureTpe)
    val loaded      = named("loaded", ptrTpe)
    val value       = named("value", p.Type.IntS32)
    val capDef      = p.StructDef(capSym, Nil, List(named("value", p.Type.IntS32)), Nil)
    val closureDef  = p.StructDef(closureSym, Nil, List(named("slot", ptrSlotTpe)), Nil)
    val closureSlot = p.Term.Select(closure, List(p.PathStep.Field("slot")), ptrSlotTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          offset,
          Some(
            p.Expr.RefTo(
              p.Term.Select(cap, List(p.PathStep.Field("value")), p.Type.IntS32),
              None,
              p.Type.IntS32,
              p.Type.Space.Global,
              p.Region.Rooted(cap)
            )
          ),
          isMutable = true
        ),
        p.Stmt.Var(closure, None, isMutable = true),
        p.Stmt.Mut(
          closureSlot.asInstanceOf[p.Term.Select],
          p.Expr.RefTo(
            p.Term.Select(offset, Nil, ptrTpe),
            None,
            ptrTpe,
            p.Type.Space.Private,
            p.Region.Rooted(cap)
          )
        ),
        p.Stmt.Var(
          loaded,
          Some(p.Expr.Index(closureSlot, p.Term.IntS64Const(0), ptrTpe)),
          isMutable = false
        ),
        p.Stmt.Var(
          value,
          Some(p.Expr.Index(p.Term.Select(loaded, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, closureDef)), NoopLog).entry

    assert(
      out.body.collectFirst { case p.Stmt.Var(`value`, Some(expr), _) => expr }.exists {
        case p.Expr.Index(p.Term.Select(root, Nil, _), _, p.Type.IntS32) => root.symbol.startsWith("#ab")
        case _                                                           => false
      },
      out.repr
    )
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef, closureDef))), Nil)
  }

  test("taking the address of an arena-relative pointer preserves its local binding slot") {
    val capSym  = sym("Cap")
    val ptrTpe  = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val refTpe  = p.Type.Ptr(ptrTpe, p.Type.Space.Global)
    val capTpe  = p.Type.Struct(capSym, Nil)
    val cap     = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val pointer = named("pointer", ptrTpe)
    val ref     = named("ref", refTpe)
    val capDef  = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          isMutable = true
        ),
        p.Stmt.Var(
          ref,
          Some(
            p.Expr.RefTo(
              p.Term.Select(pointer, Nil, ptrTpe),
              None,
              ptrTpe,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry
    assertEquals(
      out.body.collectFirst { case p.Stmt.Var(`ref`, Some(expr), _) => expr },
      Some(
        p.Expr.RefTo(
          p.Term.Select(pointer, Nil, ptrTpe),
          None,
          ptrTpe,
          p.Type.Space.Global,
          p.Region.Rooted(pointer)
        )
      )
    )
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef))), Nil)
  }

  test("encoded arena pointer arithmetic remains an arena offset") {
    val capSym    = sym("Cap")
    val holderSym = sym("Holder")
    val ptrTpe    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val address   = named("address", p.Type.IntU64)
    val advanced  = named("advanced", p.Type.IntU64)
    val ptr       = named("ptr", ptrTpe)
    val moving    = named("moving", ptrTpe)
    val next      = named("next", ptrTpe)
    val holder    = named("holder", holderTpe)
    val holderPtr = named("holderPtr", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val value     = named("value", p.Type.IntS32)
    val moved     = named("moved", p.Type.IntS32)
    val capDef    = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", ptrTpe), named("direct", ptrTpe)), Nil)
    val data      = p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)
    val slot      = p.Term.Select(holderPtr, List(p.PathStep.Field("data")), ptrTpe).asInstanceOf[p.Term.Select]
    val directSlot =
      p.Term.Select(holder, List(p.PathStep.Field("direct")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(address, Some(p.Expr.Cast(data, p.Type.IntU64)), isMutable = false),
        p.Stmt.Var(
          advanced,
          Some(
            p.Expr.IntrOp(
              p.Intr.Add(
                p.Term.Select(address, Nil, p.Type.IntU64),
                p.Term.IntU64Const(4),
                p.Type.IntU64
              )
            )
          ),
          isMutable = false
        ),
        p.Stmt.Var(
          ptr,
          Some(p.Expr.Cast(p.Term.Select(advanced, Nil, p.Type.IntU64), ptrTpe)),
          isMutable = false
        ),
        p.Stmt.Var(moving, Some(p.Expr.Alias(data)), isMutable = true),
        p.Stmt.Var(
          next,
          Some(
            p.Expr.RefTo(
              p.Term.Select(moving, Nil, ptrTpe),
              Some(p.Term.IntS64Const(1)),
              p.Type.IntS32,
              p.Type.Space.Global,
              p.Region.Rooted(cap)
            )
          ),
          isMutable = false
        ),
        p.Stmt.Mut(p.Term.Select(moving, Nil, ptrTpe), p.Expr.Alias(p.Term.Select(next, Nil, ptrTpe))),
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Var(
          holderPtr,
          Some(
            p.Expr.RefTo(
              p.Term.Select(holder, Nil, holderTpe),
              None,
              holderTpe,
              p.Type.Space.Private,
              p.Region.Rooted(holder)
            )
          ),
          isMutable = false
        ),
        p.Stmt.Mut(slot, p.Expr.Alias(p.Term.Select(ptr, Nil, ptrTpe))),
        p.Stmt.Mut(directSlot, p.Expr.Cast(p.Term.Select(advanced, Nil, p.Type.IntU64), ptrTpe)),
        p.Stmt.Var(
          value,
          Some(p.Expr.Index(slot, p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Var(
          moved,
          Some(p.Expr.Index(p.Term.Select(moving, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, holderDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Mut(`slot`, p.Expr.Alias(p.Term.Select(root, Nil, _))) => root.symbol.startsWith("#ab")
      case _                                                             => false
    })
    assert(out.body.exists {
      case p.Stmt.Mut(`directSlot`, p.Expr.Alias(p.Term.Select(root, Nil, _))) => root.symbol.startsWith("#ab")
      case _                                                                   => false
    })
    assert(out.body.collectFirst { case p.Stmt.Var(`moved`, Some(expr), _) => expr }.exists {
      case p.Expr.Index(p.Term.Select(root, Nil, _), _, p.Type.IntS32) => root.symbol.startsWith("#ab")
      case _                                                           => false
    })
  }

  test("encoded capture base remains a real pointer") {
    val capSym   = sym("Cap")
    val capTpe   = p.Type.Struct(capSym, Nil)
    val capPtr   = p.Type.Ptr(capTpe, p.Type.Space.Global)
    val cap      = named(p.Conventions.CaptureArg, capPtr)
    val address  = named("address", p.Type.IntU64)
    val advanced = named("advanced", p.Type.IntU64)
    val ptr      = named("ptr", capPtr)
    val value    = named("value", p.Type.IntS32)
    val capDef   = p.StructDef(capSym, Nil, List(named("value")), Nil)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(address, Some(p.Expr.Cast(p.Term.Select(cap, Nil, capPtr), p.Type.IntU64)), isMutable = false),
        p.Stmt.Var(
          advanced,
          Some(
            p.Expr.IntrOp(
              p.Intr.Add(
                p.Term.Select(address, Nil, p.Type.IntU64),
                p.Term.IntU64Const(0),
                p.Type.IntU64
              )
            )
          ),
          isMutable = false
        ),
        p.Stmt.Var(ptr, Some(p.Expr.Cast(p.Term.Select(advanced, Nil, p.Type.IntU64), capPtr)), isMutable = false),
        p.Stmt.Var(
          value,
          Some(p.Expr.Alias(p.Term.Select(ptr, List(p.PathStep.Field("value")), p.Type.IntS32))),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assertEquals(
      out.body.collectFirst { case p.Stmt.Var(`value`, Some(expr), _) => expr },
      Some(p.Expr.Alias(p.Term.Select(ptr, List(p.PathStep.Field("value")), p.Type.IntS32)))
    )
  }

  test("null-initialized arena pointer may converge through a self update") {
    val capSym    = sym("Cap")
    val holderSym = sym("Holder")
    val ptrTpe    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val capTpe    = p.Type.Struct(capSym, Nil)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val pointer   = named("pointer", ptrTpe)
    val address   = named("address", p.Type.IntU64)
    val advanced  = named("advanced", p.Type.IntU64)
    val holder    = named("holder", holderTpe)
    val capDef    = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("data", ptrTpe)), Nil)
    val slot      = p.Term.Select(holder, List(p.PathStep.Field("data")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.NullPtrConst(p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))),
          isMutable = true
        ),
        p.Stmt.Mut(
          p.Term.Select(pointer, Nil, ptrTpe),
          p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
        ),
        p.Stmt.Var(address, Some(p.Expr.Cast(p.Term.Select(pointer, Nil, ptrTpe), p.Type.IntU64))),
        p.Stmt.Var(
          advanced,
          Some(
            p.Expr.IntrOp(
              p.Intr.Add(p.Term.Select(address, Nil, p.Type.IntU64), p.Term.IntU64Const(4), p.Type.IntU64)
            )
          )
        ),
        p.Stmt.Mut(
          p.Term.Select(pointer, Nil, ptrTpe),
          p.Expr.Cast(p.Term.Select(advanced, Nil, p.Type.IntU64), ptrTpe)
        ),
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(slot, p.Expr.Alias(p.Term.Select(pointer, Nil, ptrTpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, holderDef)), NoopLog).entry
    assert(out.body.exists {
      case p.Stmt.Mut(`slot`, p.Expr.Alias(p.Term.Select(root, Nil, _))) => root.symbol.startsWith("#ab")
      case _                                                             => false
    })
  }

  test("direct reference to the capture base remains a real pointer in local storage") {
    val capSym    = sym("Cap")
    val holderSym = sym("Holder")
    val capTpe    = p.Type.Struct(capSym, Nil)
    val capPtr    = p.Type.Ptr(capTpe, p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val holder    = named("holder", holderTpe)
    val capDef    = p.StructDef(capSym, Nil, List(named("value")), Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("ptr", capPtr)), Nil)
    val slot      = p.Term.Select(holder, List(p.PathStep.Field("ptr")), capPtr).asInstanceOf[p.Term.Select]
    val reference =
      p.Expr.RefTo(p.Term.Select(cap, Nil, capPtr), None, capTpe, p.Type.Space.Global, p.Region.Rooted(cap))
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(slot, reference),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, holderDef)), NoopLog).entry
    assertEquals(out.body.collectFirst { case p.Stmt.Mut(`slot`, expr) => expr }, Some(reference))
  }

  test("subtracting two arena addresses produces a scalar delta") {
    val capSym  = sym("Cap")
    val ptrTpe  = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val capTpe  = p.Type.Struct(capSym, Nil)
    val cap     = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val address = named("address", p.Type.IntU64)
    val delta   = named("delta", p.Type.IntU64)
    val pointer = named("pointer", ptrTpe)
    val value   = named("value", p.Type.IntS32)
    val capDef  = p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil)
    val data    = p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(address, Some(p.Expr.Cast(data, p.Type.IntU64))),
        p.Stmt.Var(
          delta,
          Some(
            p.Expr.IntrOp(
              p.Intr.Sub(
                p.Term.Select(address, Nil, p.Type.IntU64),
                p.Term.Select(address, Nil, p.Type.IntU64),
                p.Type.IntU64
              )
            )
          )
        ),
        p.Stmt.Var(pointer, Some(p.Expr.Cast(p.Term.Select(delta, Nil, p.Type.IntU64), ptrTpe))),
        p.Stmt
          .Var(value, Some(p.Expr.Index(p.Term.Select(pointer, Nil, ptrTpe), p.Term.IntS64Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry
    assert(out.body.exists {
      case p.Stmt.Var(`value`, Some(p.Expr.Index(p.Term.Select(`pointer`, Nil, _), _, _)), _) => true
      case _                                                                                  => false
    })
  }

  test("arena lowering rejects mixed real and offset pointer representations") {
    val capSym  = sym("Cap")
    val capTpe  = p.Type.Struct(capSym, Nil)
    val capPtr  = p.Type.Ptr(capTpe, p.Type.Space.Global)
    val cap     = named(p.Conventions.CaptureArg, capPtr)
    val pointer = named("pointer", capPtr)
    val capDef  = p.StructDef(capSym, Nil, List(named("child", capPtr)), Nil)
    val child   = p.Term.Select(cap, List(p.PathStep.Field("child")), capPtr)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(pointer, Some(p.Expr.Alias(p.Term.Select(cap, Nil, capPtr))), isMutable = true),
        p.Stmt.Mut(p.Term.Select(pointer, Nil, capPtr), p.Expr.Alias(child)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val ex = intercept[RuntimeException](ArenaLower(program(e, defs = List(capDef)), NoopLog))
    assert(ex.getMessage.contains("arena representation changes"))
  }

  test("capture-derived pointer fields retain offsets when stored through arena pointers") {
    val capSym     = sym("Cap")
    val nodeSym    = sym("Node")
    val ptrTpe     = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val nodeTpe    = p.Type.Struct(nodeSym, Nil)
    val nodePtr    = p.Type.Ptr(nodeTpe, p.Type.Space.Global)
    val capTpe     = p.Type.Struct(capSym, Nil)
    val cap        = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val node       = named("node", nodePtr)
    val other      = named("other", ptrTpe)
    val capDef     = p.StructDef(capSym, Nil, List(named("node", nodePtr), named("other", ptrTpe)), Nil)
    val nodeDef    = p.StructDef(nodeSym, Nil, List(named("next", ptrTpe)), Nil)
    val nodeField  = p.Term.Select(cap, List(p.PathStep.Field("node")), nodePtr)
    val otherField = p.Term.Select(cap, List(p.PathStep.Field("other")), ptrTpe)
    val next       = p.Term.Select(node, List(p.PathStep.Field("next")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(node, Some(p.Expr.Alias(nodeField)), isMutable = false),
        p.Stmt.Var(other, Some(p.Expr.Alias(otherField)), isMutable = false),
        p.Stmt.Mut(next, p.Expr.Alias(p.Term.Select(other, Nil, ptrTpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef, nodeDef)), NoopLog).entry

    assert(out.body.exists {
      case p.Stmt.Mut(p.Term.Select(root, List(p.PathStep.Field("next")), _), p.Expr.Alias(value)) =>
        root.symbol.startsWith("#ab") && value == p.Term.Select(other, Nil, ptrTpe)
      case _ => false
    })
  }

  test("conditional private pointer remains outside the capture arena") {
    val capSym          = sym("Cap")
    val capTpe          = p.Type.Struct(capSym, Nil)
    val cap             = named(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val condition       = named("condition", p.Type.Bool1)
    val valuesTpe       = p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global)
    val pointerTpe      = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    val stalePointerTpe = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val values          = named("values", valuesTpe)
    val fallback        = named("fallback", p.Type.IntS32)
    val fromArray       = named("fromArray", pointerTpe)
    val fromScalar      = named("fromScalar", pointerTpe)
    val selected        = named("selected", pointerTpe)
    val winner          = named("winner", pointerTpe)
    val result          = named("result", p.Type.IntS32)
    val capDef          = p.StructDef(capSym, Nil, List(named("unused")), Nil)
    val e = entry(
      args = List(p.Arg(cap), p.Arg(condition)),
      body = List(
        p.Stmt.Var(values, None, isMutable = true),
        p.Stmt.Var(fallback, Some(p.Expr.Alias(p.Term.IntS32Const(0))), isMutable = true),
        p.Stmt.Var(
          fromArray,
          Some(
            p.Expr.RefTo(
              p.Term.Select(values, Nil, valuesTpe),
              Some(p.Term.IntS64Const(0)),
              p.Type.IntS32,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          isMutable = false
        ),
        p.Stmt.Var(
          fromScalar,
          Some(
            p.Expr.RefTo(
              p.Term.Select(fallback, Nil, p.Type.IntS32),
              None,
              p.Type.IntS32,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          ),
          isMutable = false
        ),
        p.Stmt.Var(selected, None, isMutable = true),
        p.Stmt.Cond(
          p.Term.Select(condition, Nil, p.Type.Bool1),
          List(
            p.Stmt
              .Mut(p.Term.Select(selected, Nil, pointerTpe), p.Expr.Alias(p.Term.Select(fromArray, Nil, pointerTpe)))
          ),
          List(
            p.Stmt.Mut(
              p.Term.Select(selected, Nil, pointerTpe),
              p.Expr.Alias(p.Term.Select(fromScalar, Nil, pointerTpe))
            )
          )
        ),
        p.Stmt.Var(winner, Some(p.Expr.Alias(p.Term.Select(selected, Nil, pointerTpe))), isMutable = false),
        p.Stmt.Var(
          result,
          Some(p.Expr.Index(p.Term.Select(winner, Nil, stalePointerTpe), p.Term.IntS64Const(0), p.Type.IntS32)),
          isMutable = false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assertEquals(
      out.body.collectFirst { case p.Stmt.Var(`result`, Some(expr), _) => expr },
      Some(p.Expr.Index(p.Term.Select(winner, Nil, stalePointerTpe), p.Term.IntS64Const(0), p.Type.IntS32))
    )
    assertEquals(Verify.validateRegions(program(out, defs = List(capDef))), Nil)
  }
}
