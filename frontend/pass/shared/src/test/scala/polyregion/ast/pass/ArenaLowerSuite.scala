package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class ArenaLowerSuite extends munit.FunSuite {

  test("flat arena lowering exposes a byte arena at the function boundary") {
    val capSym    = sym("Cap")
    val capStruct = p.Type.Struct(capSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, p.Type.Ptr(capStruct, p.Type.Space.Global))
    val arena     = named("#arena_base", BytePtr)
    val capDef    = p.StructDef(capSym, Nil, List(named("x")), Nil)
    val readX =
      p.Stmt.Var(named("out"), Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("x")), p.Type.IntS32))))
    val e = entry(args = List(p.Arg(cap)), body = List(readX, p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))

    val out = ArenaLower(program(e, defs = List(capDef)), NoopLog).entry

    assertEquals(out.args.map(_.named), List(arena))
    assertEquals(
      out.body.headOption,
      Some(p.Stmt.Var(cap, Some(p.Expr.Cast(p.Term.Select(arena, Nil, BytePtr), cap.tpe)), isMutable = false))
    )
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
}
