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

  test("removing the capture argument rebases boundary extents") {
    val base = buildEntry()
    val output = arg("output", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)).copy(
      boundary = Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
    )
    val count  = arg("count", p.Type.IntS32)
    val entry  = base.copy(decl = base.decl.copy(args = base.args ::: List(output, count)))
    val result = ArenaView(p.Program(entry, Nil, defs), NoopLog)

    assertEquals(
      result.entry.args.find(_.named.symbol == output.named.symbol).flatMap(_.boundary).map(_.extent),
      Some(p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1)))
    )
  }

  test("a private pointer field in a stack-local closure stays a pointer") {
    val closureSym    = p.Sym("Closure")
    val privatePtr    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private)
    val privatePtrPtr = p.Type.Ptr(privatePtr, p.Type.Space.Private)
    val closureTpe    = p.Type.Struct(closureSym, Nil)
    val closure       = named("closure", closureTpe)
    val value         = named("value", p.Type.IntS32)
    val pointer       = named("pointer", privatePtr)
    val loaded        = named("loaded", privatePtr)
    val capArg        = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val program = p.Program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(value, Some(p.Expr.Alias(p.Term.IntS32Const(42))), isMutable = true),
          p.Stmt.Var(
            pointer,
            Some(p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Private, p.Region.Opaque)),
            isMutable = true
          ),
          p.Stmt.Var(closure, None, isMutable = true),
          p.Stmt.Mut(
            p.Term.Select(closure, List(p.PathStep.Field("ref")), privatePtrPtr),
            p.Expr.RefTo(selectT(pointer), None, privatePtr, p.Type.Space.Private, p.Region.Opaque)
          ),
          p.Stmt.Var(
            loaded,
            Some(
              p.Expr.Index(
                p.Term.Select(closure, List(p.PathStep.Field("ref")), privatePtrPtr),
                p.Term.IntS64Const(0),
                privatePtr
              )
            ),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(
        p.StructDef(capSym, Nil, Nil, Nil),
        p.StructDef(closureSym, Nil, List(named("ref", privatePtrPtr)), Nil)
      )
    )

    val result = ArenaView(program, NoopLog)
    assertEquals(result.defs.find(_.name == closureSym).flatMap(_.members.headOption).map(_.tpe), Some(privatePtrPtr))
    val fieldMut = result.entry.collectAll[p.Stmt].collectFirst {
      case m @ p.Stmt.Mut(p.Term.Select(_, List(p.PathStep.Field("ref")), _), _) => m
    }
    assertEquals(fieldMut.map(_.name.tpe), Some(privatePtrPtr))
    assertEquals(fieldMut.map(_.expr.tpe), Some(privatePtrPtr))
    val loadedVar =
      result.entry.collectAll[p.Stmt].collectFirst { case v: p.Stmt.Var if v.name.symbol == loaded.symbol => v }
    assertEquals(loadedVar.map(_.name.tpe), Some(privatePtr))
    assertEquals(loadedVar.flatMap(_.expr).map(_.tpe), Some(privatePtr))
  }

  test("a private pointer to a local arena-offset slot keeps only its outer pointer") {
    val closureSym   = p.Sym("MixedClosure")
    val closureTpe   = p.Type.Struct(closureSym, Nil)
    val globalPtr    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val mixedPtr     = p.Type.Ptr(globalPtr, p.Type.Space.Private)
    val loweredMixed = p.Type.Ptr(p.Type.IntS64, p.Type.Space.Private)
    val capArg       = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val pointer      = named("pointer", globalPtr)
    val closure      = named("closure", closureTpe)
    val loaded       = named("loaded", globalPtr)
    val program = p.Program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(
            pointer,
            Some(
              p.Expr.RefTo(
                p.Term.Select(capArg.named, List(p.PathStep.Field("value")), p.Type.IntS32),
                None,
                p.Type.IntS32,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            isMutable = true
          ),
          p.Stmt.Var(closure, None, isMutable = true),
          p.Stmt.Mut(
            p.Term.Select(closure, List(p.PathStep.Field("ref")), mixedPtr),
            p.Expr.RefTo(selectT(pointer), None, globalPtr, p.Type.Space.Private, p.Region.Opaque)
          ),
          p.Stmt.Var(
            loaded,
            Some(
              p.Expr.Index(
                p.Term.Select(closure, List(p.PathStep.Field("ref")), mixedPtr),
                p.Term.IntS64Const(0),
                globalPtr
              )
            ),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(
        p.StructDef(capSym, Nil, List(named("value", p.Type.IntS32)), Nil),
        p.StructDef(closureSym, Nil, List(named("ref", mixedPtr)), Nil)
      )
    )

    val result = ArenaView(program, NoopLog)
    assertEquals(result.defs.find(_.name == closureSym).flatMap(_.members.headOption).map(_.tpe), Some(loweredMixed))
    val fieldMut = result.entry.collectAll[p.Stmt].collectFirst {
      case m @ p.Stmt.Mut(p.Term.Select(_, List(p.PathStep.Field("ref")), _), _) => m
    }
    assertEquals(fieldMut.map(_.name.tpe), Some(loweredMixed))
    assertEquals(fieldMut.map(_.expr.tpe), Some(loweredMixed))
    val loadedVar =
      result.entry.collectAll[p.Stmt].collectFirst { case v: p.Stmt.Var if v.name.symbol == loaded.symbol => v }
    assertEquals(loadedVar.map(_.name.tpe), Some(p.Type.IntS64))
    assertEquals(loadedVar.flatMap(_.expr).map(_.tpe), Some(p.Type.IntS64))
  }

  test("an arena pointer compares with null as an offset") {
    val capArg = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val isNull = named("isNull", p.Type.Bool1)
    val program = p.Program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(
            isNull,
            Some(
              p.Expr.IntrOp(
                p.Intr.LogicEq(
                  selectT(capArg.named),
                  p.Term.NullPtrConst(capTpe, p.Type.Space.Global, p.Region.Opaque)
                )
              )
            ),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, Nil, Nil))
    )

    val result     = ArenaView(program, NoopLog)
    val comparison = result.entry.collectAll[p.Expr].collectFirst { case p.Expr.IntrOp(x: p.Intr.LogicEq) => x }
    assertEquals(comparison.map(_.x.tpe), Some(p.Type.IntS64))
    assertEquals(comparison.map(_.y.tpe), Some(p.Type.IntS64))
  }
}
