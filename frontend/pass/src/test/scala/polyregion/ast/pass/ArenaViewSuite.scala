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
    val program = PassTest.program(buildEntry(), Nil, defs)
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
    val result = ArenaView(PassTest.program(entry, Nil, defs), NoopLog)

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
    val program = PassTest.program(
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

  test("a pointer cast from stack array storage keeps identity through a pointer field") {
    val selfSym    = p.Sym("SelfPointer")
    val selfTpe    = p.Type.Struct(selfSym, Nil)
    val selfPtr    = p.Type.Ptr(selfTpe, p.Type.Space.Global)
    val storage    = named("storage", p.Type.Arr(p.Type.IntU8, 16, p.Type.Space.Global))
    val rawPointer = named("rawPointer", selfPtr)
    val pointer    = named("pointer", selfPtr)
    val same       = named("same", p.Type.Bool1)
    val capArg     = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val selfMember = named("self", selfPtr)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(storage, None, isMutable = true),
          p.Stmt.Var(rawPointer, Some(p.Expr.Cast(selectT(storage), selfPtr)), isMutable = false),
          p.Stmt.Var(pointer, Some(p.Expr.Alias(selectT(rawPointer))), isMutable = false),
          p.Stmt.Mut(
            p.Term.Select(pointer, List(p.PathStep.Field(selfMember.symbol)), selfPtr),
            p.Expr.Cast(selectT(pointer), selfPtr)
          ),
          p.Stmt.Var(
            same,
            Some(
              p.Expr.IntrOp(
                p.Intr.LogicEq(
                  p.Term.Select(pointer, List(p.PathStep.Field(selfMember.symbol)), selfPtr),
                  selectT(pointer)
                )
              )
            ),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, Nil, Nil), p.StructDef(selfSym, Nil, List(selfMember), Nil))
    )

    val result = ArenaView(program, NoopLog)
    assertEquals(result.defs.find(_.name == selfSym).flatMap(_.members.headOption).map(_.tpe), Some(p.Type.IntS64))
    val fieldMut = result.entry.collectAll[p.Stmt].collectFirst {
      case m @ p.Stmt.Mut(p.Term.Select(_, List(p.PathStep.Field("self")), _), _) => m
    }
    assertEquals(fieldMut.map(_.name.tpe), Some(p.Type.IntS64))
    assertEquals(fieldMut.map(_.expr.tpe), Some(p.Type.IntS64))
    val comparison = result.entry.collectAll[p.Expr].collectFirst { case p.Expr.IntrOp(x: p.Intr.LogicEq) => x }
    assertEquals(comparison.map(_.x.tpe), Some(p.Type.IntS64))
    assertEquals(comparison.map(_.y.tpe), Some(p.Type.IntS64))
  }

  test("a nullable base adjustment from immutable local storage drops its null guard") {
    val baseSym                   = p.Sym("Base")
    val derivedSym                = p.Sym("Derived")
    val baseTpe: p.Type.Struct    = p.Type.Struct(baseSym, Nil)
    val derivedTpe: p.Type.Struct = p.Type.Struct(derivedSym, Nil)
    val derivedPtr                = p.Type.Ptr(derivedTpe, p.Type.Space.Global)
    val basePtr                   = p.Type.Ptr(baseTpe, p.Type.Space.Global)
    val local                     = named("local", derivedTpe)
    val source                    = named("source", derivedPtr)
    val adjusted                  = named("adjusted", basePtr)
    val nonNull                   = named("nonNull", p.Type.Bool1)
    val nullSource                = named("nullSource", derivedPtr)
    val isNull                    = named("isNull", p.Type.Bool1)
    val flag                      = named("flag", p.Type.IntS32)
    val capArg                    = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val program = p.Program(
      Some(
        entry(
          args = List(capArg),
          body = List(
            p.Stmt.Var(local, None, isMutable = true),
            p.Stmt.Var(
              source,
              Some(p.Expr.RefTo(selectT(local), None, derivedTpe, p.Type.Space.Private, p.Region.Opaque)),
              isMutable = false
            ),
            p.Stmt.Var(
              adjusted,
              Some(p.Expr.Alias(p.Term.NullPtrConst(baseTpe, p.Type.Space.Global, p.Region.Opaque))),
              isMutable = true
            ),
            p.Stmt.Var(
              nonNull,
              Some(
                p.Expr.IntrOp(
                  p.Intr
                    .LogicNeq(selectT(source), p.Term.NullPtrConst(derivedTpe, p.Type.Space.Global, p.Region.Opaque))
                )
              ),
              isMutable = false
            ),
            p.Stmt.Cond(
              selectT(nonNull),
              List(
                p.Stmt.Mut(
                  selectT(adjusted),
                  p.Expr.RefTo(
                    p.Term.Select(local, List(p.PathStep.Field("base")), baseTpe),
                    None,
                    baseTpe,
                    p.Type.Space.Private,
                    p.Region.Opaque
                  )
                )
              ),
              Nil
            ),
            p.Stmt.Var(
              nullSource,
              Some(p.Expr.Alias(p.Term.NullPtrConst(derivedTpe, p.Type.Space.Global, p.Region.Opaque))),
              false
            ),
            p.Stmt.Var(
              isNull,
              Some(
                p.Expr.IntrOp(
                  p.Intr
                    .LogicEq(selectT(nullSource), p.Term.NullPtrConst(derivedTpe, p.Type.Space.Global, p.Region.Opaque))
                )
              ),
              false
            ),
            p.Stmt.Var(flag, Some(p.Expr.Alias(p.Term.IntS32Const(0))), true),
            p.Stmt.Cond(
              selectT(isNull),
              List(p.Stmt.Mut(selectT(flag), p.Expr.Alias(p.Term.IntS32Const(1)))),
              List(p.Stmt.Mut(selectT(flag), p.Expr.Alias(p.Term.IntS32Const(2))))
            ),
            p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
          )
        )
      ),
      Nil,
      List(
        p.StructDef(capSym, Nil, Nil, Nil),
        p.StructDef(baseSym, Nil, List(named("value", p.Type.IntS32)), Nil),
        p.StructDef(derivedSym, Nil, List(named("padding", p.Type.IntS32), named("base", baseTpe)), List(baseTpe))
      )
    )

    val result = ArenaView(program, NoopLog)
    assertEquals(
      result.entry.collectAll[p.Stmt].collect { case c: p.Stmt.Cond => c },
      Nil,
      result.entry.body.map(_.repr).mkString("\n")
    )
    assert(
      result.entry.collectAll[p.Stmt].collect { case m: p.Stmt.Mut => m }.exists(_.name.root.symbol == adjusted.symbol)
    )
    val flagWrites = result.entry.collectAll[p.Stmt].collect {
      case p.Stmt.Mut(p.Term.Select(root, Nil, _), p.Expr.Alias(p.Term.IntS32Const(value)))
          if root.symbol == flag.symbol =>
        value
    }
    assertEquals(flagWrites, List(1))
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
    val program = PassTest.program(
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
    val program = PassTest.program(
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

  test("arena atomic and volatile accesses use a typed scalar view") {
    val value   = named("value", p.Type.IntU32)
    val capArg  = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val ptrTpe  = p.Type.Ptr(p.Type.IntU32, p.Type.Space.Global)
    val pointer = named("pointer", ptrTpe)
    val loaded  = named("loaded", p.Type.IntU32)
    val stored  = named("stored", p.Type.Unit0)
    val swapped = named("swapped", p.Type.IntU32)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(
            pointer,
            Some(
              p.Expr.RefTo(
                p.Term.Select(capArg.named, List(p.PathStep.Field(value.symbol)), value.tpe),
                None,
                value.tpe,
                p.Type.Space.Global,
                p.Region.Rooted(capArg.named)
              )
            ),
            isMutable = false
          ),
          p.Stmt.Var(
            loaded,
            Some(p.Expr.SpecOp(p.Spec.GpuVolatileLoad(selectT(pointer), value.tpe))),
            isMutable = false
          ),
          p.Stmt.Var(
            stored,
            Some(p.Expr.SpecOp(p.Spec.GpuVolatileStore(selectT(pointer), selectT(loaded)))),
            isMutable = false
          ),
          p.Stmt.Var(
            swapped,
            Some(
              p.Expr.SpecOp(
                p.Spec.GpuAtomicRMW(
                  p.AtomicOp.Xchg,
                  selectT(pointer),
                  p.Term.IntU32Const(7),
                  p.MemScope.Device,
                  p.MemOrder.Relaxed,
                  value.tpe
                )
              )
            ),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, List(value), Nil))
    )

    val result = ArenaView(program, NoopLog)
    val ops = result.entry.collectAll[p.Expr].collect {
      case p.Expr.SpecOp(x: p.Spec.GpuAtomicRMW)     => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileLoad)  => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileStore) => x.ptr
    }
    assertEquals(ops.map(_.tpe), List.fill(3)(ptrTpe))
    val refs = result.entry.collectAll[p.Expr].collect {
      case x: p.Expr.RefTo if x.comp == value.tpe && x.space == p.Type.Space.Global => x
    }
    assertEquals(refs.size, 3)
    assert(refs.forall {
      case p.Expr.RefTo(p.Term.Select(root, Nil, _), Some(_), _, _, _) => root.symbol == "#av2"
      case _                                                           => false
    })
  }

  test("arena aggregate volatile access expands into typed scalar leaves") {
    val pairSym = sym("Pair")
    val pairTpe = p.Type.Struct(pairSym, Nil)
    val pairPtr = p.Type.Ptr(pairTpe, p.Type.Space.Global)
    val pairDef = p.StructDef(pairSym, Nil, List(named("first", p.Type.IntU32), named("second", p.Type.IntU32)), Nil)
    val capArg  = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val pointer = named("pointer", pairPtr)
    val loaded  = named("loaded", pairTpe)
    val stored  = named("stored", p.Type.Unit0)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(
            pointer,
            Some(p.Expr.Alias(p.Term.Select(capArg.named, List(p.PathStep.Field("pair")), pairPtr))),
            isMutable = false
          ),
          p.Stmt.Var(
            loaded,
            Some(p.Expr.SpecOp(p.Spec.GpuVolatileLoad(selectT(pointer), pairTpe))),
            isMutable = false
          ),
          p.Stmt.Var(
            stored,
            Some(p.Expr.SpecOp(p.Spec.GpuVolatileStore(selectT(pointer), selectT(loaded)))),
            isMutable = false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, List(named("pair", pairPtr)), Nil), pairDef)
    )

    val result = ArenaView(program, NoopLog)
    val loads = result.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.GpuVolatileLoad) =>
      x
    }
    val stores = result.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.GpuVolatileStore) =>
      x
    }
    assertEquals(loads.map(_.rtn), List.fill(2)(p.Type.IntU32))
    assertEquals(stores.map(_.value.tpe), List.fill(2)(p.Type.IntU32))
    assert((loads.map(_.ptr) ::: stores.map(_.ptr)).forall {
      case p.Term.Select(root, Nil, p.Type.Ptr(p.Type.IntU32, p.Type.Space.Global)) => root.symbol.startsWith("#vr")
      case _                                                                        => false
    })
  }
}
