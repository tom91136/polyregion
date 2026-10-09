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
    p.StructDef(capSym, Nil, List(named("node", p.Type.Ptr(nodeTpe, p.Type.Space.Global))), Nil)
  )

  // a stack-local iterator (not reachable from the capture arg) whose node pointer is chased into arena
  // memory: `p = &itVal; p->ptr->val = 42` - a mutation crossing from a real local pointer into the arena
  private def buildEntry(): p.Function = {
    val capArg = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val itVal  = named("itVal", iterTpe)
    val pp     = named("p", p.Type.Ptr(iterTpe, p.Type.Space.Global))
    val ptrTpe = p.Type.Ptr(nodeTpe, p.Type.Space.Global)
    entry(
      args = List(capArg),
      body = List(
        p.Stmt.Var(itVal, None, isMutable = true),
        p.Stmt.Mut(
          p.Term.Select(itVal, List(p.PathStep.Field("ptr")), ptrTpe).asInstanceOf[p.Term.Select],
          p.Expr.Alias(p.Term.Select(capArg.named, List(p.PathStep.Field("node")), ptrTpe))
        ),
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
      Some(p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(8)))
    )
  }

  test("the typed views take the capture argument's position") {
    val base   = buildEntry()
    val before = arg("before", p.Type.IntS32)
    val after  = arg("after", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val entry  = base.copy(decl = base.decl.copy(args = before :: base.args ::: List(after)))
    val result = ArenaView(PassTest.program(entry, Nil, defs), NoopLog)
    assertEquals(
      result.entry.args.map(_.named.symbol),
      "before" :: LogicalArenaViewAbi.bindings.map(_.symbol) ::: List("after")
    )
  }

  test("a kernel that never reads its capture keeps the capture argument in place") {
    val capArg = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val output = arg("output", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val entry = PassTest.entry(
      args = List(capArg, output),
      body = List(
        p.Stmt.Update(selectT(output.named), p.Term.IntS64Const(0), p.Term.IntS32Const(1)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val result = ArenaView(PassTest.program(entry, Nil, defs), NoopLog)
    assertEquals(result.entry.args.map(_.named.symbol), List(p.Conventions.CaptureArg, "output"))
  }

  test("a receiver capture's typed views lead the arguments") {
    val base     = buildEntry()
    val capture  = base.args.head
    val receiver = capture.copy(named = capture.named.copy(symbol = p.Conventions.ThisReceiver))
    val after    = arg("after", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val entry = base
      .copy(decl = base.decl.copy(receiver = Some(receiver), args = List(after)))
      .modifyAll[p.Term] {
        case p.Term.Select(root, steps, tpe) if root == capture.named => p.Term.Select(receiver.named, steps, tpe)
        case term                                                     => term
      }
    val result = ArenaView(PassTest.program(entry, Nil, defs), NoopLog)
    assertEquals(result.entry.get.receiver, None)
    assertEquals(result.entry.args.map(_.named.symbol), LogicalArenaViewAbi.bindings.map(_.symbol) ::: List("after"))
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

  test("a direct reference to stack array storage uses the same identity token as a pointer local") {
    val holderSym     = p.Sym("DirectPointerHolder")
    val holderTpe     = p.Type.Struct(holderSym, Nil)
    val pointerTpe    = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val pointerMember = named("pointer", pointerTpe)
    val storage       = named("storage", p.Type.Arr(p.Type.IntS32, 2, p.Type.Space.Global))
    val pointer       = named("pointer", pointerTpe)
    val loaded        = named("loaded", pointerTpe)
    val holder        = named("holder", holderTpe)
    val copied        = named("copied", holderTpe)
    val same          = named("same", p.Type.Bool1)
    val capArg        = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val direct = p.Expr.RefTo(
      selectT(storage),
      Some(p.Term.IntS64Const(1)),
      p.Type.IntS32,
      p.Type.Space.Global,
      p.Region.Opaque
    )
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(storage, None, isMutable = true),
          p.Stmt.Var(pointer, Some(direct), isMutable = false),
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Mut(
            p.Term.Select(holder, List(p.PathStep.Field(pointerMember.symbol)), pointerTpe),
            direct
          ),
          p.Stmt.Var(
            loaded,
            Some(p.Expr.Alias(p.Term.Select(holder, List(p.PathStep.Field(pointerMember.symbol)), pointerTpe))),
            isMutable = false
          ),
          p.Stmt.Var(copied, None, isMutable = true),
          p.Stmt.Mut(
            p.Term.Select(copied, List(p.PathStep.Field(pointerMember.symbol)), pointerTpe),
            p.Expr.Alias(selectT(loaded))
          ),
          p.Stmt.Var(
            same,
            Some(
              p.Expr.IntrOp(
                p.Intr.LogicEq(
                  selectT(loaded),
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
      List(p.StructDef(capSym, Nil, Nil, Nil), p.StructDef(holderSym, Nil, List(pointerMember), Nil))
    )

    val result = ArenaView(program, NoopLog)
    assertEquals(result.defs.find(_.name == holderSym).flatMap(_.members.headOption).map(_.tpe), Some(p.Type.IntS64))
    val fieldMut = result.entry.collectAll[p.Stmt].collectFirst {
      case m @ p.Stmt.Mut(p.Term.Select(_, List(p.PathStep.Field("pointer")), _), _) => m
    }
    assertEquals(fieldMut.map(_.name.tpe), Some(p.Type.IntS64))
    assertEquals(fieldMut.map(_.expr.tpe), Some(p.Type.IntS64))
    val loadedVar = result.entry.collectAll[p.Stmt].collectFirst {
      case v: p.Stmt.Var if v.name.symbol == loaded.symbol => v
    }
    assertEquals(loadedVar.map(_.name.tpe), Some(p.Type.IntS64))
    assertEquals(loadedVar.flatMap(_.expr).map(_.tpe), Some(p.Type.IntS64))
    val copiedFieldMut = result.entry.collectAll[p.Stmt].collectFirst {
      case m @ p.Stmt.Mut(p.Term.Select(root, List(p.PathStep.Field("pointer")), _), _)
          if root.symbol == copied.symbol =>
        m
    }
    assertEquals(copiedFieldMut.map(_.name.tpe), Some(p.Type.IntS64))
    assertEquals(copiedFieldMut.map(_.expr.tpe), Some(p.Type.IntS64))
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

  test("a pointer chosen between arena and private storage and only loaded loads in each branch") {
    val capArg   = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val i64Ptr   = p.Type.Ptr(p.Type.IntS64, p.Type.Space.Global)
    val local    = named("local", p.Type.IntS64)
    val localRef = named("localRef", i64Ptr)
    val arenaRef = named("arenaRef", i64Ptr)
    val picked   = named("picked", i64Ptr)
    val alias    = named("alias", i64Ptr)
    val value    = named("value", p.Type.IntS64)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS64Const(5))), isMutable = true),
          p.Stmt.Var(
            localRef,
            Some(p.Expr.RefTo(selectT(local), None, p.Type.IntS64, p.Type.Space.Global, p.Region.Opaque)),
            false
          ),
          p.Stmt.Var(
            arenaRef,
            Some(
              p.Expr.RefTo(
                p.Term.Select(capArg.named, List(p.PathStep.Field("limit")), p.Type.IntS64),
                None,
                p.Type.IntS64,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Var(picked, None, isMutable = true),
          p.Stmt.Cond(
            p.Term.Bool1Const(true),
            List(p.Stmt.Mut(selectT(picked), p.Expr.Alias(selectT(arenaRef)))),
            List(p.Stmt.Mut(selectT(picked), p.Expr.Alias(selectT(localRef))))
          ),
          p.Stmt.Var(alias, Some(p.Expr.Alias(selectT(picked))), false),
          p.Stmt.Var(value, Some(p.Expr.Index(selectT(alias), p.Term.IntS64Const(0), p.Type.IntS64)), false),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, List(named("limit", p.Type.IntS64)), Nil))
    )
    val stmts = ArenaView(program, NoopLog).entry.toList.flatMap(_.collectAll[p.Stmt])
    assert(!stmts.exists {
      case p.Stmt.Var(n, _, _) => n.symbol == "picked"
      case _                   => false
    })
    assert(stmts.exists {
      case p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(_, Nil, p.Type.IntS64))), _) => n.symbol == "value"
      case _                                                                          => false
    })
  }

  test("a pointer chosen between arena and private storage and loaded into a local loads in each branch") {
    val capArg   = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val i64Ptr   = p.Type.Ptr(p.Type.IntS64, p.Type.Space.Global)
    val local    = named("local", p.Type.IntS64)
    val localRef = named("localRef", i64Ptr)
    val arenaRef = named("arenaRef", i64Ptr)
    val picked   = named("picked", i64Ptr)
    val result   = named("result", p.Type.IntS64)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(result, None, isMutable = true),
          p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS64Const(5))), isMutable = true),
          p.Stmt.Var(
            localRef,
            Some(p.Expr.RefTo(selectT(local), None, p.Type.IntS64, p.Type.Space.Global, p.Region.Opaque)),
            false
          ),
          p.Stmt.Var(
            arenaRef,
            Some(
              p.Expr.RefTo(
                p.Term.Select(capArg.named, List(p.PathStep.Field("limit")), p.Type.IntS64),
                None,
                p.Type.IntS64,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Var(picked, None, isMutable = true),
          p.Stmt.Cond(
            p.Term.Bool1Const(true),
            List(p.Stmt.Mut(selectT(picked), p.Expr.Alias(selectT(arenaRef)))),
            List(p.Stmt.Mut(selectT(picked), p.Expr.Alias(selectT(localRef))))
          ),
          p.Stmt.Mut(selectT(result), p.Expr.Index(selectT(picked), p.Term.IntS64Const(0), p.Type.IntS64)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, List(named("limit", p.Type.IntS64)), Nil))
    )
    val stmts = ArenaView(program, NoopLog).entry.toList.flatMap(_.collectAll[p.Stmt])
    assert(!stmts.exists {
      case p.Stmt.Var(n, _, _) => n.symbol == "picked"
      case _                   => false
    })
  }

  test("a pointer chosen by address expressions and loaded through a copy loads in each branch") {
    val capArg = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val i64Ptr = p.Type.Ptr(p.Type.IntS64, p.Type.Space.Global)
    val local  = named("local", p.Type.IntS64)
    val picked = named("picked", i64Ptr)
    val copy   = named("copy", i64Ptr)
    val value  = named("value", p.Type.IntS64)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(local, Some(p.Expr.Alias(p.Term.IntS64Const(5))), isMutable = true),
          p.Stmt.Var(picked, None, isMutable = true),
          p.Stmt.Cond(
            p.Term.Bool1Const(true),
            List(
              p.Stmt.Mut(
                selectT(picked),
                p.Expr.RefTo(
                  p.Term.Select(capArg.named, List(p.PathStep.Field("limit")), p.Type.IntS64),
                  None,
                  p.Type.IntS64,
                  p.Type.Space.Global,
                  p.Region.Opaque
                )
              )
            ),
            List(
              p.Stmt.Mut(
                selectT(picked),
                p.Expr.RefTo(selectT(local), None, p.Type.IntS64, p.Type.Space.Global, p.Region.Opaque)
              )
            )
          ),
          p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(picked))), isMutable = true),
          p.Stmt.Var(value, Some(p.Expr.Index(selectT(copy), p.Term.IntS64Const(0), p.Type.IntS64)), false),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      Nil,
      List(p.StructDef(capSym, Nil, List(named("limit", p.Type.IntS64)), Nil))
    )
    val stmts = ArenaView(program, NoopLog).entry.toList.flatMap(_.collectAll[p.Stmt])
    assert(!stmts.exists {
      case p.Stmt.Var(n, _, _) => n.symbol == "picked" || n.symbol == "copy"
      case _                   => false
    })
  }

  test("an arena pointer local reassigned to null stores a null offset") {
    val capArg  = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val current = named("current", capArg.named.tpe)
    val program = PassTest.program(
      entry(
        args = List(capArg),
        body = List(
          p.Stmt.Var(current, Some(p.Expr.Alias(selectT(capArg.named))), isMutable = true),
          p.Stmt.Mut(selectT(current), p.Expr.Alias(p.Term.NullPtrConst(capTpe, p.Type.Space.Global, p.Region.Opaque))),
          p.Stmt.Var(
            named("isNull", p.Type.Bool1),
            Some(
              p.Expr.IntrOp(
                p.Intr.LogicEq(selectT(current), p.Term.NullPtrConst(capTpe, p.Type.Space.Global, p.Region.Opaque))
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
    val stores = ArenaView(program, NoopLog).entry.toList.flatMap(_.collectAll[p.Stmt]).collect {
      case p.Stmt.Mut(p.Term.Select(_, Nil, p.Type.IntS64), expr) => expr
    }
    assertEquals(stores, List(p.Expr.Alias(p.Term.IntS64Const(0))))
  }

  test("arena atomic and volatile accesses use a typed scalar view") {
    val value    = named("value", p.Type.IntU32)
    val capArg   = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))
    val ptrTpe   = p.Type.Ptr(p.Type.IntU32, p.Type.Space.Global)
    val pointer  = named("pointer", ptrTpe)
    val loaded   = named("loaded", p.Type.IntU32)
    val stored   = named("stored", p.Type.Unit0)
    val swapped  = named("swapped", p.Type.IntU32)
    val compared = named("compared", p.Type.IntU32)
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
          p.Stmt.Var(
            compared,
            Some(
              p.Expr.SpecOp(
                p.Spec.GpuAtomicCAS(
                  selectT(pointer),
                  p.Term.IntU32Const(3),
                  p.Term.IntU32Const(4),
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
      case p.Expr.SpecOp(x: p.Spec.GpuAtomicCAS)     => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileLoad)  => x.ptr
      case p.Expr.SpecOp(x: p.Spec.GpuVolatileStore) => x.ptr
    }
    assertEquals(ops.map(_.tpe), List.fill(4)(ptrTpe))
    val refs = result.entry.collectAll[p.Expr].collect {
      case x: p.Expr.RefTo if x.comp == value.tpe && x.space == p.Type.Space.Global => x
    }
    assertEquals(refs.size, 4)
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

  private val holderSym  = sym("NativeHolder")
  private val holderTpe  = p.Type.Struct(holderSym, Nil)
  private val pointerTpe = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)

  private def holderField(holder: p.Named): p.Term.Select =
    p.Term.Select(holder, List(p.PathStep.Field("pointer")), pointerTpe).asInstanceOf[p.Term.Select]

  private def holderProgram(args: List[p.Arg], body: List[p.Stmt]): p.Program = PassTest.program(
    entry(args = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global)) :: args, body = body),
    Nil,
    List(p.StructDef(capSym, Nil, Nil, Nil), p.StructDef(holderSym, Nil, List(named("pointer", pointerTpe)), Nil))
  )

  private def loadAndIndex(source: p.Term, loaded: p.Named): List[p.Stmt] = List(
    p.Stmt.Var(loaded, Some(p.Expr.Alias(source)), isMutable = false),
    p.Stmt.Var(
      named("value", p.Type.IntS32),
      Some(p.Expr.Index(selectT(loaded), p.Term.IntS64Const(0), p.Type.IntS32)),
      isMutable = false
    ),
    p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
  )

  private def indexedPointer(result: p.Program): Option[p.Term] =
    result.entry.collectAll[p.Stmt].collectFirst {
      case p.Stmt.Var(n, Some(p.Expr.Index(pointer, _, _)), _) if n.symbol == "value" => pointer
    }

  test("a native global pointer written once into a local aggregate forwards to its source") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val loaded   = named("loaded", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named)))
        ) ::: loadAndIndex(holderField(holder), loaded)
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
    assert(!result.entry.collectAll[p.Stmt].exists {
      case p.Stmt.Mut(p.Term.Select(root, _, _), _) => root.symbol == holder.symbol
      case _                                        => false
    })
  }

  test("a capture copy patched from its explicit pointer argument forwards to that argument") {
    val viewSym   = sym("View")
    val viewTpe   = p.Type.Struct(viewSym, Nil)
    val outerSym  = sym("Outer")
    val outerTpe  = p.Type.Struct(outerSym, Nil)
    val capArg    = arg(p.Conventions.CaptureArg, p.Type.Ptr(outerTpe, p.Type.Space.Global))
    val explicit  = arg("#capture_ptr_0", pointerTpe)
    val copy      = named("#capture_copy_0", viewTpe)
    val loaded    = named("loaded", pointerTpe)
    val copyField = p.Term.Select(copy, List(p.PathStep.Field("m_p")), pointerTpe).asInstanceOf[p.Term.Select]
    val program = PassTest.program(
      entry(
        args = List(capArg, explicit),
        body = List(
          p.Stmt
            .Var(copy, Some(p.Expr.Alias(p.Term.Select(capArg.named, List(p.PathStep.Field("view")), viewTpe))), true),
          p.Stmt.Mut(copyField, p.Expr.Alias(selectT(explicit.named)))
        ) ::: loadAndIndex(copyField, loaded)
      ),
      Nil,
      List(
        p.StructDef(outerSym, Nil, List(named("view", viewTpe)), Nil),
        p.StructDef(viewSym, Nil, List(named("m_p", pointerTpe), named("m_count", p.Type.IntS64)), Nil)
      )
    )
    val result = ArenaView(program, NoopLog)
    assertEquals(indexedPointer(result), Some(selectT(explicit.named)))
  }

  test("a copy of a by-value aggregate argument patched right after its definition forwards to the patch") {
    val viewSym   = sym("View")
    val viewTpe   = p.Type.Struct(viewSym, Nil)
    val byValue   = arg("functor", viewTpe)
    val explicit  = arg("#capture_ptr_0", pointerTpe)
    val copy      = named("#capture_copy_0", viewTpe)
    val loaded    = named("loaded", pointerTpe)
    val copyField = p.Term.Select(copy, List(p.PathStep.Field("m_p")), pointerTpe).asInstanceOf[p.Term.Select]
    val program = PassTest.program(
      entry(
        args = List(byValue, explicit),
        body = List(
          p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(byValue.named))), true),
          p.Stmt.Mut(copyField, p.Expr.Alias(selectT(explicit.named)))
        ) ::: loadAndIndex(copyField, loaded)
      ),
      Nil,
      List(p.StructDef(viewSym, Nil, List(named("m_p", pointerTpe), named("m_count", p.Type.IntS64)), Nil))
    )
    val result = ArenaView(program, NoopLog)
    assertEquals(indexedPointer(result), Some(selectT(explicit.named)))
  }

  test("a copy of a by-value aggregate argument read before its patch stays rejected") {
    val viewSym   = sym("View")
    val viewTpe   = p.Type.Struct(viewSym, Nil)
    val byValue   = arg("functor", viewTpe)
    val explicit  = arg("#capture_ptr_0", pointerTpe)
    val copy      = named("#capture_copy_0", viewTpe)
    val copyField = p.Term.Select(copy, List(p.PathStep.Field("m_p")), pointerTpe).asInstanceOf[p.Term.Select]
    val program = PassTest.program(
      entry(
        args = List(byValue, explicit),
        body = List(
          p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(byValue.named))), true),
          p.Stmt.Var(
            named("early", p.Type.IntS32),
            Some(p.Expr.Index(copyField, p.Term.IntS64Const(0), p.Type.IntS32)),
            false
          ),
          p.Stmt.Mut(copyField, p.Expr.Alias(selectT(explicit.named)))
        ) ::: loadAndIndex(copyField, named("loaded", pointerTpe))
      ),
      Nil,
      List(p.StructDef(viewSym, Nil, List(named("m_p", pointerTpe), named("m_count", p.Type.IntS64)), Nil))
    )
    intercept[IllegalArgumentException](ArenaView(program, NoopLog))
  }

  private val pairSym = sym("Pair")
  private val pairTpe = p.Type.Struct(pairSym, Nil)

  private def pairProgram(args: List[p.Arg], body: List[p.Stmt]): p.Program = PassTest.program(
    entry(args = arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global)) :: args, body = body),
    Nil,
    List(
      p.StructDef(capSym, Nil, Nil, Nil),
      p.StructDef(holderSym, Nil, List(named("pointer", pointerTpe)), Nil),
      p.StructDef(pairSym, Nil, List(named("first", pointerTpe), named("second", holderTpe)), Nil)
    )
  )

  private def indexedPointers(result: p.Program): List[p.Term] =
    result.entry.collectAll[p.Expr].collect { case p.Expr.Index(pointer, _, _) => pointer }

  test("every patched leaf of one local aggregate forwards to its own argument") {
    val left  = arg("left", pointerTpe)
    val right = arg("right", pointerTpe)
    val pair  = named("pair", pairTpe)
    val first = p.Term.Select(pair, List(p.PathStep.Field("first")), pointerTpe).asInstanceOf[p.Term.Select]
    val second = p.Term
      .Select(pair, List(p.PathStep.Field("second"), p.PathStep.Field("pointer")), pointerTpe)
      .asInstanceOf[p.Term.Select]
    val result = ArenaView(
      pairProgram(
        List(left, right),
        List(
          p.Stmt.Var(pair, None, isMutable = true),
          p.Stmt.Mut(first, p.Expr.Alias(selectT(left.named))),
          p.Stmt.Mut(second, p.Expr.Alias(selectT(right.named))),
          p.Stmt.Var(named("a", p.Type.IntS32), Some(p.Expr.Index(first, p.Term.IntS64Const(0), p.Type.IntS32)), false),
          p.Stmt
            .Var(named("b", p.Type.IntS32), Some(p.Expr.Index(second, p.Term.IntS64Const(0), p.Type.IntS32)), false),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      NoopLog
    )
    assertEquals(indexedPointers(result).toSet, Set[p.Term](selectT(left.named), selectT(right.named)))
  }

  test("a struct copy of a forwarded local forwards its pointer leaf") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val copy     = named("copy", holderTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
          p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(holder))), isMutable = false)
        ) ::: loadAndIndex(holderField(copy), named("loaded", pointerTpe))
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
  }

  test("a forwarded leaf read into an address-taken pointer local still forwards") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val loaded   = named("loaded", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
          p.Stmt.Var(loaded, Some(p.Expr.Alias(holderField(holder))), isMutable = true),
          p.Stmt.Var(
            named("address", p.Type.Ptr(pointerTpe, p.Type.Space.Private)),
            Some(p.Expr.RefTo(selectT(loaded), None, pointerTpe, p.Type.Space.Private, p.Region.Opaque)),
            isMutable = false
          ),
          p.Stmt.Var(
            named("value", p.Type.IntS32),
            Some(p.Expr.Index(selectT(loaded), p.Term.IntS64Const(0), p.Type.IntS32)),
            false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      NoopLog
    )
    assert(!result.entry.collectAll[p.Stmt].exists {
      case p.Stmt.Mut(p.Term.Select(root, _, _), _) => root.symbol == holder.symbol
      case _                                        => false
    })
  }

  test("a local aggregate constructed through its own address forwards its pointer leaf") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val self     = named("self", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val alias    = named("alias", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Var(
            self,
            Some(p.Expr.RefTo(selectT(holder), None, holderTpe, p.Type.Space.Private, p.Region.Opaque)),
            false
          ),
          p.Stmt.Var(alias, Some(p.Expr.Alias(selectT(self))), isMutable = true),
          p.Stmt.Mut(
            p.Term.Select(alias, List(p.PathStep.Field("pointer")), pointerTpe).asInstanceOf[p.Term.Select],
            p.Expr.Alias(selectT(external.named))
          )
        ) ::: loadAndIndex(holderField(holder), named("loaded", pointerTpe))
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
  }

  private def loadedDefinition(result: p.Program): Option[p.Expr] =
    result.entry.flatMap(_.collectAll[p.Stmt].collectFirst {
      case p.Stmt.Var(n, Some(expr), _) if n.symbol == "loaded" => expr
    })

  private def offsetFrom(external: p.Named)(definition: Option[p.Expr]): Boolean = definition.exists {
    case p.Expr.RefTo(p.Term.Select(root, Nil, _), Some(_), p.Type.IntS32, p.Type.Space.Global, _) => root == external
    case _                                                                                         => false
  }

  test("a pointer derived from an argument stored in a local aggregate becomes an offset from it") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val derived  = named("derived", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Var(
            derived,
            Some(
              p.Expr.RefTo(
                selectT(external.named),
                Some(p.Term.IntS32Const(1)),
                p.Type.IntS32,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(derived)))
        ) ::: loadAndIndex(holderField(holder), named("loaded", pointerTpe))
      ),
      NoopLog
    )
    assert(offsetFrom(external.named)(loadedDefinition(result)), loadedDefinition(result))
  }

  test("a derived pointer slot keeps its offset through whole-aggregate copies") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val copy     = named("copy", holderTpe)
    val derived  = named("derived", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Var(
            derived,
            Some(
              p.Expr.RefTo(
                selectT(external.named),
                Some(p.Term.IntS64Const(2)),
                p.Type.IntS32,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(derived))),
          p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(holder))), isMutable = true)
        ) ::: loadAndIndex(holderField(copy), named("loaded", pointerTpe))
      ),
      NoopLog
    )
    assert(offsetFrom(external.named)(loadedDefinition(result)), loadedDefinition(result))
  }

  test("a store through a derived pointer slot addresses the argument") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val derived  = named("derived", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Var(
            derived,
            Some(
              p.Expr.RefTo(
                selectT(external.named),
                Some(p.Term.IntS32Const(1)),
                p.Type.IntS32,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(derived))),
          p.Stmt.Update(holderField(holder), p.Term.IntS64Const(0), p.Term.IntS32Const(5)),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      NoopLog
    )
    val stmts  = result.entry.toList.flatMap(_.collectAll[p.Stmt])
    val target = stmts.collectFirst { case p.Stmt.Update(p.Term.Select(root, Nil, _), _, _) => root }
    assert(
      target.exists(t =>
        offsetFrom(external.named)(stmts.collectFirst { case p.Stmt.Var(`t`, init, _) => init }.flatten)
      ),
      stmts
    )
  }

  test("a reassigned pointer local advanced from an argument becomes an offset from it") {
    val external = arg("external", pointerTpe)
    val cursor   = named("cursor", pointerTpe)
    val next     = named("next", pointerTpe)
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(cursor, Some(p.Expr.Alias(selectT(external.named))), isMutable = true),
          p.Stmt.Var(
            next,
            Some(
              p.Expr.RefTo(
                selectT(cursor),
                Some(p.Term.IntS32Const(1)),
                p.Type.IntS32,
                p.Type.Space.Global,
                p.Region.Opaque
              )
            ),
            false
          ),
          p.Stmt.Mut(selectT(cursor), p.Expr.Alias(selectT(next)))
        ) ::: loadAndIndex(selectT(cursor), named("loaded", pointerTpe))
      ),
      NoopLog
    )
    assert(offsetFrom(external.named)(loadedDefinition(result)), loadedDefinition(result))
  }

  test("a reference slot holding a local's address forwards field reads through it to that local") {
    val wrapperSym = sym("NativeWrapper")
    val wrapperTpe = p.Type.Struct(wrapperSym, Nil)
    val holderRef  = p.Type.Ptr(holderTpe, p.Type.Space.Global)
    val external   = arg("external", pointerTpe)
    val target     = named("target", holderTpe)
    val reference  = named("reference", holderRef)
    val wrapper    = named("wrapper", wrapperTpe)
    val copy       = named("copy", wrapperTpe)
    val result = ArenaView(
      PassTest.program(
        entry(
          args = List(arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global)), external),
          body = List(
            p.Stmt.Var(target, None, isMutable = true),
            p.Stmt.Mut(holderField(target), p.Expr.Alias(selectT(external.named))),
            p.Stmt.Var(
              reference,
              Some(p.Expr.RefTo(selectT(target), None, holderTpe, p.Type.Space.Global, p.Region.Opaque)),
              false
            ),
            p.Stmt.Var(wrapper, None, isMutable = true),
            p.Stmt.Mut(
              p.Term.Select(wrapper, List(p.PathStep.Field("inner")), holderRef).asInstanceOf[p.Term.Select],
              p.Expr.Alias(selectT(reference))
            ),
            p.Stmt.Var(copy, Some(p.Expr.Alias(selectT(wrapper))), isMutable = true)
          ) ::: loadAndIndex(
            p.Term.Select(copy, List(p.PathStep.Field("inner"), p.PathStep.Field("pointer")), pointerTpe),
            named("loaded", pointerTpe)
          )
        ),
        Nil,
        List(
          p.StructDef(capSym, Nil, Nil, Nil),
          p.StructDef(holderSym, Nil, List(named("pointer", pointerTpe)), Nil),
          p.StructDef(wrapperSym, Nil, List(named("inner", holderRef)), Nil)
        )
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
  }

  test("a dereferenced reference slot reads the referenced local field") {
    val counterSym = sym("NativeCounter")
    val counterTpe = p.Type.Struct(counterSym, Nil)
    val cellSym    = sym("NativeCell")
    val cellTpe    = p.Type.Struct(cellSym, Nil)
    val counter    = named("counter", counterTpe)
    val reference  = named("reference", pointerTpe)
    val cell       = named("cell", cellTpe)
    val result = ArenaView(
      PassTest.program(
        entry(
          args = List(arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global))),
          body = List(
            p.Stmt.Var(counter, None, isMutable = true),
            p.Stmt.Var(
              reference,
              Some(
                p.Expr.RefTo(
                  p.Term.Select(counter, List(p.PathStep.Field("count")), p.Type.IntS32),
                  None,
                  p.Type.IntS32,
                  p.Type.Space.Global,
                  p.Region.Opaque
                )
              ),
              false
            ),
            p.Stmt.Var(cell, None, isMutable = true),
            p.Stmt.Mut(
              p.Term.Select(cell, List(p.PathStep.Field("target")), pointerTpe).asInstanceOf[p.Term.Select],
              p.Expr.Alias(selectT(reference))
            ),
            p.Stmt.Var(
              named("value", p.Type.IntS32),
              Some(
                p.Expr.Alias(p.Term.Select(cell, List(p.PathStep.Field("target"), p.PathStep.Deref), p.Type.IntS32))
              ),
              false
            ),
            p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
          )
        ),
        Nil,
        List(
          p.StructDef(capSym, Nil, Nil, Nil),
          p.StructDef(counterSym, Nil, List(named("count", p.Type.IntS32)), Nil),
          p.StructDef(cellSym, Nil, List(named("target", pointerTpe)), Nil)
        )
      ),
      NoopLog
    )
    val read = result.entry.toList.flatMap(_.collectAll[p.Stmt]).collectFirst {
      case p.Stmt.Var(n, Some(p.Expr.Alias(term)), _) if n.symbol == "value" => term
    }
    assertEquals(read, Some(p.Term.Select(counter, List(p.PathStep.Field("count")), p.Type.IntS32)))
  }

  test("a self pointer round-tripped through a same-layout struct pointer forwards its pointer leaf") {
    val twinSym  = sym("NativeHolderTwin")
    val twinTpe  = p.Type.Struct(twinSym, Nil)
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val self     = named("self", p.Type.Ptr(holderTpe, p.Type.Space.Global))
    val twin     = named("twin", p.Type.Ptr(twinTpe, p.Type.Space.Global))
    val back     = named("back", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val result = ArenaView(
      PassTest.program(
        entry(
          args = List(arg(p.Conventions.CaptureArg, p.Type.Ptr(capTpe, p.Type.Space.Global)), external),
          body = List(
            p.Stmt.Var(holder, None, isMutable = true),
            p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
            p.Stmt.Var(
              self,
              Some(p.Expr.RefTo(selectT(holder), None, holderTpe, p.Type.Space.Global, p.Region.Opaque)),
              false
            ),
            p.Stmt.Var(twin, Some(p.Expr.Cast(selectT(self), twin.tpe)), false),
            p.Stmt.Var(back, Some(p.Expr.Cast(selectT(twin), back.tpe)), false)
          ) ::: loadAndIndex(
            p.Term.Select(back, List(p.PathStep.Field("pointer")), pointerTpe),
            named("loaded", pointerTpe)
          )
        ),
        Nil,
        List(
          p.StructDef(capSym, Nil, Nil, Nil),
          p.StructDef(holderSym, Nil, List(named("pointer", pointerTpe)), Nil),
          p.StructDef(twinSym, Nil, List(named("pointer", pointerTpe)), Nil)
        )
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
  }

  test("a local aggregate that is only ever written drops with its pointer stores") {
    val holder  = named("holder", holderTpe)
    val storage = named("storage", p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global))
    val index   = named("index", p.Type.IntS64)
    val program = holderProgram(
      Nil,
      List(
        p.Stmt.Var(storage, None, isMutable = true),
        p.Stmt.Var(index, Some(p.Expr.Alias(p.Term.IntS64Const(1))), isMutable = true),
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(
          holderField(holder),
          p.Expr.RefTo(selectT(storage), Some(selectT(index)), p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val result = ArenaView(program, NoopLog)
    assert(!result.entry.exists(_.collectAll[p.Stmt].exists {
      case p.Stmt.Var(n, _, _) => n.symbol == "holder"
      case _                   => false
    }))
  }

  test("a self pointer reached through pointee-preserving casts forwards its pointer leaf") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val self     = named("self", p.Type.Ptr(holderTpe, p.Type.Space.Global))
    val viewed   = named("viewed", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val result = ArenaView(
      holderProgram(
        List(external),
        List(
          p.Stmt.Var(holder, None, isMutable = true),
          p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
          p.Stmt.Var(
            self,
            Some(p.Expr.RefTo(selectT(holder), None, holderTpe, p.Type.Space.Global, p.Region.Opaque)),
            false
          ),
          p.Stmt.Var(viewed, Some(p.Expr.Cast(selectT(self), viewed.tpe)), false)
        ) ::: loadAndIndex(
          p.Term.Select(viewed, List(p.PathStep.Field("pointer")), pointerTpe),
          named("loaded", pointerTpe)
        )
      ),
      NoopLog
    )
    assertEquals(indexedPointer(result), Some(selectT(external.named)))
  }

  test("a scalar self pointer indexed through a reassignable alias keeps its definition") {
    val flag  = named("flag", p.Type.Bool1)
    val self  = named("self", p.Type.Ptr(p.Type.Bool1, p.Type.Space.Private))
    val alias = named("alias", p.Type.Ptr(p.Type.Bool1, p.Type.Space.Private))
    val result = ArenaView(
      holderProgram(
        Nil,
        List(
          p.Stmt.Var(flag, Some(p.Expr.Alias(p.Term.Bool1Const(true))), isMutable = true),
          p.Stmt.Var(
            self,
            Some(p.Expr.RefTo(selectT(flag), None, p.Type.Bool1, p.Type.Space.Private, p.Region.Opaque)),
            false
          ),
          p.Stmt.Var(alias, Some(p.Expr.Alias(selectT(self))), isMutable = true),
          p.Stmt.Var(
            named("value", p.Type.Bool1),
            Some(p.Expr.Index(selectT(alias), p.Term.IntS64Const(0), p.Type.Bool1)),
            false
          ),
          p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        )
      ),
      NoopLog
    )
    assert(result.entry.exists(_.collectAll[p.Stmt].exists {
      case p.Stmt.Var(n, _, _) => n.symbol == "self"
      case _                   => false
    }))
  }

  test("a struct copy merged from two different sources stays rejected") {
    val external = arg("external", pointerTpe)
    val other    = arg("other", pointerTpe)
    val holder   = named("holder", holderTpe)
    val another  = named("another", holderTpe)
    val merged   = named("merged", holderTpe)
    val program = holderProgram(
      List(external, other),
      List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
        p.Stmt.Var(another, None, isMutable = true),
        p.Stmt.Mut(holderField(another), p.Expr.Alias(selectT(other.named))),
        p.Stmt.Var(merged, None, isMutable = true),
        p.Stmt.Cond(
          p.Term.Bool1Const(true),
          List(p.Stmt.Mut(selectT(merged), p.Expr.Alias(selectT(holder)))),
          List(p.Stmt.Mut(selectT(merged), p.Expr.Alias(selectT(another))))
        )
      ) ::: loadAndIndex(holderField(merged), named("loaded", pointerTpe))
    )
    val error = intercept[IllegalArgumentException](ArenaView(program, NoopLog))
    assert(error.getMessage.contains("native global pointers in local aggregate slots"), error.getMessage)
  }

  test("a local aggregate pointer slot with two sources stays rejected") {
    val external = arg("external", pointerTpe)
    val other    = arg("other", pointerTpe)
    val holder   = named("holder", holderTpe)
    val program = holderProgram(
      List(external, other),
      List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
        p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(other.named)))
      ) ::: loadAndIndex(holderField(holder), named("loaded", pointerTpe))
    )
    val error = intercept[IllegalArgumentException](ArenaView(program, NoopLog))
    assert(error.getMessage.contains("native global pointers in local aggregate slots"))
  }

  test("a local aggregate pointer slot whose holder escapes by address stays rejected") {
    val external = arg("external", pointerTpe)
    val holder   = named("holder", holderTpe)
    val address  = named("address", p.Type.Ptr(holderTpe, p.Type.Space.Private))
    val program = holderProgram(
      List(external),
      List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(holderField(holder), p.Expr.Alias(selectT(external.named))),
        p.Stmt.Var(
          address,
          Some(p.Expr.RefTo(selectT(holder), None, holderTpe, p.Type.Space.Private, p.Region.Opaque)),
          isMutable = false
        ),
        p.Stmt.Var(
          named("present", p.Type.Bool1),
          Some(
            p.Expr.IntrOp(
              p.Intr.LogicNeq(selectT(address), p.Term.NullPtrConst(holderTpe, p.Type.Space.Private, p.Region.Opaque))
            )
          ),
          isMutable = false
        )
      ) ::: loadAndIndex(holderField(holder), named("loaded", pointerTpe))
    )
    val error = intercept[IllegalArgumentException](ArenaView(program, NoopLog))
    assert(error.getMessage.contains("native global pointers in local aggregate slots"))
  }
}
