package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class AddressRefinementSuite extends munit.FunSuite {

  import AddressRefinement.*

  private val capSym = sym("Cap")
  private val capTpe = p.Type.Struct(capSym, Nil)
  private val ptrTpe = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
  private val capPtr = p.Type.Ptr(capTpe, p.Type.Space.Global)

  private def analyse(body: List[p.Stmt], args: List[p.Arg], members: List[p.Named] = List(named("data", ptrTpe))) = {
    val e = entry(args = args, body = body :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
    AddressRefinement.solve(program(e, defs = List(p.StructDef(capSym, Nil, members, Nil))), e)
  }

  private def encoding(analysis: Solution, name: p.Named): Option[Encoding] =
    analysis.facts.get(Query.Binding(name.symbol)).flatMap(_.encoding)

  test("abstract address joins preserve provenance and space correlations") {
    val local = AddressValue.absolute(
      Some(Provenance.Local("local")),
      Some(p.Type.Space.Private)
    )
    val parameter = AddressValue.absolute(
      Some(Provenance.Parameter("parameter")),
      Some(p.Type.Space.Global)
    )

    assertEquals(
      local.join(parameter).alternatives,
      Set(
        AbstractAddress.Absolute(Some(Provenance.Local("local")), Some(p.Type.Space.Private)),
        AbstractAddress.Absolute(Some(Provenance.Parameter("parameter")), Some(p.Type.Space.Global))
      )
    )
    assertEquals(AddressValue.arenaRoot("capture").encodings, Set(Encoding.Absolute))
    assert(AddressValue.arenaRoot("capture").hasArenaRoot)
  }

  test("taking a local scalar address refines a stale global pointer to private") {
    val value   = named("value", p.Type.IntS32)
    val pointer = named("pointer", ptrTpe)
    val out     = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(value, None, isMutable = true),
        p.Stmt.Var(
          pointer,
          Some(p.Expr.RefTo(selectT(value), None, p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      Nil
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.refinedSpace(pointer), Some(p.Type.Space.Private))
  }

  test("casting local array storage refines a stale global pointer to private") {
    val storage = named("storage", p.Type.Arr(p.Type.IntU8, 16, p.Type.Space.Global))
    val pointer = named("pointer", p.Type.Ptr(p.Type.IntU8, p.Type.Space.Global))
    val analysis = analyse(
      List(
        p.Stmt.Var(storage, None, isMutable = true),
        p.Stmt.Var(pointer, Some(p.Expr.Cast(selectT(storage), pointer.tpe)), isMutable = false)
      ),
      Nil
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.refinedSpace(pointer), Some(p.Type.Space.Private))
  }

  test("the logical model encodes a local arena-root address relatively") {
    val cap     = named(p.Conventions.CaptureArg, capPtr)
    val pointer = named("pointer", capPtr)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.RefTo(selectT(cap), None, capTpe, p.Type.Space.Global, p.Region.Opaque)),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val logical = AddressRefinement.solve(
      program(e, defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))),
      e,
      AddressModel.Logical
    )

    assertEquals(logical.diagnostics, Nil)
    assertEquals(encoding(logical, pointer), Some(Encoding.ArenaRelative))
  }

  test("alias constraints close transitively from a capture pointer field") {
    val cap = named(p.Conventions.CaptureArg, capPtr)
    val a   = named("a", ptrTpe)
    val b   = named("b", ptrTpe)
    val out = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(
          a,
          Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          isMutable = false
        ),
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a))), isMutable = false),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(b), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, a), Some(Encoding.ArenaRelative))
    assertEquals(encoding(analysis, b), Some(Encoding.ArenaRelative))
  }

  test("a pointer field of the offload capture is valid at an update site") {
    val cap = named(p.Conventions.ThisReceiver, capPtr)
    val analysis = analyse(
      List(
        p.Stmt.Update(
          p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe),
          p.Term.IntS64Const(0),
          p.Term.IntS32Const(42)
        )
      ),
      List(p.Arg(cap))
    )

    assertEquals(analysis.diagnostics, Nil)
  }

  test("logical arena-view address loads survive lowered base adjustment") {
    val baseSym     = sym("Base")
    val derivedSym  = sym("Derived")
    val baseTpe     = p.Type.Struct(baseSym, Nil)
    val derivedTpe  = p.Type.Struct(derivedSym, Nil)
    val views       = LogicalArenaViewAbi.bindings
    val addressView = LogicalArenaViewAbi.addressBinding
    val loaded      = named("loaded", p.Type.IntS64)
    val carrier     = named("carrier", p.Type.IntS64)
    val offset      = named("offset", p.Type.IntS64)
    val address     = named("address", p.Type.IntS64)
    val pointer     = named("pointer", p.Type.Ptr(baseTpe, p.Type.Space.Global))
    val value       = named("value", p.Type.IntS32)
    val e = entry(
      args = views.map(p.Arg(_)),
      body = List(
        p.Stmt.Var(
          loaded,
          Some(p.Expr.Index(selectT(addressView), p.Term.IntS64Const(0), p.Type.IntS64)),
          false
        ),
        p.Stmt.Var(carrier, Some(p.Expr.Alias(p.Term.IntS64Const(0))), true),
        p.Stmt.Mut(selectT(carrier), p.Expr.Alias(selectT(loaded))),
        p.Stmt.Var(offset, Some(p.Expr.OffsetOf(derivedTpe, "#base_Base")), false),
        p.Stmt.Var(
          address,
          Some(p.Expr.IntrOp(p.Intr.Add(selectT(carrier), selectT(offset), p.Type.IntS64))),
          false
        ),
        p.Stmt.Var(pointer, Some(p.Expr.Cast(selectT(address), pointer.tpe)), false),
        p.Stmt.Var(
          value,
          Some(p.Expr.Alias(p.Term.Select(pointer, List(p.PathStep.Field("value")), p.Type.IntS32))),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(baseSym, Nil, List(named("value", p.Type.IntS32)), Nil),
          p.StructDef(derivedSym, Nil, List(named("#base_Base", baseTpe)), Nil)
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, carrier), Some(Encoding.ArenaRelative))
    assertEquals(encoding(analysis, pointer), Some(Encoding.ArenaRelative))
  }

  test("taking a reference to a materialised value produces a absolute address") {
    val cap     = named(p.Conventions.CaptureArg, capPtr)
    val pointer = named("pointer", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Private))
    val value   = named("value", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(
          pointer,
          Some(
            p.Expr.RefTo(
              p.Term.IntS32Const(10),
              None,
              p.Type.IntS32,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          ),
          false
        ),
        p.Stmt.Var(value, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.Absolute))
  }

  test("an address derived through a local arena pointer retains its arena origin") {
    val baseSym    = sym("Base")
    val derivedSym = sym("Derived")
    val baseTpe    = p.Type.Struct(baseSym, Nil)
    val derivedTpe = p.Type.Struct(derivedSym, Nil)
    val derivedPtr = p.Type.Ptr(derivedTpe, p.Type.Space.Global)
    val basePtr    = p.Type.Ptr(baseTpe, p.Type.Space.Global)
    val cap        = named(p.Conventions.CaptureArg, capPtr)
    val derived    = named("derived", derivedPtr)
    val adjusted   = named("adjusted", basePtr)
    val value      = named("value", p.Type.IntS32)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          derived,
          Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("derived")), derivedPtr))),
          false
        ),
        p.Stmt.Var(
          adjusted,
          Some(
            p.Expr.RefTo(
              p.Term.Select(derived, List(p.PathStep.Field("#base_Base")), baseTpe),
              None,
              baseTpe,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          false
        ),
        p.Stmt.Var(
          value,
          Some(p.Expr.Alias(p.Term.Select(adjusted, List(p.PathStep.Field("value")), p.Type.IntS32))),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, List(named("derived", derivedPtr)), Nil),
          p.StructDef(baseSym, Nil, List(named("value", p.Type.IntS32)), Nil),
          p.StructDef(derivedSym, Nil, List(named("#base_Base", baseTpe)), Nil)
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, adjusted), Some(Encoding.ArenaRelative))
  }

  test("null is refined by an arena assignment through a branch join") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val analysis = analyse(
      List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.Cond(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          ),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.NullPtrConst(p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))
            )
          )
        )
      ),
      List(p.Arg(cap), p.Arg(condition))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.ArenaRelative))
  }

  test("a standalone null pointer is representable during logical base adjustment") {
    val baseSym    = sym("NullBase")
    val derivedSym = sym("NullDerived")
    val baseTpe    = p.Type.Struct(baseSym, Nil)
    val derivedTpe = p.Type.Struct(derivedSym, Nil)
    val derivedPtr = p.Type.Ptr(derivedTpe, p.Type.Space.Global)
    val basePtr    = p.Type.Ptr(baseTpe, p.Type.Space.Global)
    val pointer    = named("pointer", derivedPtr)
    val adjusted   = named("adjusted", basePtr)
    val e = entry(
      body = List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.NullPtrConst(derivedTpe, p.Type.Space.Global, p.Region.Opaque))),
          false
        ),
        p.Stmt.Var(
          adjusted,
          Some(
            p.Expr.RefTo(
              p.Term.Select(pointer, List(p.PathStep.Field("#base_NullBase")), baseTpe),
              None,
              baseTpe,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(baseSym, Nil, Nil, Nil),
          p.StructDef(derivedSym, Nil, List(named("#base_NullBase", baseTpe)), Nil)
        )
      ),
      e,
      AddressModel.Logical
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.ArenaRelative))
  }

  test("a loop joins its zero-iteration null path with a converged arena assignment") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val out       = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.NullPtrConst(p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))),
          isMutable = true
        ),
        p.Stmt.While(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          )
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap), p.Arg(condition))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.ArenaRelative))
  }

  test("a one-sided branch cannot make an uninitialised pointer safe to dereference") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val out       = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.Cond(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          ),
          Nil
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap), p.Arg(condition))
    )

    assert(analysis.diagnostics.exists(_.code == "unresolved-pointer-use"), analysis.diagnostics.mkString("\n"))
  }

  test("a zero-trip loop cannot make an uninitialised pointer safe to dereference") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val out       = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.While(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          )
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap), p.Arg(condition))
    )

    assert(analysis.diagnostics.exists(_.code == "unresolved-pointer-use"), analysis.diagnostics.mkString("\n"))
  }

  test("mixed arena and external coercions select native local storage and plan one resolution") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val external  = named("external", ptrTpe)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val analysis = analyse(
      List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.Cond(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          ),
          List(p.Stmt.Mut(selectT(pointer), p.Expr.Alias(selectT(external))))
        )
      ),
      List(p.Arg(cap), p.Arg(external), p.Arg(condition))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.Absolute))
    assertEquals(
      analysis.coercions.count(x =>
        x.target == Query.Binding(pointer.symbol) && x.coercion == Coercion.ResolveRelative
      ),
      1
    )
  }

  test("a local aggregate pointer slot selects native storage for an arena value") {
    val holderSym = sym("Holder")
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val holder    = named("holder", holderTpe)
    val field =
      p.Term.Select(holder, List(p.PathStep.Field("pointer")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(field, p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(holderSym, Nil, List(named("pointer", ptrTpe)), Nil)
        )
      ),
      e
    )

    val slot = Query.Slot(holder.symbol, List(p.PathStep.Field("pointer")))
    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.facts.get(slot).flatMap(_.encoding), Some(Encoding.Absolute))
    assertEquals(
      analysis.coercions.count(x => x.target == slot && x.coercion == Coercion.ResolveRelative),
      1
    )
  }

  test("pointer arithmetic on an aggregate field uses the stored pointer address space") {
    val holderSym = sym("Holder")
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val holder    = named("holder", holderTpe)
    val pointer   = named("pointer", ptrTpe)
    val field =
      p.Term.Select(holder, List(p.PathStep.Field("pointer")), ptrTpe).asInstanceOf[p.Term.Select]
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Mut(field, p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
        p.Stmt.Var(
          pointer,
          Some(
            p.Expr.RefTo(
              field,
              Some(p.Term.IntS64Const(1)),
              p.Type.IntS32,
              p.Type.Space.Global,
              p.Region.Opaque
            )
          ),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(holderSym, Nil, List(named("pointer", ptrTpe)), Nil)
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.bindings(pointer.symbol).spaces, Set(p.Type.Space.Global))
  }

  test("dynamic array indices share their aggregate pointer-slot encoding") {
    val holderSym = sym("DynamicHolder")
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val arrayTpe  = p.Type.Arr(holderTpe, 8, p.Type.Space.Private)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val array     = named("array", arrayTpe)
    val writeAt   = named("writeAt", p.Type.IntS32)
    val readAt    = named("readAt", p.Type.IntS32)
    val pointer   = named("pointer", ptrTpe)
    def field(index: p.Named): p.Term.Select =
      p.Term.Select(
        array,
        List(p.PathStep.IndexDyn(selectT(index)), p.PathStep.Field("pointer")),
        ptrTpe
      )
    val e = entry(
      args = List(p.Arg(cap), p.Arg(writeAt), p.Arg(readAt)),
      body = List(
        p.Stmt.Var(array, None, isMutable = true),
        p.Stmt.Mut(
          field(writeAt),
          p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
        ),
        p.Stmt.Var(pointer, Some(p.Expr.Alias(field(readAt))), false),
        p.Stmt.Update(selectT(pointer), p.Term.IntS64Const(0), p.Term.IntS32Const(1)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(holderSym, Nil, List(named("pointer", ptrTpe)), Nil)
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.Absolute))
  }

  test("an aggregate copy retains pointer slots written through a nested subobject address") {
    val ownerSym  = sym("Owner")
    val futureSym = sym("Future")
    val ownerTpe  = p.Type.Struct(ownerSym, Nil)
    val futureTpe = p.Type.Struct(futureSym, Nil)
    val ownerPtr  = p.Type.Ptr(ownerTpe, p.Type.Space.Private)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val source    = named("source", futureTpe)
    val owner     = named("owner", ownerPtr)
    val copied    = named("copied", futureTpe)
    val pointer   = named("pointer", ptrTpe)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(source, None, isMutable = true),
        p.Stmt.Var(
          owner,
          Some(
            p.Expr.RefTo(
              p.Term.Select(source, List(p.PathStep.Field("owner")), ownerTpe),
              None,
              ownerTpe,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          ),
          false
        ),
        p.Stmt.Mut(
          p.Term.Select(owner, List(p.PathStep.Field("counter")), ptrTpe),
          p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
        ),
        p.Stmt.Var(copied, Some(p.Expr.Alias(selectT(source))), isMutable = true),
        p.Stmt.Var(
          pointer,
          Some(
            p.Expr.Alias(
              p.Term.Select(
                copied,
                List(p.PathStep.Field("owner"), p.PathStep.Field("counter")),
                ptrTpe
              )
            )
          ),
          false
        ),
        p.Stmt.Update(selectT(pointer), p.Term.IntS64Const(0), p.Term.IntS32Const(1)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(ownerSym, Nil, List(named("counter", ptrTpe)), Nil),
          p.StructDef(futureSym, Nil, List(named("owner", ownerTpe)), Nil)
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, pointer), Some(Encoding.Absolute))
  }

  test("taking the address of a pointer binding and loading it preserves the stored encoding") {
    val cap     = named(p.Conventions.CaptureArg, capPtr)
    val pointer = named("pointer", ptrTpe)
    val refTpe  = p.Type.Ptr(ptrTpe, p.Type.Space.Private)
    val ref     = named("ref", refTpe)
    val loaded  = named("loaded", ptrTpe)
    val analysis = analyse(
      List(
        p.Stmt.Var(
          pointer,
          Some(p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          false
        ),
        p.Stmt.Var(
          ref,
          Some(p.Expr.RefTo(selectT(pointer), None, ptrTpe, p.Type.Space.Private, p.Region.Rooted(pointer))),
          false
        ),
        p.Stmt.Var(loaded, Some(p.Expr.Index(selectT(ref), p.Term.IntS64Const(0), ptrTpe)), false)
      ),
      List(p.Arg(cap))
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, loaded), Some(Encoding.ArenaRelative))
  }

  test("an external absolute address cannot be stored in a capture arena pointer slot") {
    val cap      = named(p.Conventions.CaptureArg, capPtr)
    val external = named("external", ptrTpe)
    val analysis = analyse(
      List(
        p.Stmt.Mut(
          p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe),
          p.Expr.Alias(selectT(external))
        )
      ),
      List(p.Arg(cap), p.Arg(external))
    )

    assert(analysis.diagnostics.exists(_.code == "absolute-to-arena-slot"), analysis.diagnostics.mkString("\n"))
  }

  test("an unknown pointer result cannot be dereferenced silently") {
    val cap     = named(p.Conventions.CaptureArg, capPtr)
    val pointer = named("pointer", ptrTpe)
    val out     = named("out", p.Type.IntS32)
    val analysis = analyse(
      List(
        p.Stmt.Var(pointer, Some(p.Expr.ForeignCall("unknown_pointer", Nil, ptrTpe)), false),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false)
      ),
      List(p.Arg(cap))
    )

    val diagnostic = analysis.diagnostics.find(_.code == "unresolved-pointer-use")
    assert(diagnostic.nonEmpty, analysis.diagnostics.mkString("\n"))
    assert(diagnostic.exists(_.message.contains("pointer")))
  }

  test("a helper return preserves the encoding of its pointer argument") {
    val cap      = named(p.Conventions.CaptureArg, capPtr)
    val input    = named("input", ptrTpe)
    val returned = named("returned", ptrTpe)
    val out      = named("out", p.Type.IntS32)
    val helper = fn(
      "identity",
      args = List(p.Arg(input)),
      rtn = ptrTpe,
      body = List(p.Stmt.Return(p.Expr.Alias(selectT(input)))),
      visibility = p.Function.Visibility.Internal
    )
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          returned,
          Some(
            p.Expr.Invoke(
              p.Type.FnRef(helper.name),
              Nil,
              None,
              List(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)),
              ptrTpe
            )
          ),
          false
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(returned), p.Term.IntS64Const(0), p.Type.IntS32)), false),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(e, functions = List(helper), defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, returned), Some(Encoding.ArenaRelative))
  }

  test("a pointer-returning helper remains polymorphic across arena and native call sites") {
    val cap          = named(p.Conventions.CaptureArg, capPtr)
    val external     = named("external", ptrTpe)
    val input        = named("input", ptrTpe)
    val arenaResult  = named("arenaResult", ptrTpe)
    val nativeResult = named("nativeResult", ptrTpe)
    val arenaValue   = named("arenaValue", p.Type.IntS32)
    val nativeValue  = named("nativeValue", p.Type.IntS32)
    val helper = fn(
      "identity",
      args = List(p.Arg(input)),
      rtn = ptrTpe,
      body = List(p.Stmt.Return(p.Expr.Alias(selectT(input)))),
      visibility = p.Function.Visibility.Internal
    )
    def invoke(argument: p.Term): p.Expr =
      p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, List(argument), ptrTpe)
    val e = entry(
      args = List(p.Arg(cap), p.Arg(external)),
      body = List(
        p.Stmt.Var(
          arenaResult,
          Some(invoke(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          false
        ),
        p.Stmt.Var(nativeResult, Some(invoke(selectT(external))), false),
        p.Stmt.Var(arenaValue, Some(p.Expr.Index(selectT(arenaResult), p.Term.IntS64Const(0), p.Type.IntS32)), false),
        p.Stmt.Var(
          nativeValue,
          Some(p.Expr.Index(selectT(nativeResult), p.Term.IntS64Const(0), p.Type.IntS32)),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(e, functions = List(helper), defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, arenaResult), Some(Encoding.ArenaRelative))
    assertEquals(encoding(analysis, nativeResult), Some(Encoding.Absolute))
  }

  test("pointer-return summaries compose through a helper call chain") {
    val cap      = named(p.Conventions.CaptureArg, capPtr)
    val innerArg = named("innerArg", ptrTpe)
    val outerArg = named("outerArg", ptrTpe)
    val returned = named("returned", ptrTpe)
    val out      = named("out", p.Type.IntS32)
    val inner = fn(
      "inner",
      args = List(p.Arg(innerArg)),
      rtn = ptrTpe,
      body = List(p.Stmt.Return(p.Expr.Alias(selectT(innerArg)))),
      visibility = p.Function.Visibility.Internal
    )
    val outer = fn(
      "outer",
      args = List(p.Arg(outerArg)),
      rtn = ptrTpe,
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(p.Type.FnRef(inner.name), Nil, None, List(selectT(outerArg)), ptrTpe)
        )
      ),
      visibility = p.Function.Visibility.Internal
    )
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(
          returned,
          Some(
            p.Expr.Invoke(
              p.Type.FnRef(outer.name),
              Nil,
              None,
              List(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe)),
              ptrTpe
            )
          ),
          false
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(returned), p.Term.IntS64Const(0), p.Type.IntS32)), false),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        functions = List(inner, outer),
        defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, returned), Some(Encoding.ArenaRelative))
  }

  test("pointer-return summaries include inherited capture facts in their identity") {
    val cap          = named(p.Conventions.CaptureArg, capPtr)
    val external     = named("external", ptrTpe)
    val captured     = named("captured", ptrTpe)
    val outerArg     = named("captured", ptrTpe)
    val arenaResult  = named("arenaResult", ptrTpe)
    val nativeResult = named("nativeResult", ptrTpe)
    val arenaValue   = named("arenaValue", p.Type.IntS32)
    val nativeValue  = named("nativeValue", p.Type.IntS32)
    val inner = fn(
      "capturedIdentity",
      moduleCaptures = List(p.Arg(captured)),
      rtn = ptrTpe,
      body = List(p.Stmt.Return(p.Expr.Alias(selectT(captured)))),
      visibility = p.Function.Visibility.Internal
    )
    val outer = fn(
      "forwardCapture",
      args = List(p.Arg(outerArg)),
      rtn = ptrTpe,
      body = List(
        p.Stmt.Return(p.Expr.Invoke(p.Type.FnRef(inner.name), Nil, None, Nil, ptrTpe))
      ),
      visibility = p.Function.Visibility.Internal
    )
    def invoke(argument: p.Term): p.Expr =
      p.Expr.Invoke(p.Type.FnRef(outer.name), Nil, None, List(argument), ptrTpe)
    val e = entry(
      args = List(p.Arg(cap), p.Arg(external)),
      body = List(
        p.Stmt.Var(
          arenaResult,
          Some(invoke(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))),
          false
        ),
        p.Stmt.Var(nativeResult, Some(invoke(selectT(external))), false),
        p.Stmt.Var(arenaValue, Some(p.Expr.Index(selectT(arenaResult), p.Term.IntS64Const(0), p.Type.IntS32)), false),
        p.Stmt.Var(
          nativeValue,
          Some(p.Expr.Index(selectT(nativeResult), p.Term.IntS64Const(0), p.Type.IntS32)),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        functions = List(inner, outer),
        defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(encoding(analysis, arenaResult), Some(Encoding.ArenaRelative))
    assertEquals(encoding(analysis, nativeResult), Some(Encoding.Absolute))
  }

  test("an unresolved pointer cannot escape through a function return") {
    val pointer = named("pointer", ptrTpe)
    val e = entry(
      body = List(
        p.Stmt.Var(pointer, Some(p.Expr.ForeignCall("unknown_pointer", Nil, ptrTpe)), false),
        p.Stmt.Return(p.Expr.Alias(selectT(pointer)))
      )
    ).modifyDecl(_.copy(rtn = ptrTpe))
    val analysis = AddressRefinement.solve(program(e), e)

    val diagnostic = analysis.diagnostics.find(_.code == "unresolved-pointer-use")
    assert(diagnostic.exists(_.message.contains("exported return")), analysis.diagnostics.mkString("\n"))
  }

  test("a absolute address cannot hide an incompatible address-space join") {
    val local     = named("local", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local))
    val global    = named("global", ptrTpe)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val out       = named("out", p.Type.IntS32)
    val e = entry(
      args = List(p.Arg(local), p.Arg(global), p.Arg(condition)),
      body = List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.Cond(
          selectT(condition),
          List(p.Stmt.Mut(selectT(pointer), p.Expr.Cast(selectT(local), ptrTpe))),
          List(p.Stmt.Mut(selectT(pointer), p.Expr.Alias(selectT(global))))
        ),
        p.Stmt.Var(out, Some(p.Expr.Index(selectT(pointer), p.Term.IntS64Const(0), p.Type.IntS32)), false),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(program(e), e, AddressModel.Logical)

    assert(
      analysis.diagnostics.exists(_.code == "incompatible-address-spaces"),
      analysis.diagnostics.mkString("\n")
    )
  }

  test("an unresolved pointer field in a local aggregate cannot be traversed") {
    val nodeSym   = sym("Node")
    val holderSym = sym("Holder")
    val nodeTpe   = p.Type.Struct(nodeSym, Nil)
    val nodePtr   = p.Type.Ptr(nodeTpe, p.Type.Space.Global)
    val holderTpe = p.Type.Struct(holderSym, Nil)
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val holder    = named("holder", holderTpe)
    val value     = named("value", p.Type.IntS32)
    val e = entry(
      args = List(p.Arg(cap)),
      body = List(
        p.Stmt.Var(holder, None, isMutable = true),
        p.Stmt.Var(
          value,
          Some(
            p.Expr.Alias(
              p.Term.Select(
                holder,
                List(p.PathStep.Field("node"), p.PathStep.Field("value")),
                p.Type.IntS32
              )
            )
          ),
          false
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(capSym, Nil, Nil, Nil),
          p.StructDef(holderSym, Nil, List(named("node", nodePtr)), Nil),
          p.StructDef(nodeSym, Nil, List(named("value", p.Type.IntS32)), Nil)
        )
      ),
      e
    )

    assert(analysis.diagnostics.exists(_.code == "unresolved-pointer-use"), analysis.diagnostics.mkString("\n"))
  }

  test("a downcast from a local base subobject recovers the derived aggregate slots") {
    val baseSym                   = sym("Base")
    val derivedSym                = sym("Derived")
    val baseTpe: p.Type.Struct    = p.Type.Struct(baseSym, Nil)
    val derivedTpe: p.Type.Struct = p.Type.Struct(derivedSym, Nil)
    val emptyTpe                  = p.Type.Struct(sym("EmptyBaseStorage"), Nil)
    val baseField                 = s"${p.Conventions.BaseFieldPrefix}_${baseSym.fqcn}"
    val localPtr                  = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local)
    val basePtr                   = p.Type.Ptr(baseTpe, p.Type.Space.Private)
    val derivedPtr                = p.Type.Ptr(derivedTpe, p.Type.Space.Private)
    val storage                   = named("storage", derivedTpe)
    val local                     = named("local", localPtr)
    val base                      = named("base", basePtr)
    val derived                   = named("derived", derivedPtr)
    val loaded                    = named("loaded", ptrTpe)
    val value                     = named("value", p.Type.IntS32)
    val dataPath                  = List(p.PathStep.Field(baseField), p.PathStep.Field("data"))
    val e = entry(
      body = List(
        p.Stmt.Var(storage, None, isMutable = true),
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(p.Term.Select(storage, dataPath, ptrTpe), p.Expr.Alias(selectT(local))),
        p.Stmt.Var(
          base,
          Some(
            p.Expr.RefTo(
              p.Term.Select(storage, List(p.PathStep.Field(baseField)), baseTpe),
              None,
              baseTpe,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Var(derived, Some(p.Expr.Cast(selectT(base), derivedPtr))),
        p.Stmt.Var(
          loaded,
          Some(
            p.Expr.Alias(
              p.Term.Select(
                derived,
                p.PathStep.Deref :: dataPath,
                ptrTpe
              )
            )
          )
        ),
        p.Stmt.Var(value, Some(p.Expr.Index(selectT(loaded), p.Term.IntS64Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(baseSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(derivedSym, Nil, List(named(baseField, emptyTpe)), List(baseTpe))
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.refinedSpace(loaded), Some(p.Type.Space.Local))
  }

  test("an inherited pointer projection does not duplicate its base-subobject path") {
    val baseSym                   = sym("ProjectionBase")
    val derivedSym                = sym("ProjectionDerived")
    val baseTpe: p.Type.Struct    = p.Type.Struct(baseSym, Nil)
    val derivedTpe: p.Type.Struct = p.Type.Struct(derivedSym, Nil)
    val emptyTpe                  = p.Type.Struct(sym("ProjectionBaseStorage"), Nil)
    val baseField                 = s"${p.Conventions.BaseFieldPrefix}_${baseSym.fqcn}"
    val basePtr                   = p.Type.Ptr(baseTpe, p.Type.Space.Private)
    val localPtr                  = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local)
    val storage                   = named("storage", derivedTpe)
    val local                     = named("local", localPtr)
    val base                      = named("base", basePtr)
    val loaded                    = named("loaded", ptrTpe)
    val value                     = named("value", p.Type.IntS32)
    val slotPath                  = List(p.PathStep.Field(baseField), p.PathStep.Field("data"))
    val projected =
      p.Term.Select(base, p.PathStep.Deref :: slotPath, ptrTpe)
    val e = entry(
      body = List(
        p.Stmt.Var(storage, None, isMutable = true),
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(p.Term.Select(storage, slotPath, ptrTpe), p.Expr.Alias(selectT(local))),
        p.Stmt.Var(
          base,
          Some(
            p.Expr.RefTo(
              p.Term.Select(storage, List(p.PathStep.Field(baseField)), baseTpe),
              None,
              baseTpe,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Var(
          loaded,
          Some(
            p.Expr.Alias(
              projected
            )
          )
        ),
        p.Stmt.Var(value, Some(p.Expr.Index(selectT(loaded), p.Term.IntS64Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(baseSym, Nil, List(named("data", ptrTpe)), Nil),
          p.StructDef(derivedSym, Nil, List(named(baseField, emptyTpe)), List(baseTpe))
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.refinedSpace(loaded), Some(p.Type.Space.Local))
  }

  test("a base-subobject alias can project a direct member of its derived storage") {
    val baseSym                   = sym("Adaptor")
    val derivedSym                = sym("Permutation")
    val baseTpe: p.Type.Struct    = p.Type.Struct(baseSym, Nil)
    val derivedTpe: p.Type.Struct = p.Type.Struct(derivedSym, Nil)
    val emptyTpe                  = p.Type.Struct(sym("AdaptorStorage"), Nil)
    val baseField                 = s"${p.Conventions.BaseFieldPrefix}_${baseSym.fqcn}"
    val basePtr                   = p.Type.Ptr(baseTpe, p.Type.Space.Private)
    val localPtr                  = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Local)
    val storage                   = named("storage", derivedTpe)
    val local                     = named("local", localPtr)
    val base                      = named("base", basePtr)
    val loaded                    = named("loaded", ptrTpe)
    val value                     = named("value", p.Type.IntS32)
    val projected                 = p.Term.Select(base, List(p.PathStep.Deref, p.PathStep.Field("element")), ptrTpe)
    val e = entry(
      body = List(
        p.Stmt.Var(storage, None, isMutable = true),
        p.Stmt.Var(
          local,
          Some(p.Expr.Alloc(p.Type.IntS32, p.Term.IntS64Const(4), p.Type.Space.Local, p.Region.Rooted(local)))
        ),
        p.Stmt.Mut(
          p.Term.Select(storage, List(p.PathStep.Field("element")), ptrTpe),
          p.Expr.Alias(selectT(local))
        ),
        p.Stmt.Var(
          base,
          Some(
            p.Expr.RefTo(
              p.Term.Select(storage, List(p.PathStep.Field(baseField)), baseTpe),
              None,
              baseTpe,
              p.Type.Space.Private,
              p.Region.Opaque
            )
          )
        ),
        p.Stmt.Var(
          loaded,
          Some(p.Expr.Alias(projected))
        ),
        p.Stmt.Var(value, Some(p.Expr.Index(selectT(loaded), p.Term.IntS64Const(0), p.Type.IntS32))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(
        e,
        defs = List(
          p.StructDef(baseSym, Nil, List(named("iterator", ptrTpe)), Nil),
          p.StructDef(
            derivedSym,
            Nil,
            List(named(baseField, emptyTpe), named("element", ptrTpe)),
            List(baseTpe)
          )
        )
      ),
      e
    )

    assertEquals(analysis.diagnostics, Nil)
    assertEquals(analysis.refinedSpace(loaded), Some(p.Type.Space.Local))
  }

  test("logical address model rejects a local binding that joins arena-relative and absolute addresses") {
    val cap       = named(p.Conventions.CaptureArg, capPtr)
    val external  = named("external", ptrTpe)
    val condition = named("condition", p.Type.Bool1)
    val pointer   = named("pointer", ptrTpe)
    val e = entry(
      args = List(p.Arg(cap), p.Arg(external), p.Arg(condition)),
      body = List(
        p.Stmt.Var(pointer, None, isMutable = true),
        p.Stmt.Cond(
          selectT(condition),
          List(
            p.Stmt.Mut(
              selectT(pointer),
              p.Expr.Alias(p.Term.Select(cap, List(p.PathStep.Field("data")), ptrTpe))
            )
          ),
          List(p.Stmt.Mut(selectT(pointer), p.Expr.Alias(selectT(external))))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val analysis = AddressRefinement.solve(
      program(e, defs = List(p.StructDef(capSym, Nil, List(named("data", ptrTpe)), Nil))),
      e,
      AddressModel.Logical
    )

    assert(
      analysis.diagnostics.exists(_.code == "logical-mixed-encoding"),
      analysis.diagnostics.mkString("\n")
    )
  }
}
