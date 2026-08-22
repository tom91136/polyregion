package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ListBuffer

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// generic single-arena lowering for flat-address-space binding-slot backends (c_source, opencl1_1, metal).
// the dispatch marshals the whole reachable capture graph into one device arena and binds the arena base as
// the capture arg, so the capture sits at offset 0 and every pointer slot holds an i64 byte offset. rooted
// field reads are unchanged; ONLY an arena-relative (Opaque) pointer deref changes - the offset resolves
// through `(T*)&arena8[(u64)off]`. a Select walks step by step, resolving against the arena the moment the
// running value is a pointer LOADED from memory (an already-read pointer field, then stepped through), so a
// chain like vector ref -> records -> _M_p -> chars resolves itself one hop at a time
// examples:
//   cap.x                       ->  cap.x                                       // rooted read, unchanged
//   p[i]   (p Opaque offset)    ->  ((T*)&arena8[(u64)p])[i]                    // arena-relative deref
//   vec.records._M_p[k]         ->  resolve _M_p hop, then ((T*)&arena8[..])[k] // loaded ptr crossed mid-path
//   &result.#base (result off)  ->  (T*)((u64)&real - (u64)arena8)              // keep the address in offset space
// edge cases:
//   pointer Rooted at a stack local  ->  not arena-resolved (real pointer, left to the backend)
//   arena pointer's outer space      ->  forced Global (conformant OpenCL C rejects a space-changing cast)
//   inline deref in an Intr/Math op  ->  every operand term routed through the arena
object ArenaLower extends ProgramPass {

  private enum ArenaRep {
    case Unknown, Plain, Base, Offset, Mixed
    case Reference(symbol: String)
  }

  private case class ArenaSlot(root: String, path: List[p.PathStep])

  private case class ArenaState(
      capture: p.Named,
      rooted: Map[p.Named, p.Region],
      localAggregates: Set[String],
      roots: Map[String, ArenaRep] = Map.empty,
      slots: Map[ArenaSlot, ArenaRep] = Map.empty
  ) {
    def regionAt(term: p.Term): p.Region = Provenance.at(rooted, term)

    def localRoot(root: p.Named): Option[String] =
      rooted
        .get(root)
        .orElse(rooted.collectFirst { case (named, region) if named.symbol == root.symbol => region })
        .collect { case p.Region.Rooted(n) if localAggregates(n.symbol) => n.symbol }
        .orElse(Option.when(localAggregates(root.symbol))(root.symbol))

    def slot(root: p.Named, steps: List[p.PathStep]): Option[(Int, ArenaRep)] =
      localRoot(root).flatMap { symbol =>
        slots.iterator
          .collect { case (ArenaSlot(`symbol`, path), rep) if steps.startsWith(path) => path.size -> rep }
          .maxByOption(_._1)
      }

    def root(symbol: String): ArenaRep       = roots.getOrElse(symbol, ArenaRep.Unknown)
    def isArenaRooted(term: p.Term): Boolean = regionAt(term) == p.Region.Rooted(capture)
    def offsetRoots: Set[String]             = roots.collect { case (symbol, ArenaRep.Offset) => symbol }.toSet
  }

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private val ctr = new AtomicLong(0L)

  private def pointeeSpace(t: p.Type): p.Type.Space = t match {
    case p.Type.Ptr(_, s) => s
    case _                => p.Type.Space.Global
  }
  private def isPtr(t: p.Type): Boolean = t match { case _: p.Type.Ptr => true; case _ => false }
  private def canBeArenaOffset(t: p.Type): Boolean = t match {
    case p.Type.Ptr(_, p.Type.Space.Global) => true
    case _                                  => false
  }
  private def canTermBeArenaOffset(t: p.Term): Boolean = t match {
    case p.Term.Select(root, Nil, _) => canBeArenaOffset(root.tpe)
    case _                           => canBeArenaOffset(t.tpe)
  }
  private def isAddressInt(t: p.Type): Boolean     = t == p.Type.IntU64 || t == p.Type.IntS64
  private def canCarryArenaRep(t: p.Type): Boolean = canBeArenaOffset(t) || isAddressInt(t)

  // an arena pointer's data lives in the single global buffer, so its outermost address space is always
  // global; conformant OpenCL C rejects a cast that changes it (nvidia tolerates a stray private outer)
  private def globalOuter(t: p.Type): p.Type = t match {
    case p.Type.Ptr(c, _) => p.Type.Ptr(c, p.Type.Space.Global)
    case _                => t
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val members = program.defs.iterator.map(d => d.name -> d.members).toMap
    program.copy(entry = run(members, program.entry), functions = program.functions.map(run(members, _)))
  }

  private def run(members: Map[p.Sym, List[p.Named]], f: p.Function): p.Function = captureRoot(f) match {
    case None => f
    case Some((capN, _)) =>
      val rooted = Provenance.derivedIn(f)
      val localAggregates = f
        .collectAll[p.Stmt]
        .collect {
          case p.Stmt.Var(n, _, _) if !isPtr(n.tpe) => n.symbol
        }
        .toSet
      val state     = analyseArenaState(f, ArenaState(capN, rooted, localAggregates))
      val arena8    = p.Named("#arena_base", BytePtr)
      val rewritten = mapStmtsRec(f.body)(rwLeaf(members, state, arena8))
      val capDecl   = p.Stmt.Var(capN, Some(p.Expr.Cast(sel(arena8), capN.tpe)), isMutable = false)
      def replaceCapture(a: p.Arg): p.Arg = if (a.named == capN) a.copy(named = arena8, boundary = None) else a
      f.copy(
        decl = f.decl.copy(
          receiver = f.receiver.map(replaceCapture),
          args = f.args.map(replaceCapture)
        ),
        body = capDecl :: rewritten
      )
  }

  private def analyseArenaState(f: p.Function, initial: ArenaState): ArenaState = {
    import ArenaRep.*

    val statements = f.collectAll[p.Stmt]
    val assignments = statements
      .flatMap {
        case p.Stmt.Var(n, Some(e), _) if canCarryArenaRep(n.tpe)               => Some(n.symbol -> e)
        case p.Stmt.Mut(p.Term.Select(n, Nil, _), e) if canCarryArenaRep(n.tpe) => Some(n.symbol -> e)
        case _                                                                  => None
      }
      .groupMap(_._1)(_._2)
    val arenaRooted = initial.rooted.collect {
      case (named, p.Region.Rooted(root)) if root == initial.capture && canBeArenaOffset(named.tpe) => named.symbol
    }.toSet
    val slotAssignments = statements
      .flatMap {
        case p.Stmt.Mut(p.Term.Select(root, steps, tpe), e) if steps.nonEmpty && isPtr(tpe) =>
          initial.localRoot(root).map(symbol => ArenaSlot(symbol, steps) -> e)
        case _ => None
      }
      .groupMap(_._1)(_._2)
    val slotCopies = statements.flatMap {
      case p.Stmt.Var(target, Some(p.Expr.Alias(p.Term.Select(source, prefix, _))), _) if !isPtr(target.tpe) =>
        initial.localRoot(source).map((target.symbol, _, prefix))
      case p.Stmt.Mut(p.Term.Select(target, Nil, tpe), p.Expr.Alias(p.Term.Select(source, prefix, _))) if !isPtr(tpe) =>
        initial.localRoot(source).map((target.symbol, _, prefix))
      case _ => None
    }

    def joinAssignment(x: ArenaRep, y: ArenaRep): ArenaRep =
      if (x == Unknown) y
      else if (y == Unknown || x == y) x
      else Mixed

    def termRep(state: ArenaState, t: p.Term): ArenaRep = t match {
      case p.Term.Select(root, Nil, _) if root.symbol == state.capture.symbol => Base
      case p.Term.Select(root, Nil, _)                                        => state.root(root.symbol)
      case p.Term.Select(root, steps, _) if steps.nonEmpty =>
        state.slot(root, steps) match {
          case Some((depth, Base | Offset)) if depth < steps.size && canBeArenaOffset(t.tpe) => Offset
          case Some((_, rep))                                                                => rep
          case None if !canBeArenaOffset(t.tpe)                                              => Plain
          case None if state.localRoot(root).nonEmpty                                        => Unknown
          case None if state.isArenaRooted(t)                                                => Offset
          case None if state.root(root.symbol) == Offset                                     => Offset
          case None                                                                          => Plain
        }
      case _: p.Term.NullPtrConst => Unknown
      case _                      => Plain
    }

    def arithmeticRep(x: ArenaRep): ArenaRep = x match {
      case Reference(_) => Plain
      case other        => other
    }
    def addRep(x: ArenaRep, y: ArenaRep): ArenaRep = (arithmeticRep(x), arithmeticRep(y)) match {
      case (Mixed, _) | (_, Mixed)         => Mixed
      case (Offset, Base) | (Base, Offset) => Mixed
      case (Offset, Offset) | (Base, Base) => Mixed
      case (Offset, _) | (_, Offset)       => Offset
      case (Base, _) | (_, Base)           => Base
      case (Unknown, _) | (_, Unknown)     => Unknown
      case _                               => Plain
    }
    def subRep(x: ArenaRep, y: ArenaRep): ArenaRep = (arithmeticRep(x), arithmeticRep(y)) match {
      case (Mixed, _) | (_, Mixed)        => Mixed
      case (Offset | Base, Offset | Base) => Plain
      case (Offset, Plain)                => Offset
      case (Base, Plain)                  => Base
      case (Unknown, _) | (_, Unknown)    => Unknown
      case (Plain, Plain | Offset | Base) => Plain
      case _                              => Plain
    }

    def exprRep(state: ArenaState, e: p.Expr): ArenaRep = e match {
      // `&p` is a real pointer to the local binding slot for `p`; indexing it later loads whatever
      // representation `p` carries, rather than a pointer stored in the device arena.
      case p.Expr.RefTo(p.Term.Select(root, Nil, tpe), None, comp, _, _) if isPtr(tpe) && comp == tpe =>
        Reference(root.symbol)
      case p.Expr.RefTo(t, _, _, p.Type.Space.Global, _) if isPtr(t.tpe) =>
        termRep(state, t)
      case p.Expr.RefTo(t, _, _, p.Type.Space.Global, _) =>
        t match {
          case p.Term.Select(_, steps, _) if steps.nonEmpty && state.isArenaRooted(t) =>
            Offset
          case p.Term.Select(root, _, _) if state.root(root.symbol) == Offset => Offset
          case _                                                              => Plain
        }
      case p.Expr.Alias(_: p.Term.NullPtrConst) => Unknown
      case p.Expr.Alias(t)                      => termRep(state, t)
      case p.Expr.Cast(t, _)                    => termRep(state, t)
      // Pointer values stored in the arena are byte-offset tokens too. This is the extra hop in
      // pointer-to-pointer captures: indexing the outer offset loads an encoded inner pointer.
      case p.Expr.Index(base, _, p.Type.Ptr(_, p.Type.Space.Global)) =>
        termRep(state, base) match {
          case Reference(symbol) => state.root(symbol)
          case Offset | Base     => Offset
          case Mixed             => Mixed
          case Unknown           => Unknown
          case _                 => Plain
        }
      case p.Expr.IntrOp(p.Intr.Add(x, y, _)) => addRep(termRep(state, x), termRep(state, y))
      case p.Expr.IntrOp(p.Intr.Sub(x, y, _)) => subRep(termRep(state, x), termRep(state, y))
      case _                                  => Plain
    }

    @annotation.tailrec
    def close(state: ArenaState, remaining: Int): ArenaState = {
      val roots = assignments.view
        .mapValues { es =>
          es.iterator.map(exprRep(state, _)).reduce(joinAssignment)
        }
        .map { case (symbol, rep) => symbol -> (if (rep == Plain && arenaRooted(symbol)) Offset else rep) }
        .toMap
      val directSlots = slotAssignments.iterator.map { case (slot, es) =>
        val rep = es.iterator
          .map(exprRep(state, _))
          // ArenaLower stores real pointers in local aggregates, even when their source is encoded.
          .map { case Offset => Base; case x => x }
          .reduce(joinAssignment)
        slot -> rep
      }
      val copiedSlots = slotCopies.iterator.flatMap { case (target, source, prefix) =>
        state.slots.iterator.collect {
          case (ArenaSlot(`source`, path), rep) if path.startsWith(prefix) =>
            ArenaSlot(target, path.drop(prefix.size)) -> rep
        }
      }
      val next = state.copy(
        roots = roots,
        slots = (directSlots ++ copiedSlots).toList.groupMapReduce(_._1)(_._2)(joinAssignment)
      )
      if (next == state || remaining == 0) next else close(next, remaining - 1)
    }

    val state = close(initial, assignments.size + slotAssignments.size + slotCopies.size + 1)
    val mixed = state.roots.collect { case (symbol, Mixed) => symbol }.toList.sorted ++ state.slots
      .collect { case (ArenaSlot(root, path), Mixed) =>
        s"$root.${path.mkString(".")}"
      }
      .toList
      .sorted
    if (mixed.nonEmpty)
      throw RuntimeException(
        s"arena representation changes between real pointers and byte offsets: ${mixed.mkString(", ")}"
      )
    state
  }

  private def rwLeaf(
      members: Map[p.Sym, List[p.Named]],
      state: ArenaState,
      arena8: p.Named
  )(leaf: p.Stmt): List[p.Stmt] = {
    import ArenaRep.*

    val offsetRoots = state.offsetRoots
    val pre         = ListBuffer.empty[p.Stmt]
    val arenaBases  = scala.collection.mutable.Set.empty[String]

    def arenaLValue(t: p.Term): Boolean = t match {
      case p.Term.Select(root, _, _) =>
        offsetRoots(root.symbol) || state.isArenaRooted(t)
      case _ => false
    }
    def localAggregateLValue(t: p.Term): Boolean = state.regionAt(t) match {
      case p.Region.Rooted(n) => state.localAggregates(n.symbol)
      case p.Region.Opaque    => false
    }
    def offsetTerm(t: p.Term): Boolean = t match {
      case p.Term.Select(root, steps, _) =>
        state.slot(root, steps) match {
          case Some((depth, Base))     => depth < steps.size
          case Some((_, Offset))       => true
          case Some((_, Plain))        => false
          case Some((_, Unknown))      => false
          case Some((_, Mixed))        => false
          case Some((_, Reference(_))) => false
          case None =>
            offsetRoots(root.symbol) ||
            (steps.nonEmpty && canBeArenaOffset(t.tpe) && arenaBases(root.symbol)) ||
            (steps.nonEmpty && canBeArenaOffset(t.tpe) && state.isArenaRooted(t))
        }
      case _ => false
    }
    def offsetExpr(e: p.Expr): Boolean = e match {
      case p.Expr.RefTo(t, _, _, p.Type.Space.Global, _) =>
        if (isPtr(t.tpe)) offsetVal(t) else arenaLValue(t)
      case p.Expr.Alias(t)                    => offsetTerm(t)
      case p.Expr.Cast(t, _)                  => offsetTerm(t)
      case p.Expr.IntrOp(p.Intr.Add(x, y, _)) => offsetTerm(x) || offsetTerm(y)
      case p.Expr.IntrOp(p.Intr.Sub(x, _, _)) => offsetTerm(x)
      case _                                  => false
    }
    def offsetVal(t: p.Term): Boolean =
      canTermBeArenaOffset(t) && offsetTerm(t)
    def offsetNamed(n: p.Named): p.Named =
      if (canBeArenaOffset(n.tpe) && offsetRoots(n.symbol))
        n.copy(tpe = globalOuter(n.tpe))
      else n

    def memberTpe(sym: p.Sym, field: String): Option[p.Type] =
      members.get(sym).flatMap(_.find(_.symbol == field).map(_.tpe))
    def stepTpe(cur: p.Type, step: p.PathStep): p.Type = step match {
      case p.PathStep.Field(f) =>
        (cur match {
          case p.Type.Ptr(p.Type.Struct(s, _), _) => memberTpe(s, f)
          case p.Type.Struct(s, _)                => memberTpe(s, f)
          case _                                  => None
        }).getOrElse(cur)
      case _ => cur match { case p.Type.Ptr(c, _) => c; case p.Type.Arr(c, _, _) => c; case _ => cur }
    }

    def toU64(t: p.Term): (p.Term, List[p.Stmt]) =
      if (t.tpe == U64) (t, Nil)
      else {
        val n = p.Named(s"#ao${ctr.incrementAndGet()}", U64);
        (sel(n), List(p.Stmt.Var(n, Some(p.Expr.Cast(t, U64)), isMutable = false)))
      }
    def emitU64(t: p.Term): p.Term = {
      val (u, ss) = toU64(t); pre ++= ss; u
    }
    def bind(hint: String, e: p.Expr): p.Term = {
      val n = p.Named(s"#$hint${ctr.incrementAndGet()}", e.tpe); pre += p.Stmt.Var(n, Some(e), isMutable = false);
      sel(n)
    }
    def byteSize(t: p.Type): p.Term =
      scalarBytes(t).fold(bind("sz", p.Expr.SizeOf(t)))(n => p.Term.IntU64Const(n.toLong))
    def offsetAt(base: p.Term, idx: Option[p.Term], comp: p.Type): p.Term =
      idx.fold(emitU64(base)) { i =>
        val scaled = bind("om", p.Expr.IntrOp(p.Intr.Mul(emitU64(rwTerm(i)), byteSize(comp), U64)))
        bind("op", p.Expr.IntrOp(p.Intr.Add(emitU64(base), scaled, U64)))
      }

    // bind `(baseTpe)&arena8[(u64)offVal]` and return the base var: a real arena pointer for an offset
    def arenaBase(offVal: p.Term, baseTpe0: p.Type): p.Named = {
      val baseTpe       = globalOuter(baseTpe0)
      val (off, offPre) = toU64(offVal)
      pre ++= offPre
      val ep = p.Named(s"#ae${ctr.incrementAndGet()}", BytePtr)
      pre += p.Stmt.Var(
        ep,
        Some(p.Expr.RefTo(sel(arena8), Some(off), p.Type.IntS8, p.Type.Space.Global, p.Region.Rooted(arena8))),
        isMutable = false
      )
      val b = p.Named(s"#ab${ctr.incrementAndGet()}", baseTpe)
      pre += p.Stmt.Var(b, Some(p.Expr.Cast(sel(ep), baseTpe)), isMutable = false)
      arenaBases += b.symbol
      b
    }

    def rwStep(s: p.PathStep): p.PathStep = s match {
      case p.PathStep.IndexDyn(i) => p.PathStep.IndexDyn(rwTerm(i))
      case x                      => x
    }

    // walk the path, arena-resolving every loaded-pointer deref; a bare value or pure rooted access stays
    def rwTerm(t: p.Term): p.Term = t match {
      case p.Term.Select(root, Nil, _) if offsetRoots(root.symbol) =>
        val n = offsetNamed(root)
        p.Term.Select(n, Nil, n.tpe)
      case sel0 @ p.Term.Select(_, Nil, _) => sel0
      case p.Term.Select(root, steps, resultT) =>
        val rootN = offsetNamed(root)
        val base0 = if (offsetVal(sel(rootN))) arenaBase(sel(rootN), rootN.tpe) else rootN
        val (baseN, accSteps, _) = steps.foldLeft((base0, List.empty[p.PathStep], rootN.tpe)) {
          case ((baseN, accSteps, curTpe), raw) =>
            // a loaded pointer (a pointer field already read, then stepped through) resolves to the arena
            val (b, acc) =
              if (isPtr(curTpe) && accSteps.nonEmpty && offsetVal(p.Term.Select(baseN, accSteps, curTpe)))
                (arenaBase(p.Term.Select(baseN, accSteps, curTpe), curTpe), Nil)
              else (baseN, accSteps)
            val step = rwStep(raw)
            (b, acc :+ step, stepTpe(curTpe, step))
        }
        p.Term.Select(baseN, accSteps, resultT)
      case x => x
    }

    def rwExpr(e: p.Expr): p.Expr = e match {
      case p.Expr.Alias(t)   => p.Expr.Alias(rwTerm(t))
      case p.Expr.Cast(t, a) => p.Expr.Cast(rwTerm(t), a)
      case p.Expr.Index(b, i, comp) =>
        if (isPtr(b.tpe) && offsetVal(b))
          p.Expr.Index(sel(arenaBase(rwTerm(b), p.Type.Ptr(comp, pointeeSpace(b.tpe)))), rwTerm(i), comp)
        else p.Expr.Index(rwTerm(b), rwTerm(i), comp)
      // address-of through an arena offset pointer (`&p[i]`) remains an offset token. emitting a real
      // address here would make the next arena deref add the arena base twice.
      case p.Expr.RefTo(t, idx, comp, p.Type.Space.Global, _) if isPtr(t.tpe) && offsetVal(t) =>
        p.Expr.Cast(offsetAt(rwTerm(t), idx, comp), p.Type.Ptr(comp, p.Type.Space.Global))
      // address-of a non-pointer subobject reached through an arena-relative pointer (e.g. `&result.#base`
      // where result is an offset): the result must stay an offset, else a downstream deref re-adds the
      // arena base. take the real address then subtract the base back to offset space
      case p.Expr.RefTo(t, idx, comp, sp, r) if !isPtr(t.tpe) && arenaLValue(t) =>
        val real = p.Named(s"#ar${ctr.incrementAndGet()}", p.Type.Ptr(comp, sp))
        pre += p.Stmt.Var(
          real,
          Some(p.Expr.RefTo(rwTerm(t), idx.map(rwTerm), comp, sp, p.Region.Opaque)),
          isMutable = false
        )
        val ra = p.Named(s"#aru${ctr.incrementAndGet()}", U64);
        pre += p.Stmt.Var(ra, Some(p.Expr.Cast(sel(real), U64)), isMutable = false)
        val ba = p.Named(s"#arb${ctr.incrementAndGet()}", U64);
        pre += p.Stmt.Var(ba, Some(p.Expr.Cast(sel(arena8), U64)), isMutable = false)
        val off = p.Named(s"#aro${ctr.incrementAndGet()}", U64)
        pre += p.Stmt.Var(off, Some(p.Expr.IntrOp(p.Intr.Sub(sel(ra), sel(ba), U64))), isMutable = false)
        p.Expr.Cast(sel(off), p.Type.Ptr(comp, p.Type.Space.Global))
      case p.Expr.RefTo(t, idx, comp, sp, r)     => p.Expr.RefTo(rwTerm(t), idx.map(rwTerm), comp, sp, r)
      case p.Expr.Alloc(c, sz, sp, r)            => p.Expr.Alloc(c, rwTerm(sz), sp, r)
      case p.Expr.ForeignCall(n, args, rtn)      => p.Expr.ForeignCall(n, args.map(rwTerm), rtn)
      case p.Expr.Invoke(n, ts, recv, args, rtn) => p.Expr.Invoke(n, ts, recv.map(rwTerm), args.map(rwTerm), rtn)
      // an arithmetic/spec op can hold an inline deref operand (e.g. `acc + p->value`); route every term
      // through the arena. there are no IndexDyn steps here (address canonicalisation does not run before
      // ArenaLower) so a Select carries no nested term and is visited once
      case op: p.Expr.IntrOp => op.modifyAll[p.Term](rwTerm)
      case op: p.Expr.MathOp => op.modifyAll[p.Term](rwTerm)
      case p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op, ptr, value, scope, order, rtn)) if offsetVal(ptr) =>
        p.Expr.SpecOp(
          p.Spec.GpuAtomicRMW(op, sel(arenaBase(rwTerm(ptr), ptr.tpe)), rwTerm(value), scope, order, rtn)
        )
      case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(ptr, rtn)) if offsetVal(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuVolatileLoad(sel(arenaBase(rwTerm(ptr), ptr.tpe)), rtn))
      case p.Expr.SpecOp(p.Spec.GpuVolatileStore(ptr, value)) if offsetVal(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuVolatileStore(sel(arenaBase(rwTerm(ptr), ptr.tpe)), rwTerm(value)))
      case op: p.Expr.SpecOp => op.modifyAll[p.Term](rwTerm)
      case x                 => x
    }

    val out = leaf match {
      case p.Stmt.Var(n, Some(e), m) => p.Stmt.Var(offsetNamed(n), Some(rwExpr(e)), m)
      case p.Stmt.Var(n, None, m)    => p.Stmt.Var(offsetNamed(n), None, m)
      case p.Stmt.Mut(t @ p.Term.Select(root, steps, targetTpe), e)
          if steps.nonEmpty && localAggregateLValue(t) && isPtr(targetTpe) && offsetExpr(e) =>
        val token = bind("ot", rwExpr(e))
        p.Stmt.Mut(
          rwTerm(t).asInstanceOf[p.Term.Select],
          p.Expr.Alias(sel(arenaBase(token, e.tpe)))
        )
      case p.Stmt.Mut(t, e) => p.Stmt.Mut(rwTerm(t).asInstanceOf[p.Term.Select], rwExpr(e))
      case p.Stmt.Update(lhs, i, v) =>
        if (isPtr(lhs.tpe) && offsetVal(lhs))
          p.Stmt.Update(sel(arenaBase(rwTerm(lhs), lhs.tpe)), rwTerm(i), rwTerm(v))
        else p.Stmt.Update(rwTerm(lhs).asInstanceOf[p.Term.Select], rwTerm(i), rwTerm(v))
      case p.Stmt.Return(e) => p.Stmt.Return(rwExpr(e))
      case s                => s
    }
    (pre += out).toList
  }
}
