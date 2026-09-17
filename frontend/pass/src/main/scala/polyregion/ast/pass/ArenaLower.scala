package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ListBuffer

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// generic single-arena lowering for flat-address-space binding-slot backends (c_source, opencl1_1, metal).
// the dispatch marshals the whole reachable capture graph into one device arena and binds the arena base as
// the capture arg, so the capture sits at offset 0 and every pointer slot holds an i64 byte offset. rooted
// field reads are unchanged; ONLY a solved arena-relative pointer deref changes - the offset resolves
// through `(T*)&arena8[(u64)off]`. a Select walks step by step, resolving against the arena the moment the
// running value is a pointer LOADED from memory (an already-read pointer field, then stepped through), so a
// chain like vector ref -> records -> _M_p -> chars resolves itself one hop at a time
// examples:
//   cap.x                       ->  cap.x                                       // rooted read, unchanged
//   p[i]   (p arena offset)     ->  ((T*)&arena8[(u64)p])[i]                    // arena-relative deref
//   vec.records._M_p[k]         ->  resolve _M_p hop, then ((T*)&arena8[..])[k] // loaded ptr crossed mid-path
//   &result.#base (result off)  ->  (T*)((u64)&real - (u64)arena8)              // keep the address in offset space
// edge cases:
//   pointer Rooted at a stack local  ->  not arena-resolved (real pointer, left to the backend)
//   arena pointer's outer space      ->  forced Global (conformant OpenCL C rejects a space-changing cast)
//   inline deref in an Intr/Math op  ->  every operand term routed through the arena
object ArenaLower extends ProgramPass {

  private case class ArenaState(
      localAggregates: Set[String],
      analysis: AddressRefinement.Solution
  ) {
    import AddressRefinement.{Encoding, Provenance, Query}

    def root(symbol: String): Option[Encoding] = analysis.bindings.get(symbol).flatMap(_.encoding)

    def fact(root: p.Named): AddressRefinement.AddressValue =
      analysis.bindings.getOrElse(root.symbol, AddressRefinement.AddressValue())

    def localRoot(root: p.Named): Option[(String, List[p.PathStep])] =
      fact(root).provenances
        .collectFirst { case Provenance.Local(symbol, path) if localAggregates(symbol) => symbol -> path }
        .orElse(Option.when(localAggregates(root.symbol))(root.symbol -> Nil))

    def slot(root: p.Named, steps: List[p.PathStep]): Option[(Int, AddressRefinement.AddressValue)] =
      localRoot(root).flatMap { case (symbol, prefix) =>
        val suffix = if (steps.headOption.contains(p.PathStep.Deref)) steps.tail else steps
        val path   = prefix ++ suffix
        analysis.slots.iterator
          .collect {
            case (Query.Slot(`symbol`, candidate), pointer) if AddressRefinement.slotPrefix(candidate, path) =>
              (candidate.size - prefix.size + steps.size - suffix.size) -> pointer
          }
          .toList
          .groupBy(_._1)
          .maxByOption(_._1)
          .map { case (depth, pointers) =>
            depth -> pointers.iterator.map(_._2).reduce(_.join(_))
          }
      }

    def isArenaRooted(term: p.Term): Boolean = {
      val origins = analysis.value(term).provenances ++ (term match {
        case p.Term.Select(root, _, _) => fact(root).provenances
        case _                         => Set.empty
      })
      origins.exists { case Provenance.ArenaRoot(_) => true; case _ => false }
    }

    def needsResolve(target: p.Named, source: p.Expr): Boolean =
      root(target.symbol).contains(Encoding.Absolute) &&
        analysis.result(source).encoding.contains(Encoding.ArenaRelative)

    lazy val offsetRoots: Set[String] = analysis.bindings.iterator.collect {
      case (symbol, pointer) if pointer.encoding.contains(Encoding.ArenaRelative) => symbol
    }.toSet
  }

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

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
  // an arena pointer's data lives in the single global buffer, so its outermost address space is always
  // global; conformant OpenCL C rejects a cast that changes it (nvidia tolerates a stray private outer)
  private def globalOuter(t: p.Type): p.Type = t match {
    case p.Type.Ptr(c, _) => p.Type.Ptr(c, p.Type.Space.Global)
    case _                => t
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val members = program.defs.iterator.map(d => d.name -> d.members).toMap
    program.copy(
      entry = program.entry.map(run(program, members, _)),
      functions = program.functions.map(run(program, members, _))
    )
  }

  private def run(program: p.Program, members: Map[p.Sym, List[p.Named]], f: p.Function): p.Function = captureRoot(
    f
  ) match {
    case None => f
    case Some((capN, _)) =>
      val statements = f.collectAll[p.Stmt]
      val localAggregates = statements.iterator.collect {
        case p.Stmt.Var(n, _, _) if !isPtr(n.tpe) => n.symbol
      }.toSet
      val state     = ArenaState(localAggregates, AddressRefinement.solve(program, f).requireSolved)
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

  private def rwLeaf(
      members: Map[p.Sym, List[p.Named]],
      state: ArenaState,
      arena8: p.Named
  )(leaf: p.Stmt): List[p.Stmt] = {
    import AddressRefinement.{Provenance, Encoding}

    val offsetRoots = state.offsetRoots
    val pre         = ListBuffer.empty[p.Stmt]
    val arenaBases  = scala.collection.mutable.Set.empty[String]

    def arenaLValue(t: p.Term): Boolean = t match {
      case p.Term.Select(root, _, _) =>
        offsetRoots(root.symbol) || state.isArenaRooted(t)
      case _ => false
    }
    def localAggregateLValue(t: p.Term): Boolean = t match {
      case p.Term.Select(root, _, _) => state.localRoot(root).nonEmpty
      case _                         => false
    }
    def offsetTerm(t: p.Term): Boolean = t match {
      case p.Term.Select(root, steps, _) =>
        state.slot(root, steps) match {
          case Some((_, pointer)) if pointer.encoding.contains(Encoding.ArenaRelative) => true
          case Some((depth, pointer))
              if depth < steps.size && pointer.encoding.contains(Encoding.Absolute) && pointer.provenances.exists {
                case Provenance.ArenaRoot(_) => true
                case _                       => false
              } =>
            true
          case Some(_) => false
          case None =>
            state.analysis.value(t).encoding.contains(Encoding.ArenaRelative) ||
            offsetRoots(root.symbol) ||
            // A pointer loaded from an arena-backed field and kept in a local aggregate is a
            // real arena address, but pointer fields reached through it still hold arena offsets.
            // Treat the field load as encoded so the next dereference is rebased exactly once.
            (steps.nonEmpty && canBeArenaOffset(t.tpe) && state.fact(root).hasArenaRoot) ||
            (steps.nonEmpty && canBeArenaOffset(t.tpe) && arenaBases(root.symbol)) ||
            (steps.nonEmpty && canBeArenaOffset(t.tpe) && state.isArenaRooted(t))
        }
      case _ => false
    }
    def offsetExpr(e: p.Expr): Boolean =
      state.analysis.result(e).encoding.contains(Encoding.ArenaRelative)
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
    def resolveAddress(offVal: p.Term, baseTpe0: p.Type): p.Named = {
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
        val base0 = if (offsetVal(sel(rootN))) resolveAddress(sel(rootN), rootN.tpe) else rootN
        val (baseN, accSteps, _) = steps.foldLeft((base0, List.empty[p.PathStep], rootN.tpe)) {
          case ((baseN, accSteps, curTpe), raw) =>
            // a loaded pointer (a pointer field already read, then stepped through) resolves to the arena
            val (b, acc) =
              if (isPtr(curTpe) && accSteps.nonEmpty && offsetVal(p.Term.Select(baseN, accSteps, curTpe)))
                (resolveAddress(p.Term.Select(baseN, accSteps, curTpe), curTpe), Nil)
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
          p.Expr.Index(sel(resolveAddress(rwTerm(b), p.Type.Ptr(comp, pointeeSpace(b.tpe)))), rwTerm(i), comp)
        else p.Expr.Index(rwTerm(b), rwTerm(i), comp)
      // `&p` addresses the local slot which holds an arena-relative pointer value. It is not
      // arithmetic on that value, so preserve the RefTo for the backend instead of converting
      // `p` itself into an arena offset address.
      case p.Expr.RefTo(t @ p.Term.Select(root, Nil, tpe), None, comp, sp, _) if isPtr(tpe) && comp == tpe =>
        p.Expr.RefTo(rwTerm(t), None, comp, sp, p.Region.Rooted(root))
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
          p.Spec.GpuAtomicRMW(op, sel(resolveAddress(rwTerm(ptr), ptr.tpe)), rwTerm(value), scope, order, rtn)
        )
      case p.Expr.SpecOp(p.Spec.GpuAtomicCAS(ptr, expected, desired, scope, order, rtn)) if offsetVal(ptr) =>
        p.Expr.SpecOp(
          p.Spec.GpuAtomicCAS(
            sel(resolveAddress(rwTerm(ptr), ptr.tpe)),
            rwTerm(expected),
            rwTerm(desired),
            scope,
            order,
            rtn
          )
        )
      case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(ptr, rtn)) if offsetVal(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuVolatileLoad(sel(resolveAddress(rwTerm(ptr), ptr.tpe)), rtn))
      case p.Expr.SpecOp(p.Spec.GpuVolatileStore(ptr, value)) if offsetVal(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuVolatileStore(sel(resolveAddress(rwTerm(ptr), ptr.tpe)), rwTerm(value)))
      case op: p.Expr.SpecOp => op.modifyAll[p.Term](rwTerm)
      case x                 => x
    }

    // A local pointer may join a capture-relative offset and an independently bound real pointer.
    // Normalise only offset-producing coercions to real arena addresses; retaining the pointer
    // preserves later mutation and arbitrary indexing through the selected branch.
    def rwAssigned(n: p.Named, e: p.Expr): p.Expr = {
      val rewritten = rwExpr(e)
      if (isPtr(n.tpe) && state.needsResolve(n, e)) {
        val offset = bind("am", rewritten)
        p.Expr.Alias(sel(resolveAddress(offset, n.tpe)))
      } else rewritten
    }

    val out = leaf match {
      case p.Stmt.Var(n, Some(e), m) => p.Stmt.Var(offsetNamed(n), Some(rwAssigned(n, e)), m)
      case p.Stmt.Var(n, None, m)    => p.Stmt.Var(offsetNamed(n), None, m)
      case p.Stmt.Mut(t @ p.Term.Select(root, steps, targetTpe), e)
          if steps.nonEmpty && localAggregateLValue(t) && isPtr(targetTpe) && offsetExpr(e) =>
        // rwExpr may already materialise an encoded field load while walking a pointer chain.
        // Only call resolveAddress for an expression that is still an offset token; otherwise this
        // local aggregate would receive a pointer rebased twice.
        val lowered = rwExpr(e)
        val token   = bind("ot", lowered)
        val value   = p.Expr.Alias(sel(resolveAddress(token, e.tpe)))
        p.Stmt.Mut(
          rwTerm(t).asInstanceOf[p.Term.Select],
          value
        )
      case p.Stmt.Mut(t @ p.Term.Select(root, Nil, _), e) =>
        p.Stmt.Mut(rwTerm(t).asInstanceOf[p.Term.Select], rwAssigned(root, e))
      case p.Stmt.Mut(t, e) => p.Stmt.Mut(rwTerm(t).asInstanceOf[p.Term.Select], rwExpr(e))
      case p.Stmt.Update(lhs, i, v) =>
        if (isPtr(lhs.tpe) && offsetVal(lhs))
          p.Stmt.Update(sel(resolveAddress(rwTerm(lhs), lhs.tpe)), rwTerm(i), rwTerm(v))
        else p.Stmt.Update(rwTerm(lhs).asInstanceOf[p.Term.Select], rwTerm(i), rwTerm(v))
      case p.Stmt.Return(e) => p.Stmt.Return(rwExpr(e))
      case s                => s
    }
    (pre += out).toList
  }
}
