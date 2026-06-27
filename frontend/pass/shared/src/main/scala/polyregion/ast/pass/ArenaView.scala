package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ListBuffer

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// generic single-arena lowering for logical SPIR-V (Vulkan glcompute): no flat address space, no int<->ptr
// cast, no pointer load/store/offset. the capture sits at arena offset 0 and every pointer is an i64 byte
// offset; each deref reads/writes through a fixed roster of typed scalar "view" descriptors indexed by
// `offset / sizeof(elem)`, the only legal access form. pointer struct fields are retyped to i64 so an arena
// object's `_M_p` and a local iterator's `_M_current` are uniform
// examples:
//   cap                         ->  0                                      // capture is arena offset 0
//   cap.x   (scalar field)      ->  view_i32[offsetof(cap, x) / 4]         // read/write via the typed view
//   p[i]    (p Opaque offset)   ->  view_T[(p + i*sizeof T) / sizeof T]    // arena-relative deref
//   it._M_current               ->  the i64 offset directly (field retyped)
//   struct value read           ->  local copy, scalar leaves filled from views (loadAgg)
// edge cases:
//   pointer Rooted at a stack local  ->  stays a real pointer (e.g. inlined std::min over two locals)
//   Float16 field                    ->  own f16 view (numeric Cast can't bitcast; i16 view would convert)
//   ForRange bound / Cond cond       ->  stepped Select hoisted into a Var first (hoistInlineTerms)
//   reduction scratch arg            ->  kept leading, a real workgroup pointer ahead of the views
object ArenaView extends ProgramPass {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private val ctr = new AtomicLong(0L)

  private val Global = p.Type.Space.Global
  // fixed view roster, canonical binding order; the dispatch binds the one arena buffer to all. Float16 needs
  // its own view: polyast Cast is numeric, so reading f16 bits via the i16 view would int-convert. unused
  // views are pruned by the backend
  private val viewTpes: List[p.Type] =
    List(p.Type.IntS8, p.Type.IntS16, p.Type.IntS32, p.Type.IntS64, p.Type.Float32, p.Type.Float64, p.Type.Float16)

  private def isPtr(t: p.Type): Boolean  = t match { case _: p.Type.Ptr => true; case _ => false }
  private def pointee(t: p.Type): p.Type = t match { case p.Type.Ptr(c, _) => c; case _ => t }
  private def elem(t: p.Type): p.Type = t match {
    case p.Type.Ptr(c, _) => c; case p.Type.Arr(c, _, _) => c; case _ => t
  }
  // a pointer (or array-of-pointer) struct field holds an arena byte offset, so retype to i64 (same layout)
  private def i64ify(t: p.Type): p.Type = t match {
    case _: p.Type.Ptr       => I64
    case p.Type.Arr(c, n, s) => p.Type.Arr(i64ify(c), n, s)
    case _                   => t
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    // ORIGINAL member types drive the offset walk (each pointer field's pointee struct); retyping preserves
    // the layout, so emitted OffsetOf resolves the same against the retyped def
    val members = program.defs.iterator.map(d => d.name -> d.members).toMap
    // union: copy only the canonical (largest, head) member
    val unions  = program.defs.iterator.filter(_.isUnion).map(_.name).toSet
    val retyped = program.defs.map(d => d.copy(members = d.members.map(m => m.copy(tpe = i64ify(m.tpe)))))
    program.copy(defs = retyped, entry = run(members, unions, program.entry))
  }

  // lift a stepped Select (the only term shape that can carry an arena access) out of a ForRange bound or
  // Cond condition into a preceding Var, so the leaf rewriter handles it; bare vars and constants stay
  private def hoistInlineTerms(stmts: List[p.Stmt]): List[p.Stmt] = {
    def lift(hint: String, t: p.Term): (List[p.Stmt], p.Term) = t match {
      case p.Term.Select(_, steps, _) if steps.nonEmpty =>
        val n = p.Named(s"#$hint${ctr.incrementAndGet()}", t.tpe);
        (List(p.Stmt.Var(n, Some(p.Expr.Alias(t)), isMutable = false)), sel(n))
      case _ => (Nil, t)
    }
    stmts.flatMap {
      case p.Stmt.ForRange(i, lb, ub, st, body) =>
        val (lbS, lbT) = lift("flb", lb); val (ubS, ubT) = lift("fub", ub); val (stS, stT) = lift("fst", st)
        lbS ::: ubS ::: stS ::: List(p.Stmt.ForRange(i, lbT, ubT, stT, hoistInlineTerms(body)))
      case p.Stmt.Cond(c, t, e) =>
        val (cS, cT) = lift("cnd", c); cS ::: List(p.Stmt.Cond(cT, hoistInlineTerms(t), hoistInlineTerms(e)))
      case p.Stmt.While(c, body)           => List(p.Stmt.While(c, hoistInlineTerms(body)))
      case p.Stmt.Annotated(inner, pos, k) => hoistInlineTerms(List(inner)).map(p.Stmt.Annotated(_, pos, k))
      case s                               => List(s)
    }
  }

  private def run(members: Map[p.Sym, List[p.Named]], unions: Set[p.Sym], f: p.Function): p.Function = captureRoot(
    f
  ) match {
    case None => f
    case Some((capN, capTpe)) =>
      val derived = Provenance.derivedIn(f, arena = true)
      val views   = viewTpes.zipWithIndex.map((t, i) => p.Named(s"#av$i", p.Type.Ptr(t, Global)))

      def arenaRegion(r: p.Region): Boolean = r match {
        case p.Region.Opaque       => true
        case p.Region.Rooted(root) => root == capN
      }
      // a named pointer is an arena offset iff Opaque or Rooted at the capture; the capture itself is the
      // arena root (offset 0). a pointer Rooted at a stack local stays a real pointer
      def isArena(n: p.Named): Boolean = n == capN || derived.get(n).exists(arenaRegion)

      // ForRange bounds / Cond conditions hold terms inline (not in a visited leaf); hoist any stepped Select
      // into a preceding Var. bounds are loop-invariant so hoisting once is sound; While conds are plain vars
      val body =
        mapStmtsRec(hoistInlineTerms(f.body))(
          rewriteLeaf(members, unions, capN, capTpe, views, derived, arenaRegion, isArena)
        )
      // neutralise view binding slots to an i8 view so the slot stays aligned, so we can avoid dragging unused types in
      val usedViews = body.flatMap(_.collectWhere[p.Term] { case p.Term.Select(r, _, _) => r.symbol }).toSet
      val pinnedViews =
        views.map(v => if (usedViews(v.symbol)) v else p.Named(v.symbol, p.Type.Ptr(p.Type.IntS8, Global)))
      // the views replace ONLY the capture; a reduction also has a Local-AS partials/scratch arg (kept,
      // a real workgroup pointer) which must stay first to line up with the dispatch's leading Scratch arg
      val keptArgs    = f.args.filterNot(_.named == capN)
      val newReceiver = if (f.receiver.exists(_.named == capN)) None else f.receiver
      f.copy(
        receiver = newReceiver,
        moduleCaptures = Nil,
        termCaptures = Nil,
        args = keptArgs ++ pinnedViews.map(p.Arg(_)),
        body = body
      )
  }

  private def rewriteLeaf(
      members: Map[p.Sym, List[p.Named]],
      unions: Set[p.Sym],
      capN: p.Named,
      capTpe: p.Type.Struct,
      views: List[p.Named],
      derived: Map[p.Named, p.Region],
      arenaRegion: p.Region => Boolean,
      isArena: p.Named => Boolean
  )(leaf: p.Stmt): List[p.Stmt] = {
    val pre = ListBuffer.empty[p.Stmt]

    def fresh(hint: String, t: p.Type): p.Named = p.Named(s"#$hint${ctr.incrementAndGet()}", t)
    def bind(hint: String, e: p.Expr): p.Term = e match {
      case p.Expr.Alias(t) => t
      case other => val n = fresh(hint, other.tpe); pre += p.Stmt.Var(n, Some(other), isMutable = false); sel(n)
    }
    def i64(v: Long): p.Term     = p.Term.IntS64Const(v)
    def asI64(t: p.Term): p.Term = if (t.tpe == I64) t else bind("ai", p.Expr.Cast(t, I64))
    def add(a: p.Term, b: p.Term): p.Term =
      if (b == i64(0)) a else bind("ao", p.Expr.IntrOp(p.Intr.Add(a, asI64(b), I64)))

    def memberTpe(sym: p.Sym, field: String): p.Type =
      members.get(sym).flatMap(_.find(_.symbol == field).map(_.tpe)).getOrElse(I64)
    // union: copy/read just the canonical (largest, head) member
    def canonicalMembers(sym: p.Sym): List[p.Named] = {
      val ms = members.getOrElse(sym, Nil); if (unions.contains(sym)) ms.take(1) else ms
    }
    def structSym(t: p.Type): Option[p.Sym] = t match { case p.Type.Struct(s, _) => Some(s); case _ => None }
    def arenaTerm(t: p.Term): Boolean       = arenaRegion(Provenance.at(derived, t, arena = true))

    def viewFor(t: p.Type): (p.Named, p.Type, Int) = t match {
      case _: p.Type.Ptr                              => (views(3), p.Type.IntS64, 3)
      case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8 => (views(0), p.Type.IntS8, 0)
      case p.Type.IntU16 | p.Type.IntS16              => (views(1), p.Type.IntS16, 1)
      case p.Type.IntU32 | p.Type.IntS32              => (views(2), p.Type.IntS32, 2)
      case p.Type.Float32                             => (views(4), p.Type.Float32, 2)
      case p.Type.IntU64 | p.Type.IntS64              => (views(3), p.Type.IntS64, 3)
      case p.Type.Float64                             => (views(5), p.Type.Float64, 3)
      case p.Type.Float16                             => (views(6), p.Type.Float16, 1)
      case _                                          => (views(3), p.Type.IntS64, 3)
    }
    def indexOf(off: p.Term, sh: Int): p.Term =
      if (sh == 0) off else bind("ix", p.Expr.IntrOp(p.Intr.BSR(off, i64(sh.toLong), I64)))
    def isAgg(t: p.Type): Boolean = t match {
      case _: p.Type.Struct => true; case _: p.Type.Arr => true; case _ => false
    }
    def loadAt(off: p.Term, t: p.Type): p.Term =
      if (isAgg(t)) loadAgg(off, t)
      else {
        val (v, comp, sh) = viewFor(t)
        val raw           = bind("ld", p.Expr.Index(sel(v), indexOf(off, sh), comp))
        if (t == comp || isPtr(t)) raw else bind("lc", p.Expr.Cast(raw, t))
      }
    // a struct/array read by value cannot go through a scalar view; materialise a local copy, filling each
    // scalar leaf from the arena (pointer fields are i64 offsets in the retyped def, so they copy as i64)
    def loadAgg(off: p.Term, t: p.Type): p.Term = {
      val sv = fresh("sv", t); pre += p.Stmt.Var(sv, None, isMutable = true)
      def fill(prefix: List[p.PathStep], o: p.Term, ft: p.Type): Unit = ft match {
        case s: p.Type.Struct =>
          canonicalMembers(s.name).foreach { m =>
            fill(
              prefix :+ p.PathStep.Field(m.symbol),
              add(o, asI64(bind("of", p.Expr.OffsetOf(ft, m.symbol)))),
              i64ify(m.tpe)
            )
          }
        case p.Type.Arr(elem, n, _) =>
          (0 until n).foreach(e =>
            fill(prefix :+ p.PathStep.Index(e), add(o, mulBytes(i64(e.toLong), elem)), i64ify(elem))
          )
        case scalar => pre += p.Stmt.Mut(p.Term.Select(sv, prefix, scalar), p.Expr.Alias(loadAt(o, scalar)))
      }
      fill(Nil, off, t)
      sel(sv)
    }
    def storeAt(off: p.Term, t: p.Type, value: p.Term): p.Stmt = {
      val (v, comp, sh) = viewFor(t)
      val sv            = if (value.tpe == comp || isPtr(value.tpe)) value else bind("sc", p.Expr.Cast(value, comp))
      p.Stmt.Update(sel(v), indexOf(off, sh), sv)
    }
    // store a struct/array value into the arena scalar-leaf by scalar-leaf (the dual of loadAgg); the source
    // is read field-wise through the normal term rewrite
    def storeAgg(off: p.Term, t: p.Type, src: p.Term): List[p.Stmt] = {
      val srcSel          = src match { case s: p.Term.Select => s; case _ => bindTerm("sv", src) }
      val (sRoot, sSteps) = (srcSel.root, srcSel.steps)
      val out             = ListBuffer.empty[p.Stmt]
      def copy(prefix: List[p.PathStep], o: p.Term, ft: p.Type): Unit = ft match {
        case s: p.Type.Struct =>
          canonicalMembers(s.name)
            .foreach(m =>
              copy(
                prefix :+ p.PathStep.Field(m.symbol),
                add(o, asI64(bind("of", p.Expr.OffsetOf(ft, m.symbol)))),
                i64ify(m.tpe)
              )
            )
        case p.Type.Arr(elem, n, _) =>
          (0 until n).foreach(e =>
            copy(prefix :+ p.PathStep.Index(e), add(o, mulBytes(i64(e.toLong), elem)), i64ify(elem))
          )
        case scalar => out += storeAt(o, scalar, rwTerm(p.Term.Select(sRoot, sSteps ::: prefix, scalar)))
      }
      copy(Nil, off, t)
      out.toList
    }
    def storeVal(off: p.Term, t: p.Type, value: p.Term): List[p.Stmt] =
      if (isAgg(t)) storeAgg(off, t, value) else List(storeAt(off, t, value))
    def byteSize(t: p.Type): p.Term = scalarBytes(t) match {
      case Some(n) => i64(n)
      case None    => asI64(bind("sz", p.Expr.SizeOf(t)))
    }
    def mulBytes(idx: p.Term, comp: p.Type): p.Term =
      if (idx == i64(0)) i64(0) else bind("mo", p.Expr.IntrOp(p.Intr.Mul(asI64(idx), byteSize(comp), I64)))

    def i64Var(n: p.Named): p.Named = p.Named(n.symbol, I64)
    def base(root: p.Named): (p.Term, p.Type) =
      if (root == capN) (i64(0), capTpe) else (sel(i64Var(root)), pointee(root.tpe))

    def rwStep(s: p.PathStep): p.PathStep = s match {
      case p.PathStep.IndexDyn(i) => p.PathStep.IndexDyn(rwTerm(i)); case x: p.PathStep => x
    }
    def bindTerm(hint: String, t: p.Term): p.Term.Select = {
      val n = fresh(hint, t.tpe); pre += p.Stmt.Var(n, Some(p.Expr.Alias(t)), isMutable = false); sel(n)
    }

    // arena byte-offset walk from a base offset + pointee type; a Field/Index on a loaded pointer field
    // auto-derefs it (the `ptr->field` idiom carries no explicit Deref), an explicit Deref does its own load
    def offsetFrom(off0: p.Term, cur0: p.Type, steps: List[p.PathStep]): p.Term = {
      def deref(off: p.Term, cur: p.Type): (p.Term, p.Type) = (loadAt(off, I64), pointee(cur))
      steps
        .foldLeft((off0, cur0)) {
          case ((off, cur), p.PathStep.Field(field)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (
              add(o, asI64(bind("of", p.Expr.OffsetOf(c, field)))),
              structSym(c).fold(c)(s => memberTpe(s, field))
            )
          case ((off, cur), p.PathStep.Deref) => deref(off, cur)
          case ((off, cur), p.PathStep.Index(k)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (add(o, mulBytes(i64(k.toLong), elem(c))), elem(c))
          case ((off, cur), p.PathStep.IndexDyn(idx)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (add(o, mulBytes(rwTerm(idx), elem(c))), elem(c))
        }
        ._1
    }
    def offsetTo(root: p.Named, steps: List[p.PathStep]): p.Term = { val (o, c) = base(root); offsetFrom(o, c, steps) }

    // first pointer field a later step dereferences - the local->arena crossing in a Select rooted at a
    // local (an iterator's `_M_node` read off the stack, then chased in). ORIGINAL member types drive this
    def findCrossing(rootTpe: p.Type, steps: List[p.PathStep]): Option[(List[p.PathStep], p.Type, List[p.PathStep])] = {
      val n = steps.length
      def go(cur: p.Type, i: Int): Option[(List[p.PathStep], p.Type, List[p.PathStep])] =
        if (i >= n) None
        else
          steps(i) match {
            case p.PathStep.Field(f) =>
              val c  = if (isPtr(cur)) pointee(cur) else cur
              val ft = structSym(c).fold(c)(s => memberTpe(s, f))
              if (isPtr(ft) && i < n - 1) Some((steps.take(i + 1), pointee(ft), steps.drop(i + 1)))
              else go(ft, i + 1)
            case p.PathStep.Deref                             => go(pointee(cur), i + 1)
            case p.PathStep.Index(_) | p.PathStep.IndexDyn(_) => go(elem(cur), i + 1)
          }
      go(rootTpe, 0)
    }

    // arena byte offset of the lvalue a Select denotes; None if the whole access stays in local memory
    def lvalueOffset(root: p.Named, steps: List[p.PathStep]): Option[p.Term] =
      if (isArena(root)) Some(offsetTo(root, steps))
      else
        findCrossing(root.tpe, steps).map { case (prefix, pointeeT, suffix) =>
          offsetFrom(bindTerm("lo", p.Term.Select(root, prefix.map(rwStep), I64)), pointeeT, suffix)
        }

    // the i64 offset value a pointer-typed term denotes
    def ptrValue(t: p.Term): p.Term = t match {
      case p.Term.Select(root, Nil, _) => if (root == capN) i64(0) else sel(i64Var(root))
      case p.Term.Select(root, steps, _) =>
        lvalueOffset(root, steps) match {
          case Some(off) => loadAt(off, I64)
          case None      => p.Term.Select(root, steps.map(rwStep), I64) // pure-local pointer field, read directly
        }
      case _ => asI64(t)
    }

    def rwTerm(t: p.Term): p.Term = t match {
      case p.Term.Select(root, Nil, _) if root == capN => i64(0) // the capture itself is arena offset 0
      case p.Term.Select(root, Nil, _) if isArena(root) && isPtr(root.tpe) => sel(i64Var(root))
      case p.Term.Select(root, steps, resultT) if steps.nonEmpty =>
        lvalueOffset(root, steps) match {
          case Some(off) => loadAt(off, if (isPtr(resultT)) I64 else resultT)
          case None      => p.Term.Select(root, steps.map(rwStep), if (isPtr(resultT) && arenaTerm(t)) I64 else resultT)
        }
      case x => x
    }

    // i64 base offset for an indexed arena access (Some), else None to keep a real local pointer: a pointer
    // base is loaded (its value is the offset), an array base IS the offset (its lvalue location)
    def derefOffset(base: p.Term): Option[p.Term] =
      if (!arenaTerm(base)) None
      else if (isPtr(base.tpe)) Some(ptrValue(base))
      else Some(addrOffset(base))
    // offset of an arena data lvalue whose address is taken (`&obj.field`, field non-pointer)
    def addrOffset(base: p.Term): p.Term = base match {
      case p.Term.Select(root, steps, _) => lvalueOffset(root, steps).getOrElse(asI64(rwTerm(base)))
      case _                             => asI64(rwTerm(base))
    }

    def rwExpr(e: p.Expr): p.Expr = e match {
      case p.Expr.Alias(t) => p.Expr.Alias(rwTerm(t))
      case p.Expr.Cast(from, as) if isPtr(from.tpe) && arenaTerm(from) =>
        val v = ptrValue(from)
        if (isPtr(as) || as == I64) p.Expr.Alias(v) else p.Expr.Cast(v, as)
      case p.Expr.Cast(from, as) => p.Expr.Cast(rwTerm(from), as)
      case p.Expr.RefTo(base, idx, comp, sp, r) if arenaTerm(base) =>
        val off0 = if (isPtr(base.tpe)) ptrValue(base) else addrOffset(base)
        p.Expr.Alias(add(off0, idx.fold(i64(0))(i => mulBytes(rwTerm(i), comp))))
      case p.Expr.RefTo(base, idx, comp, sp, r) => p.Expr.RefTo(rwTerm(base), idx.map(rwTerm), comp, sp, r)
      case p.Expr.Index(base, idx, comp) =>
        derefOffset(base) match {
          case Some(off0) => p.Expr.Alias(loadAt(add(off0, mulBytes(rwTerm(idx), comp)), comp))
          case None       => p.Expr.Index(rwTerm(base), rwTerm(idx), comp)
        }
      case p.Expr.Alloc(c, sz, sp, r)            => p.Expr.Alloc(c, rwTerm(sz), sp, r)
      case p.Expr.ForeignCall(n, args, rtn)      => p.Expr.ForeignCall(n, args.map(rwTerm), rtn)
      case p.Expr.Invoke(n, ts, recv, args, rtn) => p.Expr.Invoke(n, ts, recv.map(rwTerm), args.map(rwTerm), rtn)
      case op: p.Expr.IntrOp                     => op.modifyAll[p.Term](rwTerm)
      case op: p.Expr.MathOp                     => op.modifyAll[p.Term](rwTerm)
      case op: p.Expr.SpecOp                     => op.modifyAll[p.Term](rwTerm)
      case x                                     => x
    }

    def rwInit(n: p.Named, e: p.Expr): (p.Named, p.Expr) =
      if (isArena(n) && isPtr(n.tpe)) {
        val nn = i64Var(n)
        e match { case p.Expr.Alias(_: p.Term.NullPtrConst) => (nn, p.Expr.Alias(i64(0))); case _ => (nn, rwExpr(e)) }
      } else (n, rwExpr(e))

    val out = leaf match {
      case p.Stmt.Var(n, Some(e), m) => val (nn, ne) = rwInit(n, e); List(p.Stmt.Var(nn, Some(ne), m))
      case p.Stmt.Var(n, None, m)    => List(p.Stmt.Var(if (isArena(n) && isPtr(n.tpe)) i64Var(n) else n, None, m))
      case p.Stmt.Mut(p.Term.Select(n, Nil, t), e) =>
        if (isArena(n) && isPtr(n.tpe)) List(p.Stmt.Mut(p.Term.Select(i64Var(n), Nil, I64), rwExpr(e)))
        else List(p.Stmt.Mut(p.Term.Select(n, Nil, t), rwExpr(e)))
      case p.Stmt.Mut(p.Term.Select(n, steps, scalarT), e) if isArena(n) =>
        storeVal(offsetTo(n, steps), scalarT, bind("st", rwExpr(e)))
      case p.Stmt.Mut(p.Term.Select(n, steps, scalarT), e) =>
        // local struct field write; a pointer field is now i64
        val lhsT = if (isPtr(scalarT) && arenaTerm(p.Term.Select(n, steps, scalarT))) I64 else scalarT
        List(p.Stmt.Mut(p.Term.Select(n, steps.map(rwStep), lhsT), rwExpr(e)))
      case p.Stmt.Update(base @ p.Term.Select(_, _, ptrT), idx, v) =>
        derefOffset(base) match {
          case Some(off0) => storeVal(add(off0, mulBytes(rwTerm(idx), elem(ptrT))), elem(ptrT), rwTerm(v))
          case None       => List(p.Stmt.Update(rwTerm(base).asInstanceOf[p.Term.Select], rwTerm(idx), rwTerm(v)))
        }
      case p.Stmt.Return(e) => List(p.Stmt.Return(rwExpr(e)))
      case s                => List(s)
    }
    (pre ++= out).toList
  }
}
