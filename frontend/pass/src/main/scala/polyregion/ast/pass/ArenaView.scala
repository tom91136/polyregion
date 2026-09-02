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
//   local `p = &x; s.ptr = p`   ->  stable i64 identity token when s.ptr is only stored/compared
// edge cases:
//   pointer Rooted at a stack local  ->  stays a real pointer (e.g. inlined std::min over two locals)
//   local pointer field dereference  ->  not tokenised; only identity-only fields are eligible
//   Float16 field                    ->  own f16 view (numeric Cast can't bitcast; i16 view would convert)
//   ForRange bound / Cond cond       ->  stepped Select hoisted into a Var first (hoistInlineTerms)
//   reduction scratch arg            ->  kept leading, a real workgroup pointer ahead of the views
object ArenaView extends ProgramPass {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private val ctr = new AtomicLong(0L)

  private val Global = p.Type.Space.Global
  // fixed view roster, canonical binding order; the dispatch binds the one arena buffer to all. Float16 needs
  // its own view: polyast Cast is numeric, so reading f16 bits via the i16 view would int-convert. unused
  // views are pruned by the backend
  private val viewTpes: List[p.Type] =
    List(p.Type.IntS8, p.Type.IntS16, p.Type.IntS32, p.Type.IntS64, p.Type.Float32, p.Type.Float64, p.Type.Float16)

  private def isPtr(t: p.Type): Boolean  = t match { case _: p.Type.Ptr => true; case _ => false }
  private def isArr(t: p.Type): Boolean  = t match { case _: p.Type.Arr => true; case _ => false }
  private def pointee(t: p.Type): p.Type = t match { case p.Type.Ptr(c, _) => c; case _ => t }
  private def elem(t: p.Type): p.Type = t match {
    case p.Type.Ptr(c, _) => c; case p.Type.Arr(c, _, _) => c; case _ => t
  }
  // a Global pointer (or array-of-pointer) struct field holds an arena byte offset, so retype to i64
  // (same layout). Private pointer fields belong to stack-local aggregates and remain real pointers.
  private def i64ify(t: p.Type): p.Type = t match {
    case p.Type.Ptr(_, p.Type.Space.Global) => I64
    case p.Type.Ptr(c, s)                   => p.Type.Ptr(i64ify(c), s)
    case p.Type.Arr(c, n, s)                => p.Type.Arr(i64ify(c), n, s)
    case _                                  => t
  }

  private type Field = (p.Sym, String)

  private def fieldsAt(
      rootTpe: p.Type,
      steps: List[p.PathStep],
      members: Map[p.Sym, List[p.Named]]
  ): List[(Field, p.Type)] = {
    def member(sym: p.Sym, field: String): Option[p.Type] =
      members.get(sym).flatMap(_.find(_.symbol == field).map(_.tpe))
    steps
      .foldLeft((rootTpe, List.empty[(Field, p.Type)])) {
        case ((cur, found), p.PathStep.Field(field)) =>
          val owner = pointee(cur) match { case p.Type.Struct(sym, _) => Some(sym); case _ => None }
          owner.flatMap(sym => member(sym, field).map(t => (t, (sym -> field, t)))) match {
            case Some((t, resolved)) => (t, found :+ resolved)
            case None                => (cur, found)
          }
        case ((cur, found), p.PathStep.Deref)       => (pointee(cur), found)
        case ((cur, found), p.PathStep.Index(_))    => (elem(cur), found)
        case ((cur, found), p.PathStep.IndexDyn(_)) => (elem(cur), found)
      }
      ._2
  }

  private def fieldAt(rootTpe: p.Type, steps: List[p.PathStep], members: Map[p.Sym, List[p.Named]]): Option[Field] =
    fieldsAt(rootTpe, steps, members).lastOption.map(_._1)

  private def staticIndex(index: Option[p.Term]): Option[String] = index match {
    case None                        => Some("")
    case Some(p.Term.IntU8Const(v))  => Some(java.lang.Byte.toUnsignedInt(v).toString)
    case Some(p.Term.IntU16Const(v)) => Some(v.toInt.toString)
    case Some(p.Term.IntU32Const(v)) => Some(java.lang.Integer.toUnsignedLong(v).toString)
    case Some(p.Term.IntU64Const(v)) => Some(java.lang.Long.toUnsignedString(v))
    case Some(p.Term.IntS8Const(v))  => Some(v.toString)
    case Some(p.Term.IntS16Const(v)) => Some(v.toString)
    case Some(p.Term.IntS32Const(v)) => Some(v.toString)
    case Some(p.Term.IntS64Const(v)) => Some(v.toString)
    case _                           => None
  }

  private def staticPathKey(steps: List[p.PathStep]): Option[String] =
    if (steps.exists(_.isInstanceOf[p.PathStep.IndexDyn])) None
    else
      Some(
        steps
          .map {
            case p.PathStep.Field(name) => s"field:$name"
            case p.PathStep.Deref       => "deref"
            case p.PathStep.Index(idx)  => s"index:$idx"
            case _: p.PathStep.IndexDyn => throw IllegalStateException("dynamic step in static arena path")
          }
          .mkString("/")
      )

  private def localReferenceKey(base: p.Term.Select, index: Option[p.Term]): Option[String] =
    for {
      path <- staticPathKey(base.steps)
      idx  <- staticIndex(index)
    } yield s"${base.root.symbol}:$path:$idx"

  private def arenaStructs(program: p.Program, members: Map[p.Sym, List[p.Named]]): Set[p.Sym] = {
    def refs(t: p.Type): Set[p.Sym] = t match {
      case p.Type.Struct(sym, args) => Set(sym) ++ args.flatMap(refs)
      case p.Type.Ptr(c, _)         => refs(c)
      case p.Type.Arr(c, _, _)      => refs(c)
      case p.Type.Exec(_, as, r)    => as.flatMap(refs).toSet ++ refs(r)
      case _                        => Set.empty
    }
    val roots = program.entry.flatMap(captureRoot).map(_._2.name).toSet
    doUntilNotEq(roots) { (_, seen) =>
      seen ++ seen.flatMap(sym => members.getOrElse(sym, Nil).flatMap(m => refs(m.tpe)))
    }._2
  }

  private def localIdentityFields(
      program: p.Program,
      members: Map[p.Sym, List[p.Named]],
      arenaDefs: Set[p.Sym]
  ): (Set[Field], Map[p.Named, (Field, p.Term)]) = {
    val entry   = program.entry.getOrElse(throw IllegalArgumentException("ArenaView requires a program entry"))
    val derived = Provenance.derivedIn(entry, arena = true)
    val cap     = captureRoot(entry).map(_._1)
    def isLocal(t: p.Term): Boolean = Provenance.at(derived, t, arena = true) match {
      case p.Region.Rooted(root) if !cap.contains(root) =>
        root.tpe match {
          case p.Type.Ptr(_, p.Type.Space.Private | p.Type.Space.Local) => true
          case _: p.Type.Ptr                                            => false
          case _                                                        => true
        }
      case _ => false
    }
    val reassignedPointers = program.entry
      .collectAll[p.Stmt]
      .collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _: p.Type.Ptr), _) =>
        n
      }
      .toSet
    val stableLocalPointers = program.entry.collectAll[p.Stmt].foldLeft(Set.empty[p.Named]) {
      case (known, p.Stmt.Var(n, Some(p.Expr.RefTo(base @ p.Term.Select(_, steps, _), idx, _, _, _)), _))
          if isPtr(n.tpe) && !reassignedPointers(n) && isLocal(base) && staticIndex(idx).nonEmpty && !steps.exists {
            case _: p.PathStep.IndexDyn => true
            case _                      => false
          } =>
        known + n
      case (known, p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), _))
          if isPtr(n.tpe) && !reassignedPointers(n) && !reassignedPointers(root) && known(root) =>
        known + n
      case (known, p.Stmt.Var(n, Some(p.Expr.Cast(source @ p.Term.Select(root, Nil, _), _: p.Type.Ptr)), _))
          if isPtr(n.tpe) && !reassignedPointers(n) && !reassignedPointers(root) &&
            (known(root) || (isArr(root.tpe) && isLocal(source))) =>
        known + n
      case (known, _) => known
    }
    def stableLocal(t: p.Term): Boolean = t match {
      case _: p.Term.NullPtrConst      => true
      case p.Term.Select(root, Nil, _) => stableLocalPointers(root)
      case _                           => false
    }
    def localIdentity(t: p.Term): Boolean = t match {
      case _: p.Term.NullPtrConst => true
      // Stable locals are seeded only from locally rooted RefTo expressions or casts from local arrays.
      // Inlining can conservatively widen an alias's provenance to Opaque, so do not discard that stronger
      // construction proof when classifying an identity-only field write.
      case p.Term.Select(_, Nil, _) =>
        isLocal(t) || Provenance.at(derived, t, arena = true) == p.Region.Opaque || stableLocal(t)
      case _ => stableLocal(t)
    }
    def directSelectedField(t: p.Term): Option[Field] = t match {
      case p.Term.Select(root, steps, _: p.Type.Ptr) if steps.nonEmpty =>
        fieldsAt(root.tpe, steps, members).lastOption.collect { case (field, p.Type.Ptr(_, p.Type.Space.Global)) =>
          field
        }
      case _ => None
    }
    // FullOpt materialises pointer-field reads into immutable SSA aliases before ArenaView. Retain
    // the field identity through those aliases so a field used only for pointer equality is still
    // represented by a logical token rather than an illegal pointer value.
    val fieldAliases = doUntilNotEq(Map.empty[p.Named, (Field, p.Term)]) { (_, known) =>
      val discovered = program.entry
        .collectAll[p.Stmt]
        .collect { case p.Stmt.Var(n, Some(p.Expr.Alias(source)), false) =>
          directSelectedField(source)
            .map(field => n -> (field, source))
            .orElse(source match {
              case p.Term.Select(root, Nil, _) =>
                known.get(root).map { case (field, original) => n -> (field, original) }
              case _ => None
            })
        }
        .flatten
        .toMap
      known ++ discovered
    }._2
    def selectedField(t: p.Term): Option[Field] =
      directSelectedField(t).orElse {
        t match {
          case p.Term.Select(root, Nil, _: p.Type.Ptr) => fieldAliases.get(root).map(_._1)
          case _                                       => None
        }
      }
    // Logical SPIR-V cannot store a local pointer in an aggregate, but a field used only for identity can retain
    // C++ equality semantics as an i64 token. Arena-reachable or otherwise-observed fields keep normal lowering.
    // A write source is either an already-tokenisable local value or another identity field.
    def identityWriteSource(expr: p.Expr): Option[Either[Field, Boolean]] = expr match {
      case p.Expr.Alias(rhs)               => Some(selectedField(rhs).toLeft(localIdentity(rhs)))
      case p.Expr.Cast(rhs, _: p.Type.Ptr) => Some(selectedField(rhs).toLeft(localIdentity(rhs)))
      case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) =>
        Some(Right(isLocal(base) && localReferenceKey(base, index).nonEmpty))
      case _ => None
    }
    val writes = program.entry
      .collectAll[p.Stmt]
      .collect { case p.Stmt.Mut(target: p.Term.Select, expr) =>
        selectedField(target).flatMap(field => identityWriteSource(expr).map(field -> _))
      }
      .flatten
    def traversedPointerFields(t: p.Term): List[Field] = t match {
      case p.Term.Select(root, steps, _) if steps.nonEmpty =>
        fieldsAt(root.tpe, steps, members).collect { case (field, _: p.Type.Ptr) => field }
      case _ => Nil
    }
    def tally(xs: List[Field]): Map[Field, Int] = xs.groupMapReduce(identity)(_ => 1)(_ + _)
    val totalUses = tally(program.entry.collectAll[p.Term].flatMap(traversedPointerFields))
    val identityUses = tally(
      program.entry.collectAll[p.Stmt].flatMap {
        case p.Stmt.Mut(t: p.Term.Select, p.Expr.Alias(rhs)) => selectedField(t).toList ++ selectedField(rhs).toList
        case p.Stmt.Mut(t: p.Term.Select, _)                 => selectedField(t).toList
        case _                                               => Nil
      } ::: program.entry.collectAll[p.Expr].flatMap {
        case p.Expr.IntrOp(p.Intr.LogicEq(x, y))  => List(x, y).flatMap(selectedField)
        case p.Expr.IntrOp(p.Intr.LogicNeq(x, y)) => List(x, y).flatMap(selectedField)
        case _                                    => Nil
      }
    )
    val writesByField = writes.groupMap(_._1)(_._2)
    val candidates = doUntilNotEq(Set.empty[Field]) { (_, known) =>
      writesByField.collect {
        case (field @ (owner, _), sources)
            if !arenaDefs(owner) && sources.nonEmpty && totalUses.get(field) == identityUses.get(field) &&
              sources.forall {
                case Right(ok)  => ok
                case Left(from) => from == field || known(from)
              } && sources.exists {
                case Right(ok)  => ok
                case Left(from) => from != field && known(from)
              } =>
          field
      }.toSet
    }._2
    def comparable(t: p.Term): Boolean = t match {
      case _: p.Term.NullPtrConst                                   => true
      case p.Term.Select(root, Nil, _) if stableLocalPointers(root) => true
      case selected if selectedField(selected).exists(candidates)   => true
      case _                                                        => false
    }
    val unsupported = program.entry
      .collectAll[p.Expr]
      .flatMap {
        case p.Expr.IntrOp(p.Intr.LogicEq(x, y))  => List((x, y), (y, x))
        case p.Expr.IntrOp(p.Intr.LogicNeq(x, y)) => List((x, y), (y, x))
        case _                                    => Nil
      }
      .collect {
        case (selected, other) if selectedField(selected).exists(candidates) && !comparable(other) =>
          selectedField(selected).get
      }
      .toSet
    (candidates -- unsupported, fieldAliases)
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    // ORIGINAL member types drive the offset walk (each pointer field's pointee struct); retyping preserves
    // the layout, so emitted OffsetOf resolves the same against the retyped def
    val members                        = program.defs.iterator.map(d => d.name -> d.members).toMap
    val arenaDefs                      = arenaStructs(program, members)
    val (identityFields, fieldAliases) = localIdentityFields(program, members, arenaDefs)
    // union: copy only the canonical (largest, head) member
    val unions  = program.defs.iterator.filter(_.isUnion).map(_.name).toSet
    val retyped = program.defs.map(d => d.copy(members = d.members.map(m => m.copy(tpe = i64ify(m.tpe)))))
    val entry   = program.entry.getOrElse(throw IllegalArgumentException("ArenaView requires a program entry"))
    program.copy(defs = retyped, entry = Some(run(members, unions, identityFields, fieldAliases, entry)))
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

  private def run(
      members: Map[p.Sym, List[p.Named]],
      unions: Set[p.Sym],
      identityFields: Set[Field],
      fieldAliases: Map[p.Named, (Field, p.Term)],
      f: p.Function
  ): p.Function = captureRoot(
    f
  ) match {
    case None => f
    case Some((capN, capTpe)) =>
      val derived = Provenance.derivedIn(f, arena = true)
      val views   = viewTpes.zipWithIndex.map((t, i) => p.Named(s"#av$i", p.Type.Ptr(t, Global)))

      def rootedLocally(t: p.Term): Boolean = Provenance.at(derived, t, arena = true) match {
        case p.Region.Rooted(root) if root != capN =>
          root.tpe match {
            case p.Type.Ptr(_, p.Type.Space.Private | p.Type.Space.Local) => true
            case _: p.Type.Ptr                                            => false
            case _                                                        => true
          }
        case _ => false
      }
      val reassignedPointers = f
        .collectAll[p.Stmt]
        .collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _: p.Type.Ptr), _) =>
          n
        }
        .toSet
      val localPointerKeys = f.collectAll[p.Stmt].foldLeft(Map.empty[p.Named, String]) {
        case (known, p.Stmt.Var(n, Some(p.Expr.RefTo(base @ p.Term.Select(_, steps, _), idx, _, _, _)), _))
            if isPtr(n.tpe) && !reassignedPointers(n) && rootedLocally(base) && staticIndex(idx).nonEmpty && !steps
              .exists {
                case _: p.PathStep.IndexDyn => true
                case _                      => false
              } =>
          known + (n -> localReferenceKey(base, idx).get)
        case (known, p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), _))
            if isPtr(n.tpe) && !reassignedPointers(n) && !reassignedPointers(root) && known.contains(root) =>
          known + (n -> known(root))
        case (known, p.Stmt.Var(n, Some(p.Expr.Cast(source @ p.Term.Select(root, Nil, _), _: p.Type.Ptr)), _))
            if isPtr(n.tpe) && !reassignedPointers(n) && !reassignedPointers(root) &&
              (known.contains(root) || (isArr(root.tpe) && rootedLocally(source))) =>
          known + (n -> known.getOrElse(root, s"${root.symbol}:array"))
        case (known, _) => known
      }
      val directLocalPointerKeys = f
        .collectAll[p.Expr]
        .collect {
          case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) if rootedLocally(base) =>
            localReferenceKey(base, index)
        }
        .flatten
        .toSet
      val tokenByKey = (localPointerKeys.values.toSet ++ directLocalPointerKeys).toList.sorted.zipWithIndex.map {
        case (key, i) =>
          key -> (i.toLong + 1L)
      }.toMap
      val localPointerTokens = localPointerKeys.view.mapValues(tokenByKey).toMap

      // Inlined nullable base-pointer adjustments retain their source-level null guard after their actual argument
      // becomes either an immutable RefTo of stack storage or an immutable null binding. Keeping those guards
      // creates Function-pointer phis which logical SPIR-V cannot represent and some physical SPIR-V runtimes
      // miscompile for non-zero multiple-inheritance base offsets. Fold only guards with stable local proofs.
      def definitelyNonNullLocal(t: p.Term): Boolean = t match {
        case p.Term.Select(root, Nil, _: p.Type.Ptr) => !reassignedPointers(root) && rootedLocally(t)
        case _                                       => false
      }
      val stableNullPointers = f.collectAll[p.Stmt].foldLeft(Set.empty[p.Named]) {
        case (known, p.Stmt.Var(n, Some(p.Expr.Alias(_: p.Term.NullPtrConst)), _))
            if isPtr(n.tpe) && !reassignedPointers(n) =>
          known + n
        case (known, p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), _))
            if isPtr(n.tpe) && !reassignedPointers(n) && known(root) =>
          known + n
        case (known, _) => known
      }
      def definitelyNull(t: p.Term): Boolean = t match {
        case _: p.Term.NullPtrConst      => true
        case p.Term.Select(root, Nil, _) => stableNullPointers(root)
        case _                           => false
      }
      val constantConditions = f
        .collectAll[p.Stmt]
        .collect {
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(x, _: p.Term.NullPtrConst))), false)
              if definitelyNonNullLocal(x) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(_: p.Term.NullPtrConst, y))), false)
              if definitelyNonNullLocal(y) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicEq(x, y))), false)
              if definitelyNull(x) && definitelyNull(y) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(x, y))), false)
              if definitelyNull(x) && definitelyNull(y) =>
            n -> false
        }
        .toMap
      def simplifyStablePointerGuards(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
        case p.Stmt.Cond(p.Term.Select(root, Nil, _), whenTrue, whenFalse) if constantConditions.contains(root) =>
          simplifyStablePointerGuards(if (constantConditions(root)) whenTrue else whenFalse)
        case p.Stmt.Cond(c, whenTrue, whenFalse) =>
          List(p.Stmt.Cond(c, simplifyStablePointerGuards(whenTrue), simplifyStablePointerGuards(whenFalse)))
        case p.Stmt.While(c, body) => List(p.Stmt.While(c, simplifyStablePointerGuards(body)))
        case p.Stmt.ForRange(i, lb, ub, step, body) =>
          List(p.Stmt.ForRange(i, lb, ub, step, simplifyStablePointerGuards(body)))
        case t: p.Stmt.Try => List(t.mapBlocks(simplifyStablePointerGuards))
        case p.Stmt.Raise(value, exceptionKind, cleanup) =>
          List(p.Stmt.Raise(value, exceptionKind, simplifyStablePointerGuards(cleanup)))
        case p.Stmt.Annotated(inner, pos, comment) =>
          simplifyStablePointerGuards(List(inner)).map(p.Stmt.Annotated(_, pos, comment))
        case stmt => List(stmt)
      }

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
        mapStmtsRec(hoistInlineTerms(simplifyStablePointerGuards(f.body)))(
          rewriteLeaf(
            members,
            unions,
            identityFields,
            fieldAliases,
            localPointerTokens,
            tokenByKey,
            capN,
            capTpe,
            views,
            derived,
            arenaRegion,
            isArena,
            f.collectAll[p.Stmt]
              .collect {
                case p.Stmt.Mut(p.Term.Select(root, _, _), _)       => root
                case p.Stmt.Update(p.Term.Select(root, _, _), _, _) => root
              }
              .toSet
          )
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
        decl = f.decl
          .remapArgs(keptArgs ++ pinnedViews.map(p.Arg(_)))
          .copy(
            receiver = newReceiver,
            moduleCaptures = Nil,
            termCaptures = Nil
          ),
        body = body
      )
  }

  private def rewriteLeaf(
      members: Map[p.Sym, List[p.Named]],
      unions: Set[p.Sym],
      identityFields: Set[Field],
      fieldAliases: Map[p.Named, (Field, p.Term)],
      localPointerTokens: Map[p.Named, Long],
      localReferenceTokens: Map[String, Long],
      capN: p.Named,
      capTpe: p.Type.Struct,
      views: List[p.Named],
      derived: Map[p.Named, p.Region],
      arenaRegion: p.Region => Boolean,
      isArena: p.Named => Boolean,
      mutatedRoots: Set[p.Named]
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
    def isIdentityField(root: p.Named, steps: List[p.PathStep]): Boolean =
      fieldAt(root.tpe, steps, members).exists(identityFields)
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

    // Physical SPIR-V still uses ArenaView's typed scalar descriptors, but an immutable array
    // binding does not need a private aggregate copy.  Keep a pointer to the selected first
    // element and expose it through a dereferenced pointer-to-array binding.  The LLVM backend
    // can then bind the array name directly to that storage.  Restrict this to scalar arrays and
    // untouched bindings: aggregate elements need the ordinary field-wise materialisation, and a
    // write through a by-value binding must not unexpectedly alias the arena object.
    def arrayAlias(n: p.Named, e: p.Expr, isMutable: Boolean): Option[p.Stmt] =
      if (isMutable) None
      else
        (n.tpe, e) match {
          case (arr @ p.Type.Arr(component, length, _), p.Expr.Alias(source: p.Term.Select))
              if length > 0 && !isAgg(component) && source.steps.nonEmpty && lvalueOffset(
                source.root,
                source.steps
              ).nonEmpty =>
            if (mutatedRoots(n)) None
            else {
              val off              = lvalueOffset(source.root, source.steps).get
              val (view, _, shift) = viewFor(component)
              val elemPtrTpe       = p.Type.Ptr(component, Global)
              val elemPtr          = fresh("av", elemPtrTpe)
              val ptrTpe           = p.Type.Ptr(arr, Global)
              val ptr              = fresh("av", ptrTpe)
              val index            = indexOf(off, shift)
              val ref              = p.Expr.RefTo(sel(view), Some(index), component, Global, p.Region.Rooted(view))
              pre += p.Stmt.Var(elemPtr, Some(ref), isMutable = false)
              pre += p.Stmt.Var(ptr, Some(p.Expr.Cast(sel(elemPtr), ptrTpe)), isMutable = false)
              Some(
                p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(ptr, List(p.PathStep.Deref), arr))), isMutable = false)
              )
            }
          case _ => None
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
          case None =>
            p.Term.Select(root, steps.map(rwStep), if (isPtr(resultT) && arenaTerm(t)) I64 else i64ify(resultT))
        }
      case x => x
    }

    // i64 base offset for an indexed arena access (Some), else None to keep a real local pointer: a pointer
    // base is loaded (its value is the offset), an array base IS the offset (its lvalue location)
    def derefOffset(base: p.Term): Option[p.Term] =
      if (!arenaTerm(base)) None
      else if (isPtr(base.tpe)) Some(ptrValue(base))
      else Some(addrOffset(base))
    def scalarRefAt(off: p.Term, tpe: p.Type): p.Term = {
      val (view, _, sh) = viewFor(tpe)
      bind(
        "vr",
        p.Expr.RefTo(
          sel(view),
          Some(indexOf(off, sh)),
          tpe,
          Global,
          p.Region.Rooted(view)
        )
      )
    }
    def arenaScalarRef(ptr: p.Term, tpe: p.Type): p.Term = {
      if (isAgg(tpe))
        throw IllegalArgumentException(s"arena atomic access requires a scalar type; got ${tpe.repr}")
      scalarRefAt(
        derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
        tpe
      )
    }
    def volatileLoadAt(off: p.Term, tpe: p.Type): p.Term =
      if (!isAgg(tpe)) bind("vl", p.Expr.SpecOp(p.Spec.GpuVolatileLoad(scalarRefAt(off, tpe), tpe)))
      else {
        val value = fresh("vv", tpe)
        pre += p.Stmt.Var(value, None, isMutable = true)
        def load(prefix: List[p.PathStep], at: p.Term, fieldTpe: p.Type): Unit = fieldTpe match {
          case struct: p.Type.Struct =>
            canonicalMembers(struct.name).foreach { member =>
              load(
                prefix :+ p.PathStep.Field(member.symbol),
                add(at, asI64(bind("of", p.Expr.OffsetOf(fieldTpe, member.symbol)))),
                i64ify(member.tpe)
              )
            }
          case p.Type.Arr(component, size, _) =>
            (0 until size).foreach(index =>
              load(
                prefix :+ p.PathStep.Index(index),
                add(at, mulBytes(i64(index.toLong), component)),
                i64ify(component)
              )
            )
          case scalar =>
            pre += p.Stmt.Mut(
              p.Term.Select(value, prefix, scalar),
              p.Expr.Alias(bind("vl", p.Expr.SpecOp(p.Spec.GpuVolatileLoad(scalarRefAt(at, scalar), scalar))))
            )
        }
        load(Nil, off, tpe)
        sel(value)
      }
    def volatileStoreAt(off: p.Term, tpe: p.Type, value: p.Term): Unit = {
      val source          = value match { case s: p.Term.Select => s; case _ => bindTerm("vs", value) }
      val (root, initial) = (source.root, source.steps)
      def store(prefix: List[p.PathStep], at: p.Term, fieldTpe: p.Type): Unit = fieldTpe match {
        case struct: p.Type.Struct =>
          canonicalMembers(struct.name).foreach { member =>
            store(
              prefix :+ p.PathStep.Field(member.symbol),
              add(at, asI64(bind("of", p.Expr.OffsetOf(fieldTpe, member.symbol)))),
              i64ify(member.tpe)
            )
          }
        case p.Type.Arr(component, size, _) =>
          (0 until size).foreach(index =>
            store(
              prefix :+ p.PathStep.Index(index),
              add(at, mulBytes(i64(index.toLong), component)),
              i64ify(component)
            )
          )
        case scalar =>
          val done = fresh("vs", p.Type.Unit0)
          pre += p.Stmt.Var(
            done,
            Some(
              p.Expr.SpecOp(
                p.Spec.GpuVolatileStore(
                  scalarRefAt(at, scalar),
                  rwTerm(p.Term.Select(root, initial ::: prefix, scalar))
                )
              )
            ),
            isMutable = false
          )
      }
      store(Nil, off, tpe)
    }
    // offset of an arena data lvalue whose address is taken (`&obj.field`, field non-pointer)
    def addrOffset(base: p.Term): p.Term = base match {
      case p.Term.Select(root, steps, _) => lvalueOffset(root, steps).getOrElse(asI64(rwTerm(base)))
      case _                             => asI64(rwTerm(base))
    }
    def selectedIdentityField(t: p.Term): Boolean = t match {
      case p.Term.Select(root, steps, _: p.Type.Ptr) if steps.nonEmpty => isIdentityField(root, steps)
      case p.Term.Select(root, Nil, _: p.Type.Ptr) => fieldAliases.get(root).exists(x => identityFields(x._1))
      case _                                       => false
    }
    def identityComparable(t: p.Term): Option[p.Term] = t match {
      case _: p.Term.NullPtrConst                                           => Some(i64(0))
      case p.Term.Select(root, Nil, _) if localPointerTokens.contains(root) => Some(i64(localPointerTokens(root)))
      case p.Term.Select(root, Nil, _) if fieldAliases.get(root).exists(x => identityFields(x._1)) =>
        Some(rwTerm(fieldAliases(root)._2))
      case selected if selectedIdentityField(selected) => Some(rwTerm(selected))
      case _                                           => None
    }
    def equality(x: p.Term, y: p.Term, eq: (p.Term, p.Term) => p.Intr): p.Expr =
      if (selectedIdentityField(x) || selectedIdentityField(y)) {
        (identityComparable(x), identityComparable(y)) match {
          case (Some(a), Some(b)) => p.Expr.IntrOp(eq(a, b))
          case _                  => rewrittenEquality(x, y, eq)
        }
      } else rewrittenEquality(x, y, eq)

    def rewrittenEquality(x: p.Term, y: p.Term, eq: (p.Term, p.Term) => p.Intr): p.Expr = {
      val a = rwTerm(x)
      val b = rwTerm(y)
      val aa = x match {
        case _: p.Term.NullPtrConst if b.tpe == I64 => i64(0)
        case _                                      => a
      }
      val bb = y match {
        case _: p.Term.NullPtrConst if a.tpe == I64 => i64(0)
        case _                                      => b
      }
      p.Expr.IntrOp(eq(aa, bb))
    }

    def rwExpr(e: p.Expr): p.Expr = e match {
      case p.Expr.Alias(t) => p.Expr.Alias(rwTerm(t))
      case p.Expr.Cast(from, as) if isPtr(from.tpe) && arenaTerm(from) =>
        val v = ptrValue(from)
        if (isPtr(as) || as == I64) p.Expr.Alias(v) else p.Expr.Cast(v, as)
      case p.Expr.Cast(from, as) => p.Expr.Cast(rwTerm(from), as)
      case p.Expr.RefTo(base, idx, comp, p.Type.Space.Private, r) =>
        p.Expr.RefTo(rwTerm(base), idx.map(rwTerm), i64ify(comp), p.Type.Space.Private, r)
      case p.Expr.RefTo(base, idx, comp, sp, r) if arenaTerm(base) =>
        val off0 = if (isPtr(base.tpe)) ptrValue(base) else addrOffset(base)
        p.Expr.Alias(add(off0, idx.fold(i64(0))(i => mulBytes(rwTerm(i), comp))))
      case p.Expr.RefTo(base, idx, comp, sp, r) => p.Expr.RefTo(rwTerm(base), idx.map(rwTerm), comp, sp, r)
      case p.Expr.Index(base, idx, comp) =>
        derefOffset(base) match {
          case Some(off0) => p.Expr.Alias(loadAt(add(off0, mulBytes(rwTerm(idx), comp)), comp))
          case None       => p.Expr.Index(rwTerm(base), rwTerm(idx), i64ify(comp))
        }
      case p.Expr.Alloc(c, sz, sp, r)            => p.Expr.Alloc(c, rwTerm(sz), sp, r)
      case p.Expr.ForeignCall(n, args, rtn)      => p.Expr.ForeignCall(n, args.map(rwTerm), rtn)
      case p.Expr.Invoke(n, ts, recv, args, rtn) => p.Expr.Invoke(n, ts, recv.map(rwTerm), args.map(rwTerm), rtn)
      case p.Expr.IntrOp(p.Intr.LogicEq(x, y))   => equality(x, y, p.Intr.LogicEq.apply)
      case p.Expr.IntrOp(p.Intr.LogicNeq(x, y))  => equality(x, y, p.Intr.LogicNeq.apply)
      case op: p.Expr.IntrOp                     => op.modifyAll[p.Term](rwTerm)
      case op: p.Expr.MathOp                     => op.modifyAll[p.Term](rwTerm)
      case p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op, ptr, value, scope, order, rtn)) if arenaTerm(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op, arenaScalarRef(ptr, rtn), rwTerm(value), scope, order, rtn))
      case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(ptr, rtn)) if arenaTerm(ptr) =>
        if (isAgg(rtn))
          p.Expr.Alias(
            volatileLoadAt(
              derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
              rtn
            )
          )
        else p.Expr.SpecOp(p.Spec.GpuVolatileLoad(arenaScalarRef(ptr, rtn), rtn))
      case p.Expr.SpecOp(p.Spec.GpuVolatileStore(ptr, value)) if arenaTerm(ptr) =>
        if (isAgg(value.tpe)) {
          volatileStoreAt(
            derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
            value.tpe,
            value
          )
          p.Expr.Alias(p.Term.Unit0Const)
        } else p.Expr.SpecOp(p.Spec.GpuVolatileStore(arenaScalarRef(ptr, value.tpe), rwTerm(value)))
      case op: p.Expr.SpecOp => op.modifyAll[p.Term](rwTerm)
      case x                 => x
    }

    def rwInit(n: p.Named, e: p.Expr): (p.Named, p.Expr) =
      if (isArena(n) && isPtr(n.tpe)) {
        val nn = i64Var(n)
        e match { case p.Expr.Alias(_: p.Term.NullPtrConst) => (nn, p.Expr.Alias(i64(0))); case _ => (nn, rwExpr(e)) }
      } else (n, rwExpr(e))

    val out = leaf match {
      case p.Stmt.Var(n, Some(e), m) =>
        arrayAlias(n, e, m).toList match {
          case Nil     => val (nn, ne) = rwInit(n, e); List(p.Stmt.Var(nn, Some(ne), m))
          case aliases => aliases
        }
      case p.Stmt.Var(n, None, m) => List(p.Stmt.Var(if (isArena(n) && isPtr(n.tpe)) i64Var(n) else n, None, m))
      case p.Stmt.Mut(p.Term.Select(n, Nil, t), e) =>
        if (isArena(n) && isPtr(n.tpe)) List(p.Stmt.Mut(p.Term.Select(i64Var(n), Nil, I64), rwExpr(e)))
        else List(p.Stmt.Mut(p.Term.Select(n, Nil, t), rwExpr(e)))
      case p.Stmt.Mut(p.Term.Select(n, steps, scalarT), e) =>
        // mirrors rwTerm: a select crossing into arena partway (a stack-local iterator's node pointer
        // chased into the heap) still needs the byte-offset store, not a plain local field write
        lvalueOffset(n, steps) match {
          case Some(off) => storeVal(off, scalarT, bind("st", rwExpr(e)))
          case None      =>
            // local struct field write; a pointer field is now i64
            val identityField = isPtr(scalarT) && isIdentityField(n, steps)
            val lhsT          = if (identityField) I64 else i64ify(scalarT)
            val rhs = e match {
              case p.Expr.Alias(_: p.Term.NullPtrConst) if identityField => p.Expr.Alias(i64(0))
              case p.Expr.Alias(p.Term.Select(root, Nil, _)) if identityField && localPointerTokens.contains(root) =>
                p.Expr.Alias(i64(localPointerTokens(root)))
              case p.Expr.Cast(source, _: p.Type.Ptr) if identityField =>
                identityComparable(source).map(p.Expr.Alias.apply).getOrElse(rwExpr(e))
              case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) if identityField =>
                localReferenceKey(base, index)
                  .flatMap(localReferenceTokens.get)
                  .map(token => p.Expr.Alias(i64(token)))
                  .getOrElse(rwExpr(e))
              case _ => rwExpr(e)
            }
            List(p.Stmt.Mut(p.Term.Select(n, steps.map(rwStep), lhsT), rhs))
        }
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
