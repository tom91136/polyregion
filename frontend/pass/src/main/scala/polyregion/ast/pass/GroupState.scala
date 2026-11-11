package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// work-group state shared by the two lowerings of barrier kernels (GroupRegions, GroupUniform). code is cut into units,
// each run by one lane loop, and a variable used in more than one unit has to outlive the loop that defines it:
//   uniform, immutable value from group constants        ->  one shared scalar written alike by every lane
//   lane-pure immutable value (lid, lid % 16, ...)       ->  recomputed at the start of each unit that reads it
//   anything else used in more than one unit             ->  a per-lane slot in the work-group state, one array per variable
//   address taken with a barrier in scope                ->  a per-lane slot even when used in one unit
// examples:
//   l = lid; barrier(); out[l] = 1                       ->  unit 1 declares l = lid again, no state
//   a = in[lid]; barrier(); out[lid] = a                 ->  a[lid] = in[lid] in unit 0, out[lid] = a[lid] in unit 1
// edge cases:
//   a rematerialised value that divides                  ->  only by a literal other than 0 and -1, so replays cannot trap
//   whole-array assignment to a crossing array           ->  declined, the entry keeps the lane abi
private[pass] object GroupState {
  import GroupVocabulary.{Base, GroupId, Lanes, Lid, addressOfOwn, isBarrier, sel, u32}

  val U32     = p.Type.IntU32
  val U64     = p.Type.IntU64
  val Bool    = p.Type.Bool1
  val G       = p.Type.Space.Global
  val Wrapped = "#gf_value"

  def alias(t: p.Term): p.Expr = p.Expr.Alias(t)
  def intr(op: p.Intr): p.Expr = p.Expr.IntrOp(op)
  def u64(v: Long): p.Term     = p.Term.IntU64Const(v)

  def hasBarrier(s: p.Stmt): Boolean =
    s.collectFirst_[p.Expr] { case p.Expr.SpecOp(op) if isBarrier(op) => () }.isDefined

  def referenced(stmts: List[p.Stmt]): List[String] =
    stmts.flatMap(s =>
      s.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol } :::
        s.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root.symbol }
    )

  def unwrap(s: p.Stmt): p.Stmt = s match {
    case p.Stmt.Annotated(inner, _, _) => unwrap(inner)
    case other                         => other
  }

  def scalar(tpe: p.Type): Boolean = tpe match {
    case p.Type.Ptr(_, _) => true
    case t                => t.kind == p.Type.Kind.Integral || t.kind == p.Type.Kind.Fractional
  }

  // variables whose storage is referenced: an address taken, or an array bound to another name
  def addressed(stmts: List[p.Stmt]): Set[String] =
    stmts
      .flatMap(_.collectWhere[p.Expr] {
        case r: p.Expr.RefTo                                             => addressOfOwn(r).map(_.symbol)
        case p.Expr.Alias(p.Term.Select(root, Nil, p.Type.Arr(_, _, _))) => Some(root.symbol)
      })
      .flatten
      .toSet

  // variables whose address is taken while a later barrier is still in their scope
  def pinned(body: List[p.Stmt]): Set[String] = {
    val referenced = addressed(body)
    def block(stmts: List[p.Stmt]): List[String] = stmts.tails.toList.flatMap {
      case s :: rest =>
        val here = unwrap(s) match {
          case p.Stmt.Var(n, _, _) if referenced(n.symbol) && rest.exists(hasBarrier) => List(n.symbol)
          case _                                                                      => Nil
        }
        here ::: (unwrap(s) match {
          case p.Stmt.Cond(_, t, f)      => block(t) ::: block(f)
          case p.Stmt.While(_, b)        => block(b)
          case p.Stmt.ForRange(body = b) => block(b)
          case _                         => Nil
        })
      case Nil => Nil
    }
    block(body).toSet
  }

  def fixed(prepared: GroupVocabulary.Prepared): Set[String] =
    (GroupVocabulary.params(prepared.entry) ::: List(Lanes, GroupId, Base)).map(_.symbol).toSet ++
      prepared.prologue.collect { case p.Stmt.Var(n, _, _) => n.symbol }

  def fieldPath(defs: Map[p.Sym, p.StructDef], tpe: p.Type, steps: List[p.PathStep]): Boolean = steps match {
    case Nil => true
    case p.PathStep.Field(name) :: rest =>
      tpe match {
        case p.Type.Struct(sym, _) =>
          defs.get(sym).flatMap(_.members.find(_.symbol == name)).exists(m => fieldPath(defs, m.tpe, rest))
        case _ => false
      }
    case _ => false
  }

  def stableTerm(defs: Map[p.Sym, p.StructDef], fix: String => Boolean, bound: String => Boolean)(t: p.Term): Boolean =
    t match {
      case p.Term.Select(root, steps, _) =>
        (steps.isEmpty && bound(root.symbol)) || (fix(root.symbol) && fieldPath(defs, root.tpe, steps))
      case _ => true
    }

  enum Storage {
    case Lane(view: p.Named, comp: p.Type, wrapped: Boolean)
    case Shared
    case Unit
  }

  case class Remat(name: p.Named, init: p.Expr)

  case class Classes(
      crossing: List[p.Named],
      storage: Map[String, Storage],
      wrappers: List[p.StructDef],
      remat: List[Remat]
  )

  private def literal(t: p.Term): Option[Long] = t match {
    case p.Term.IntU8Const(v)  => Some(v.toLong)
    case p.Term.IntU16Const(v) => Some(v.toLong)
    case p.Term.IntU32Const(v) => Some(v.toLong)
    case p.Term.IntU64Const(v) => Some(v)
    case p.Term.IntS8Const(v)  => Some(v.toLong)
    case p.Term.IntS16Const(v) => Some(v.toLong)
    case p.Term.IntS32Const(v) => Some(v.toLong)
    case p.Term.IntS64Const(v) => Some(v)
    case _                     => None
  }

  private def safeDivisor(t: p.Term): Boolean = literal(t).exists(v => v != 0 && v != -1)

  private def total(op: p.Intr): Boolean = op match {
    case p.Intr.Div(_, y, tpe) => tpe.kind != p.Type.Kind.Integral || safeDivisor(y)
    case p.Intr.Rem(_, y, tpe) => tpe.kind != p.Type.Kind.Integral || safeDivisor(y)
    case _                     => true
  }

  def classify(
      prepared: GroupVocabulary.Prepared,
      defs: List[p.StructDef],
      units: Vector[List[p.Stmt]],
      pinned: Set[String]
  ): Either[String, Classes] = {
    val fresh  = prepared.fresh
    val defMap = defs.map(d => d.name -> d).toMap
    val all    = units.flatten.toList

    val declarations = all.flatMap(_.collectWhere[p.Stmt] { case v: p.Stmt.Var => v }).groupBy(_.name.symbol)
    val seen         = mutable.LinkedHashMap.empty[String, mutable.Set[Int]]
    units.zipWithIndex.foreach { (code, k) =>
      referenced(code).foreach(sym => if (declarations.contains(sym)) seen.getOrElseUpdate(sym, mutable.Set.empty) += k)
    }
    val used = seen.collect { case (sym, ks) if ks.size > 1 || pinned(sym) => declarations(sym).head.name }.toList

    val written = all
      .flatMap(_.collectWhere[p.Stmt] {
        case p.Stmt.Mut(p.Term.Select(root, steps, _), _) if GroupVocabulary.ownStorage(root, steps) => root.symbol
      })
      .toSet
    val rebound =
      all.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Mut(p.Term.Select(root, Nil, _), _) => root.symbol }).toSet
    val aliased = addressed(all)
    val fix     = fixed(prepared)

    var uniform                         = Set.empty[String]
    def uniformTerm(t: p.Term): Boolean = stableTerm(defMap, fix, uniform)(t)
    def uniformExpr(e: p.Expr): Boolean = e match {
      case p.Expr.Alias(t)       => uniformTerm(t)
      case p.Expr.IntrOp(op)     => op.terms.forall(uniformTerm)
      case p.Expr.MathOp(op)     => op.terms.forall(uniformTerm)
      case p.Expr.Cast(t, _)     => uniformTerm(t)
      case p.Expr.BitCast(t, _)  => uniformTerm(t)
      case p.Expr.SizeOf(_)      => true
      case p.Expr.OffsetOf(_, _) => true
      case p.Expr.SpecOp(
            op @ (_: p.Spec.GpuGroupIdx | _: p.Spec.GpuGroupSize | _: p.Spec.GpuLocalSize | _: p.Spec.GpuGlobalSize)
          ) =>
        op.terms.forall(uniformTerm)
      case p.Expr.RefTo(p.Term.Select(root, Nil, _), Some(idx), _, _, _) =>
        (fix(root.symbol) || uniform(root.symbol)) && uniformTerm(idx)
      case _ => false
    }
    val candidates = declarations.toList.sortBy(_._1).collect {
      case (sym, List(p.Stmt.Var(n, Some(e), _))) if scalar(n.tpe) && !written(sym) && !aliased(sym) && !pinned(sym) =>
        sym -> e
    }
    var changed = true
    while (changed) {
      changed = false
      candidates.foreach { (sym, e) =>
        if (!uniform(sym) && uniformExpr(e)) {
          uniform += sym
          changed = true
        }
      }
    }

    val replayed = mutable.LinkedHashMap.empty[String, p.Expr]
    def pure(t: p.Term): Boolean = t match {
      case p.Term.Select(root, steps, _) =>
        (steps.isEmpty && (root.symbol == Lid.symbol || uniform(root.symbol) || replayed.contains(root.symbol))) ||
        (fix(root.symbol) && fieldPath(defMap, root.tpe, steps))
      case _ => true
    }
    def replayable(e: p.Expr): Boolean = e match {
      case p.Expr.Alias(t)                          => pure(t)
      case p.Expr.Cast(t, _)                        => pure(t)
      case p.Expr.BitCast(t, _)                     => pure(t)
      case p.Expr.IntrOp(op)                        => op.terms.forall(pure) && total(op)
      case p.Expr.MathOp(op)                        => op.terms.forall(pure)
      case p.Expr.SizeOf(_) | p.Expr.OffsetOf(_, _) => true
      case _                                        => false
    }
    changed = true
    while (changed) {
      changed = false
      candidates.foreach { (sym, e) =>
        if (!uniform(sym) && !replayed.contains(sym) && replayable(e)) {
          replayed(sym) = e
          changed = true
        }
      }
    }
    val crossing = used.filterNot(n => replayed.contains(n.symbol))
    val remat    = replayed.toList.map((sym, e) => Remat(declarations(sym).head.name, e))

    val wrappers       = mutable.ListBuffer.empty[p.StructDef]
    val storageBuilder = Map.newBuilder[String, Storage]
    crossing.foreach { n =>
      storageBuilder += n.symbol -> (n.tpe match {
        case p.Type.Unit0           => Storage.Unit
        case _ if uniform(n.symbol) => Storage.Shared
        case arr: p.Type.Arr =>
          val name = p.Sym(List(s"#gf_lane.${n.symbol}"))
          wrappers += p.StructDef(name, Nil, List(p.Named(Wrapped, arr)), Nil)
          val comp = p.Type.Struct(name, Nil)
          Storage.Lane(fresh(s"${n.symbol}#gf_view", p.Type.Ptr(comp, G)), comp, wrapped = true)
        case tpe => Storage.Lane(fresh(s"${n.symbol}#gf_view", p.Type.Ptr(tpe, G)), tpe, wrapped = false)
      })
    }
    val storage = storageBuilder.result()

    val reassigned = crossing.find { n =>
      storage(n.symbol) match {
        case Storage.Lane(_, _, true) =>
          rebound(n.symbol) || declarations(n.symbol).exists {
            case p.Stmt.Var(_, Some(p.Expr.Alias(p.Term.Poison(_))), _) | p.Stmt.Var(_, None, _) => false
            case _                                                                               => true
          }
        case _ => false
      }
    }
    reassigned match {
      case Some(n) => Left(s"the crossing array ${n.symbol} is assigned as a whole")
      case None    => Right(Classes(crossing, storage, wrappers.toList, remat))
    }
  }

  case class Scope(at: Map[String, p.Named], re: Map[String, p.Named], prelude: List[p.Stmt])

  final class Access(prepared: GroupVocabulary.Prepared, classes: Classes) {
    private val fresh   = prepared.fresh
    private val storage = classes.storage
    private val replays = classes.remat.map(r => r.name.symbol -> r).toMap

    val perLane: List[(p.Named, Storage.Lane)] = classes.crossing.flatMap(n =>
      storage(n.symbol) match {
        case s: Storage.Lane => Some(n -> s)
        case _               => None
      }
    )
    private val views = perLane.map((n, s) => n.symbol -> s.view).toMap
    private val unitSites = classes.crossing.collect {
      case n if storage(n.symbol) == Storage.Unit => n.symbol -> fresh(s"${n.symbol}#gf_site", n.tpe)
    }.toMap

    def shared: List[p.Stmt] = classes.crossing.collect {
      case n if storage(n.symbol) == Storage.Shared || storage(n.symbol) == Storage.Unit =>
        p.Stmt.Var(n, None, isMutable = true)
    }

    def scope(code: List[p.Stmt]): Scope = {
      val names        = referenced(code).toSet
      val declaredHere = code.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol }).toSet
      val at = perLane.collect {
        case (n, s) if names(n.symbol) => n.symbol -> fresh(s"${n.symbol}#gf_at", s.view.tpe)
      }.toMap
      val order = mutable.LinkedHashSet.empty[String]
      def visit(sym: String): Unit = if (!order(sym)) {
        replays(sym).init.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root.symbol }.foreach { dep =>
          if (replays.contains(dep)) visit(dep)
        }
        order += sym
      }
      names.toList.sorted.foreach(sym => if (replays.contains(sym) && !declaredHere(sym)) visit(sym))
      val re = order.toList.map(sym => sym -> fresh(s"${sym}#gf_re", replays(sym).name.tpe)).toMap
      val prelude = order.toList.map { sym =>
        p.Stmt.Var(re(sym), Some(GroupVocabulary.renameRoots(replays(sym).init, re)))
      }
      Scope(at, re, prelude)
    }

    def slots(scope: Scope): List[p.Stmt] = scope.at.toList.sortBy(_._1).map { (sym, ptr) =>
      val Storage.Lane(view, comp, _) = storage(sym): @unchecked
      p.Stmt.Var(ptr, Some(p.Expr.RefTo(sel(view), Some(sel(Lid)), comp, G, p.Region.Rooted(view))))
    }

    // a uniform control term reads lane 0's copy of a per-lane variable
    def control(t: p.Term): p.Term = t match {
      case s @ p.Term.Select(root, steps, tpe) =>
        views.get(root.symbol).fold(s)(view => p.Term.Select(view, p.PathStep.Deref :: steps, tpe))
      case other => other
    }

    private def rooted(region: p.Region): p.Region = region match {
      case p.Region.Rooted(root) => views.get(root.symbol).fold(region)(p.Region.Rooted(_))
      case other                 => other
    }

    def rewrite[A](a: A, scope: Scope)(using Traversal[A, p.Term], Traversal[A, p.Expr]): A =
      a.modifyAll[p.Term] {
        case s @ p.Term.Select(root, steps, tpe) =>
          scope.at.get(root.symbol) match {
            case Some(ptr) =>
              val prefix = storage(root.symbol) match {
                case Storage.Lane(_, _, true) => List(p.PathStep.Deref, p.PathStep.Field(Wrapped))
                case _                        => List(p.PathStep.Deref)
              }
              p.Term.Select(ptr, prefix ::: steps, tpe)
            case None => scope.re.get(root.symbol).fold(s)(p.Term.Select(_, steps, tpe))
          }
        case p.Term.NullPtrConst(comp, space, region) => p.Term.NullPtrConst(comp, space, rooted(region))
        case other                                    => other
      }.modifyAll[p.Expr] {
        case p.Expr.RefTo(lhs, idx, comp, space, region) => p.Expr.RefTo(lhs, idx, comp, space, rooted(region))
        case p.Expr.Alloc(comp, size, space, region)     => p.Expr.Alloc(comp, size, space, rooted(region))
        case other                                       => other
      }

    def statement(s: p.Stmt, scope: Scope): List[p.Stmt] = {
      def nested(xs: List[p.Stmt]) = xs.flatMap(statement(_, scope))
      s match {
        case p.Stmt.Var(n, init, mut) =>
          storage.get(n.symbol) match {
            case Some(Storage.Lane(_, _, wrapped)) =>
              val slot = rewrite(sel(n), scope)
              (n.tpe, init) match {
                case _ if wrapped => Nil
                case (p.Type.Struct(_, _), None) =>
                  val zero = fresh("#gf_zero", n.tpe)
                  List(p.Stmt.Var(zero, None), p.Stmt.Mut(slot, alias(sel(zero))))
                case (_, None)    => Nil
                case (_, Some(e)) => List(p.Stmt.Mut(slot, rewrite(e, scope)))
              }
            case Some(Storage.Shared) => init.map(e => p.Stmt.Mut(sel(n), rewrite(e, scope))).toList
            case Some(Storage.Unit)   => List(p.Stmt.Var(unitSites(n.symbol), init.map(rewrite(_, scope)), mut))
            case None                 => List(p.Stmt.Var(n, init.map(rewrite(_, scope)), mut))
          }
        case p.Stmt.Mut(lhs, e) => List(p.Stmt.Mut(rewrite(lhs, scope), rewrite(e, scope)))
        case p.Stmt.Update(lhs, idx, v) =>
          List(p.Stmt.Update(rewrite(lhs, scope), rewrite(idx, scope), rewrite(v, scope)))
        case p.Stmt.While(c, b) => List(p.Stmt.While(rewrite(c, scope), nested(b)))
        case p.Stmt.ForRange(i, lb, ub, st, b) =>
          List(p.Stmt.ForRange(i, rewrite(lb, scope), rewrite(ub, scope), rewrite(st, scope), nested(b)))
        case p.Stmt.Cond(c, t, f)                  => List(p.Stmt.Cond(rewrite(c, scope), nested(t), nested(f)))
        case p.Stmt.Return(e)                      => List(p.Stmt.Return(rewrite(e, scope)))
        case p.Stmt.Break | p.Stmt.Cont            => List(s)
        case p.Stmt.Annotated(inner, pos, comment) => statement(inner, scope).map(p.Stmt.Annotated(_, pos, comment))
        case other => throw IllegalStateException(s"GroupFission: unexpected ${other.repr} in a region")
      }
    }

    // the state allocation: `leading` plain u32 arrays first, then one array per per-lane variable, each padded to 64 bytes
    def state(leading: List[p.Type]): (List[p.Stmt], p.Named) = {
      val state   = fresh("#gf_state", p.Type.Ptr(p.Type.IntU8, G))
      val lanes64 = fresh("#gf_lanes64", U64)
      val layout  = mutable.ListBuffer.empty[p.Stmt]
      val (starts, total) = (leading ::: perLane.map(_._2.comp)).foldLeft((List.empty[p.Term], u64(0): p.Term)) {
        case ((acc, start), comp) =>
          val size    = fresh("#gf_size", U64)
          val bytes   = fresh("#gf_bytes", U64)
          val padded  = fresh("#gf_padded", U64)
          val rounded = fresh("#gf_segment", U64)
          val end     = fresh("#gf_end", U64)
          layout ++= List(
            p.Stmt.Var(size, Some(p.Expr.SizeOf(comp))),
            p.Stmt.Var(bytes, Some(intr(p.Intr.Mul(sel(lanes64), sel(size), U64)))),
            p.Stmt.Var(padded, Some(intr(p.Intr.Add(sel(bytes), u64(63), U64)))),
            p.Stmt.Var(rounded, Some(intr(p.Intr.BAnd(sel(padded), u64(-64), U64)))),
            p.Stmt.Var(end, Some(intr(p.Intr.Add(start, sel(rounded), U64))))
          )
          (start :: acc, sel(end))
      }
      val views = perLane.zip(starts.reverse.drop(leading.size)).flatMap { case ((_, s), start) =>
        val raw = fresh("#gf_raw", state.tpe)
        List(
          p.Stmt.Var(raw, Some(p.Expr.RefTo(sel(state), Some(start), p.Type.IntU8, G, p.Region.Rooted(state)))),
          p.Stmt.Var(s.view, Some(p.Expr.Cast(sel(raw), s.view.tpe)))
        )
      }
      val call = p.Stmt.Var(state, Some(p.Expr.ForeignCall(p.Conventions.HostGroupState, List(total), state.tpe)))
      (p.Stmt.Var(lanes64, Some(p.Expr.Cast(sel(Lanes), U64))) :: layout.toList ::: List(call) ::: views, state)
    }
  }
}
