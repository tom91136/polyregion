package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// per-function pointer provenance: maps each pointer name to the root it ultimately addresses
// `derivedIn` joins over every assignment; disagreeing roots collapse to Opaque
// examples:
//   var p = alloc(...)                  ->  p     -> Rooted(p)
//   var q = &a.x                        ->  q     -> Rooted(a)        (a = its root)
//   var r = (T*) s; ... r = &a.x        ->  r     -> Rooted(a)        (Cast/RefTo trace root)
//   var p = &a.x; p = &b.x  (a,b ptr)   ->  p     -> Opaque           (two pointer roots disagree)
//   x = c ? &a : &b  (a,b non-ptr)      ->  x     -> Rooted(a)        (stack scalars stay rooted, keeps first)
//   var n = foo()                       ->  n     -> Opaque           (unknown producer)
// edge cases:
//   bare `q = ...` (re-aim)             ->  reassignedIn includes q
//   stepped `q.f = ...` (write-thru)    ->  not in reassignedIn       (keeps the address)
object Provenance {

  def isPtr(t: p.Type): Boolean = t match { case _: p.Type.Ptr => true; case _ => false }

  private def isStruct(t: p.Type): Boolean = t match { case _: p.Type.Struct => true; case _ => false }

  def spaceOf(t: p.Type): Option[p.Type.Space] = t match {
    case p.Type.Ptr(_, s)    => Some(s)
    case p.Type.Arr(_, _, s) => Some(s)
    case _                   => None
  }

  def withSpace(t: p.Type, s: p.Type.Space): p.Type = t match {
    case p.Type.Ptr(c, _)    => p.Type.Ptr(c, s)
    case p.Type.Arr(c, l, _) => p.Type.Arr(c, l, s)
    case other               => other
  }

  // rooted pointers whose own space differs from the resource they address: (pointer, root, ptrSpace, rootSpace)
  def spaceMismatches(f: p.Function): List[(p.Named, p.Named, p.Type.Space, p.Type.Space)] =
    derivedIn(f, trackSlots = true, trackEncoded = true).toList.sortBy(_._1.symbol).flatMap {
      case (n, p.Region.Rooted(r)) if r != n =>
        for {
          sn <- spaceOf(n.tpe)
          sr <- spaceOf(r.tpe) if sn != sr
        } yield (n, r, sn, sr)
      case _ => None
    }

  // disagreement is Opaque, except distinct stack scalars (e.g. `std::min(&a, &b)`) which are never arena-marshalled
  def joinRegions(x: p.Region, y: p.Region): p.Region = (x, y) match {
    case (p.Region.Rooted(a), p.Region.Rooted(b)) if a == b || (spaceOf(a.tpe).isEmpty && spaceOf(b.tpe).isEmpty) =>
      x
    case _ => p.Region.Opaque
  }

  // a pointer loaded out of memory targets a separate allocation, so stepping a rooted object to a pointer
  // is Opaque; only arena lowering needs this (arena=false keeps the Select root, behaviour-neutral)
  def selectRegion(base: p.Region, steps: List[p.PathStep], tpe: p.Type): p.Region =
    if (base == p.Region.Opaque) p.Region.Opaque
    else if (steps.isEmpty) base
    else if (
      tpe match {
        case p.Type.Ptr(_, p.Type.Space.Global | p.Type.Space.Constant) => true
        case _                                                          => false
      }
    ) p.Region.Opaque
    else base

  def derivedIn(
      f: p.Function,
      arena: Boolean = false,
      trackSlots: Boolean = false,
      trackEncoded: Boolean = false
  ): Map[p.Named, p.Region] = {
    type Slot  = (String, List[p.PathStep])
    type State = (Map[p.Named, p.Region], Map[Slot, p.Region])

    val statements = f.collectAll[p.Stmt]
    val maxSlotDepth =
      f.collectAll[p.Term].collect { case p.Term.Select(_, steps, _) => steps.size }.maxOption.getOrElse(0)
    val declared =
      (statements.collect { case p.Stmt.Var(n, _, _) => n } :::
        f.receiver.toList.map(_.named) ::: f.args.map(_.named) ::: f.moduleCaptures.map(_.named) :::
        f.termCaptures.map(_.named)).map(n => n.symbol -> n).toMap
    val parameters =
      (f.receiver.toList ::: f.args ::: f.moduleCaptures ::: f.termCaptures).map(_.named)
    val initial = parameters.collect {
      case n if isPtr(n.tpe) => n -> p.Region.Rooted(n)
    }.toMap -> Map.empty[Slot, p.Region]

    def transfer(state: State, stmt: p.Stmt): State = {
      val (m, slots) = state
      def known(root: p.Named): Option[p.Region] =
        m.get(root)
          .orElse(m.collectFirst { case (named, region) if named.symbol == root.symbol => region })
          .orElse(declared.get(root.symbol).filter(n => isPtr(n.tpe)).map(p.Region.Rooted.apply))
      def trace(root: p.Named): p.Region = known(root).getOrElse(p.Region.Rooted(root))
      def slotted(root: p.Named, steps: List[p.PathStep]): Option[(Int, p.Region)] =
        Option
          .when(trackSlots) {
            slots.iterator
              .collect {
                case ((symbol, path), region) if symbol == root.symbol && steps.startsWith(path) =>
                  path.size -> region
              }
              .maxByOption(_._1)
          }
          .flatten
      def of(root: p.Named, steps: List[p.PathStep], t: p.Type): p.Region =
        slotted(root, steps) match {
          case Some((consumed, region)) => if (arena) selectRegion(region, steps.drop(consumed), t) else region
          case None if trackSlots && steps.nonEmpty && isPtr(t) => p.Region.Opaque
          case None if arena                                    => selectRegion(trace(root), steps, t)
          case None                                             => trace(root)
        }
      def knownAt(root: p.Named, steps: List[p.PathStep], t: p.Type): Option[p.Region] =
        slotted(root, steps) match {
          case Some((consumed, region)) =>
            Some(if (arena) selectRegion(region, steps.drop(consumed), t) else region)
          case None if !trackSlots || steps.isEmpty =>
            known(root).map(region => if (arena) selectRegion(region, steps, t) else region)
          case None => None
        }
      def termRegion(term: p.Term): Option[p.Region] = term match {
        case p.Term.Select(root, steps, t) => knownAt(root, steps, t)
        case _                             => None
      }
      def addressInt(t: p.Type): Boolean = t == p.Type.IntU64 || t == p.Type.IntS64
      def encodedPointer(e: p.Expr): Option[p.Region] = e match {
        case p.Expr.Alias(p.Term.Select(root, steps, t)) if addressInt(t)     => knownAt(root, steps, t)
        case p.Expr.Cast(p.Term.Select(root, steps, t), as) if addressInt(as) => knownAt(root, steps, t)
        case p.Expr.IntrOp(p.Intr.Add(x, y, rtn)) if addressInt(rtn) =>
          (termRegion(x), termRegion(y)) match {
            case (Some(region), None) => Some(region)
            case (None, Some(region)) => Some(region)
            case _                    => None
          }
        case p.Expr.IntrOp(p.Intr.Sub(x, y, rtn)) if addressInt(rtn) =>
          (termRegion(x), termRegion(y)) match {
            case (Some(region), None) => Some(region)
            case _                    => None
          }
        case _ => None
      }
      def regionOf(n: p.Named, e: p.Expr): Option[p.Region] = e match {
        case p.Expr.Alias(_: p.Term.NullPtrConst)                    => None
        case p.Expr.RefTo(p.Term.Select(root, steps, t), _, _, _, _) => Some(of(root, steps, t))
        case p.Expr.Index(p.Term.Select(_, _, _), _, p.Type.Ptr(_, p.Type.Space.Global | p.Type.Space.Constant)) =>
          Some(p.Region.Opaque)
        case p.Expr.Index(p.Term.Select(root, steps, t), _, _)         => Some(of(root, steps, t))
        case p.Expr.Cast(p.Term.Select(root, steps, t), _: p.Type.Ptr) => Some(of(root, steps, t))
        case p.Expr.Alias(p.Term.Select(root, steps, t))               => Some(of(root, steps, t))
        case _: p.Expr.Alloc                                           => Some(p.Region.Rooted(n))
        case p.Expr.Alias(p.Term.StringConst(_))                       => Some(p.Region.Rooted(n))
        case p.Expr.Cast(p.Term.StringConst(_), _: p.Type.Ptr)         => Some(p.Region.Rooted(n))
        case _                                                         => Some(p.Region.Opaque)
      }
      def join(n: p.Named, r: p.Region): Map[p.Named, p.Region] = {
        val existing = m.get(n).orElse(m.collectFirst { case (named, region) if named.symbol == n.symbol => region })
        m.filterNot(_._1.symbol == n.symbol) + (n -> existing.fold(r)(joinRegions(_, r)))
      }
      def updateEncoded(n: p.Named, e: p.Expr): Map[p.Named, p.Region] =
        if (!trackEncoded) m
        else
          encodedPointer(e) match {
            case Some(region)                                        => join(n, region)
            case None if m.keysIterator.exists(_.symbol == n.symbol) => join(n, p.Region.Opaque)
            case None                                                => m
          }
      def joinSlot(key: Slot, r: p.Region): Map[Slot, p.Region] =
        slots.updated(key, slots.get(key).fold(r)(joinRegions(_, r)))
      def copySlots(
          target: p.Named,
          targetPrefix: List[p.PathStep],
          source: p.Named,
          sourcePrefix: List[p.PathStep]
      ): Map[Slot, p.Region] = {
        val copied = slots.iterator
          .collect {
            case ((symbol, path), region) if symbol == source.symbol && path.startsWith(sourcePrefix) =>
              (targetPrefix ++ path.drop(sourcePrefix.size)) -> region
          }
          .filter { case (path, _) => path.size <= maxSlotDepth }
          .toMap
        val targetPaths = slots.keysIterator.collect {
          case (symbol, path) if symbol == target.symbol && path.startsWith(targetPrefix) => path
        }.toSet
        (targetPaths ++ copied.keySet).foldLeft(slots) { (acc, path) =>
          val key      = target.symbol -> path
          val incoming = copied.getOrElse(path, p.Region.Opaque)
          acc.updated(key, acc.get(key).fold(incoming)(joinRegions(_, incoming)))
        }
      }
      stmt match {
        case p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, steps, _))), _) if trackSlots && isStruct(n.tpe) =>
          (m, copySlots(n, Nil, root, steps))
        case p.Stmt.Var(n, Some(e), _) =>
          (if (isPtr(n.tpe)) regionOf(n, e).fold(m)(join(n, _)) else updateEncoded(n, e), slots)
        case p.Stmt.Mut(p.Term.Select(n, targetSteps, t), p.Expr.Alias(p.Term.Select(root, sourceSteps, _)))
            if trackSlots && isStruct(t) =>
          (m, copySlots(n, targetSteps, root, sourceSteps))
        case p.Stmt.Mut(p.Term.Select(n, Nil, _), e) =>
          (if (isPtr(n.tpe)) regionOf(n, e).fold(m)(join(n, _)) else updateEncoded(n, e), slots)
        case p.Stmt.Mut(p.Term.Select(n, steps, t), e) if trackSlots && steps.nonEmpty && isPtr(t) =>
          (m, regionOf(n, e).fold(slots)(joinSlot(n.symbol -> steps, _)))
        case _ => (m, slots)
      }
    }

    def joinStates(left: State, right: State): State = {
      val (lm, ls) = left
      val (rm, rs) = right
      def joinOptional(l: Option[p.Region], r: Option[p.Region]): p.Region =
        l.zip(r).map(joinRegions).getOrElse(p.Region.Opaque)
      val names = (lm.keysIterator ++ rm.keysIterator).map(n => n.symbol -> n).toMap
      val regions = names.valuesIterator.map { n =>
        val l = lm.collectFirst { case (named, region) if named.symbol == n.symbol => region }
        val r = rm.collectFirst { case (named, region) if named.symbol == n.symbol => region }
        n -> joinOptional(l, r)
      }.toMap
      val slotKeys = ls.keySet ++ rs.keySet
      val joinedSlots = slotKeys.iterator.map { key =>
        key -> joinOptional(ls.get(key), rs.get(key))
      }.toMap
      regions -> joinedSlots
    }

    def analyse(stmts: List[p.Stmt], state: State): State =
      stmts.foldLeft(state) { (current, stmt) =>
        stmt match {
          case p.Stmt.Cond(_, trueBr, falseBr) =>
            joinStates(analyse(trueBr, current), analyse(falseBr, current))
          case p.Stmt.While(_, body)             => loop(body, current)
          case p.Stmt.ForRange(_, _, _, _, body) => loop(body, current)
          case p.Stmt.Try(body, handlers, fin) =>
            val bodyOut = analyse(body, current)
            val handled = handlers.foldLeft(bodyOut) { (acc, handler) =>
              joinStates(acc, analyse(handler.body, joinStates(current, bodyOut)))
            }
            analyse(fin, handled)
          case p.Stmt.Annotated(inner, _, _) => analyse(List(inner), current)
          case p.Stmt.Raise(_, _, cleanup)   => analyse(cleanup, current)
          case other                         => transfer(current, other)
        }
      }

    def loop(body: List[p.Stmt], entry: State): State = {
      @annotation.tailrec
      def go(current: State): State = {
        val next = joinStates(entry, analyse(body, current))
        if (next == current) current else go(next)
      }
      go(entry)
    }

    val (derived, _) =
      if (trackSlots || trackEncoded) analyse(f.body, initial)
      else statements.foldLeft(Map.empty[p.Named, p.Region] -> Map.empty[Slot, p.Region])(transfer)
    derived
  }

  def reassignedIn(f: p.Function): Set[String] =
    f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _), _) => n.symbol }.toSet

  def at(derived: Map[p.Named, p.Region], t: p.Term, arena: Boolean = false): p.Region = t match {
    case p.Term.Select(root, steps, tpe) =>
      val base = derived
        .get(root)
        .orElse(derived.collectFirst { case (named, region) if named.symbol == root.symbol => region })
        .getOrElse(p.Region.Rooted(root))
      if (arena) selectRegion(base, steps, tpe) else base
    case _: p.Term.StringConst => p.Region.Rooted(p.Named("#strconst", t.tpe))
    case _                     => p.Region.Opaque
  }
}
