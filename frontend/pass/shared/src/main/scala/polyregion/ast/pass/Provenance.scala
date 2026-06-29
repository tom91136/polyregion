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
    derivedIn(f).toList.sortBy(_._1.symbol).flatMap {
      case (n, p.Region.Rooted(r)) if r != n =>
        for {
          sn <- spaceOf(n.tpe)
          sr <- spaceOf(r.tpe) if sn != sr
        } yield (n, r, sn, sr)
      case _ => None
    }

  // disagreement is Opaque, except distinct stack scalars (e.g. `std::min(&a, &b)`) which are never arena-marshalled
  def joinRegions(x: p.Region, y: p.Region): p.Region = (x, y) match {
    case (p.Region.Rooted(a), p.Region.Rooted(b)) if a == b || (!isPtr(a.tpe) && !isPtr(b.tpe)) => x
    case _                                                                                      => p.Region.Opaque
  }

  // a pointer loaded out of memory targets a separate allocation, so stepping a rooted object to a pointer
  // is Opaque; only arena lowering needs this (arena=false keeps the Select root, behaviour-neutral)
  def selectRegion(base: p.Region, steps: List[p.PathStep], tpe: p.Type): p.Region =
    if (base == p.Region.Opaque) p.Region.Opaque
    else if (steps.isEmpty) base
    else if (isPtr(tpe)) p.Region.Opaque
    else base

  def derivedIn(f: p.Function, arena: Boolean = false): Map[p.Named, p.Region] =
    f.collectAll[p.Stmt].foldLeft(Map.empty[p.Named, p.Region]) { (m, stmt) =>
      def trace(root: p.Named): p.Region = m.getOrElse(root, p.Region.Rooted(root))
      def of(root: p.Named, steps: List[p.PathStep], t: p.Type): p.Region =
        if (arena) selectRegion(trace(root), steps, t) else trace(root)
      def regionOf(n: p.Named, e: p.Expr): p.Region = e match {
        case p.Expr.RefTo(p.Term.Select(root, steps, t), _, _, _, _)   => of(root, steps, t)
        case p.Expr.Cast(p.Term.Select(root, steps, t), _: p.Type.Ptr) => of(root, steps, t)
        case p.Expr.Alias(p.Term.Select(root, steps, t))               => of(root, steps, t)
        case _: p.Expr.Alloc                                           => p.Region.Rooted(n)
        case p.Expr.Alias(p.Term.StringConst(_))                       => p.Region.Rooted(n)
        case p.Expr.Cast(p.Term.StringConst(_), _: p.Type.Ptr)         => p.Region.Rooted(n)
        case _                                                         => p.Region.Opaque
      }
      def join(n: p.Named, r: p.Region): Map[p.Named, p.Region] =
        m + (n -> m.get(n).fold(r)(joinRegions(_, r)))
      stmt match {
        case p.Stmt.Var(n, Some(e), _) if isPtr(n.tpe)               => join(n, regionOf(n, e))
        case p.Stmt.Mut(p.Term.Select(n, Nil, _), e) if isPtr(n.tpe) => join(n, regionOf(n, e))
        case _                                                       => m
      }
    }

  def reassignedIn(f: p.Function): Set[String] =
    f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _), _) => n.symbol }.toSet

  def at(derived: Map[p.Named, p.Region], t: p.Term, arena: Boolean = false): p.Region = t match {
    case p.Term.Select(root, steps, tpe) =>
      val base = derived.getOrElse(root, p.Region.Rooted(root))
      if (arena) selectRegion(base, steps, tpe) else base
    case _: p.Term.StringConst => p.Region.Rooted(p.Named("#strconst", t.tpe))
    case _                     => p.Region.Opaque
  }
}
