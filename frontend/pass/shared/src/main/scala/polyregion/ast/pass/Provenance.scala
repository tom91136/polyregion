package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

object Provenance {

  def isPtr(t: p.Type): Boolean = t match { case _: p.Type.Ptr => true; case _ => false }

  // join over all assignments: disagreeing roots collapse to Opaque
  def derivedIn(f: p.Function): Map[p.Named, p.Region] =
    f.collectAll[p.Stmt].foldLeft(Map.empty[p.Named, p.Region]) { (m, stmt) =>
      def trace(root: p.Named): p.Region = m.getOrElse(root, p.Region.Rooted(root))
      def regionOf(n: p.Named, e: p.Expr): p.Region = e match {
        case p.Expr.RefTo(p.Term.Select(root, _, _), _, _, _, _)   => trace(root)
        case p.Expr.Cast(p.Term.Select(root, _, _), _: p.Type.Ptr) => trace(root)
        case p.Expr.Alias(p.Term.Select(root, _, _))               => trace(root)
        case _: p.Expr.Alloc                                       => p.Region.Rooted(n)
        case _                                                     => p.Region.Opaque
      }
      def join(n: p.Named, r: p.Region): Map[p.Named, p.Region] = m.get(n) match {
        case None | Some(`r`) => m + (n -> r)
        // addresses of two distinct non-pointer locals (e.g. `std::min(&a, &b)` over stack scalars)
        // stay stack-rooted: the backend selects among local vars, no binding slot needed; only a
        // join that pulls in a pointer root is truly opaque
        case Some(p.Region.Rooted(a)) =>
          r match {
            case p.Region.Rooted(b) if !isPtr(a.tpe) && !isPtr(b.tpe) => m
            case _                                                    => m + (n -> p.Region.Opaque)
          }
        case Some(_) => m + (n -> p.Region.Opaque)
      }
      stmt match {
        case p.Stmt.Var(n, Some(e), _) if isPtr(n.tpe)               => join(n, regionOf(n, e))
        case p.Stmt.Mut(p.Term.Select(n, Nil, _), e) if isPtr(n.tpe) => join(n, regionOf(n, e))
        case _                                                       => m
      }
    }

  // names reassigned by a bare Mut (re-aim); a stepped Mut writes through and keeps the address
  def reassignedIn(f: p.Function): Set[String] =
    f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _), _) => n.symbol }.toSet

  def at(derived: Map[p.Named, p.Region], t: p.Term): p.Region = t match {
    case p.Term.Select(root, _, _) => derived.getOrElse(root, p.Region.Rooted(root))
    case _                         => p.Region.Opaque
  }
}
