package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{PolyAST as p, *}
import polyregion.ast.Traversal.*

private[pass] object GroupVocabulary {
  private[pass] val U32 = p.Type.IntU32

  private[pass] val Lanes   = p.Named("#gf_lanes", U32)
  private[pass] val GroupId = p.Named("#gf_group", U32)
  private[pass] val Base    = p.Named("#gf_base", U32)
  private[pass] val Groups  = p.Named("#gf_groups", U32)
  private[pass] val Global  = p.Named("#gf_global", U32)
  private[pass] val Lid     = p.Named("#gf_lid", U32)

  private[pass] def sel(n: p.Named): p.Term.Select = p.Term.Select(n, Nil, n.tpe)
  private[pass] def u32(v: Int): p.Term            = p.Term.IntU32Const(v)
  private[pass] def spec(op: p.Spec): p.Expr       = p.Expr.SpecOp(op)

  private[pass] def isBarrier(op: p.Spec): Boolean = op match {
    case p.Spec.GpuBarrierGlobal | p.Spec.GpuBarrierLocal | p.Spec.GpuBarrierAll => true
    case _                                                                       => false
  }

  private[pass] def params(entry: p.Function): List[p.Named] =
    (entry.receiver.toList ::: entry.args ::: entry.moduleCaptures ::: entry.termCaptures).map(_.named)

  private[pass] def renameRoots[A](a: A, env: Map[String, p.Named])(using
      Traversal[A, p.Term],
      Traversal[A, p.Expr]
  ): A = {
    def rooted(region: p.Region): p.Region = region match {
      case p.Region.Rooted(root) => env.get(root.symbol).fold(region)(p.Region.Rooted(_))
      case other                 => other
    }
    if (env.isEmpty) a
    else
      a.modifyAll[p.Term] {
        case s @ p.Term.Select(root, steps, tpe)      => env.get(root.symbol).fold(s)(p.Term.Select(_, steps, tpe))
        case p.Term.NullPtrConst(comp, space, region) => p.Term.NullPtrConst(comp, space, rooted(region))
        case other                                    => other
      }.modifyAll[p.Expr] {
        case p.Expr.RefTo(lhs, idx, comp, space, region) => p.Expr.RefTo(lhs, idx, comp, space, rooted(region))
        case p.Expr.Alloc(comp, size, space, region)     => p.Expr.Alloc(comp, size, space, rooted(region))
        case other                                       => other
      }
  }

  // whether a reference to `root` through `steps` names the variable's own storage rather than memory behind it
  private[pass] def ownStorage(root: p.Named, steps: List[p.PathStep]): Boolean = root.tpe match {
    case p.Type.Ptr(_, _) => steps.isEmpty
    case _                => !steps.contains(p.PathStep.Deref)
  }

  private[pass] def addressOfOwn(refTo: p.Expr.RefTo): Option[p.Named] = refTo match {
    case p.Expr.RefTo(p.Term.Select(root, steps, _), idx, comp, _, _) =>
      (root.tpe, comp) match {
        case (p.Type.Ptr(pointee, _), p.Type.Ptr(other, _)) =>
          Option.when(steps.isEmpty && idx.isEmpty && pointee == other)(root)
        case (p.Type.Ptr(_, _), _) => None
        case _                     => Some(root)
      }
    case _ => None
  }

  private[pass] case class Prepared(entry: p.Function, prologue: List[p.Stmt], body: List[p.Stmt], fresh: Fresh)
}

private[pass] final class Fresh(taken: mutable.Set[String]) {
  private var counter = 0L
  def apply(prefix: String, tpe: p.Type): p.Named = {
    var symbol = s"$prefix$counter"
    counter += 1
    while (!taken.add(symbol)) {
      symbol = s"$prefix$counter"
      counter += 1
    }
    p.Named(symbol, tpe)
  }
}
