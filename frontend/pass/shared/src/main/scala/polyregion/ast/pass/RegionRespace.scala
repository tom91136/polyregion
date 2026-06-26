package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// marks each rooted pointer's space from the root it ultimately addresses
// a pointer declared in a space different to its root forces a provenance-stripping addrspacecast that
// miscompiles on logical SPIR-V; the fix for the drift `Verify.validateRegionSpaces` reports
// examples:
//   s = &local[i], s Global, local Local  ->  s re-stamped Local
//   p = &g[i]; q = &p[j]  (g Global)      ->  unchanged
object RegionRespace extends ProgramPass {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private def run(f: p.Function): (p.Function, Int) = {
    val respace: Map[String, p.Type.Space] =
      Provenance.spaceMismatches(f).map { case (n, _, _, sr) => n.symbol -> sr }.toMap
    if (respace.isEmpty) (f, 0)
    else {
      def reN(n: p.Named): p.Named = respace.get(n.symbol).fold(n)(s => n.copy(tpe = Provenance.withSpace(n.tpe, s)))
      def reE(e: p.Expr, s: p.Type.Space): p.Expr = e match {
        case p.Expr.Cast(from, as)                   => p.Expr.Cast(from, Provenance.withSpace(as, s))
        case p.Expr.RefTo(lhs, idx, comp, _, region) => p.Expr.RefTo(lhs, idx, comp, s, region)
        case other                                   => other
      }
      val rooted = f
        .modifyAll[p.Term] {
          // bare use re-types its own slot; a stepped use keeps the leaf type
          case p.Term.Select(n, Nil, _) if respace.contains(n.symbol) => val rn = reN(n); p.Term.Select(rn, Nil, rn.tpe)
          case p.Term.Select(n, steps, t) if respace.contains(n.symbol) => p.Term.Select(reN(n), steps, t)
          case t                                                        => t
        }
        .modifyAll[p.Stmt] {
          case p.Stmt.Var(n, e, m) if respace.contains(n.symbol) =>
            p.Stmt.Var(reN(n), e.map(reE(_, respace(n.symbol))), m)
          case p.Stmt.Mut(p.Term.Select(n, Nil, t), e) if respace.contains(n.symbol) =>
            val s = respace(n.symbol); p.Stmt.Mut(p.Term.Select(reN(n), Nil, Provenance.withSpace(t, s)), reE(e, s))
          case s => s
        }
      (rooted, respace.size)
    }
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val (entry, ec)      = run(program.entry)
    val (functions, fcs) = program.functions.map(run).unzip
    val total            = ec + fcs.sum
    if (total > 0) log.info(s"respaced $total rooted pointer(s) to their resource's address space")
    program.copy(entry = entry, functions = functions)
  }
}
