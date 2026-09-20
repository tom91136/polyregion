package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// marks each rooted pointer's space from the root it ultimately addresses
// a pointer declared in a space different to its inferred address forces a provenance-stripping addrspacecast that
// miscompiles on logical SPIR-V; the fix for the drift `Verify.validateRegionSpaces` reports
// examples:
//   s = &local[i], s Global, local Local  ->  s re-stamped Local
//   p = &g[i]; q = &p[j]  (g Global)      ->  unchanged
object RegionRespace extends ProgramPass {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private[pass] def run(
      program: p.Program,
      f: p.Function,
      requireSolved: Boolean = true,
      adaptPointerStores: Boolean = true
  ): (p.Function, Int) = {
    val solved   = AddressRefinement.solve(program, f)
    val analysis = if (requireSolved) solved.requireSolved else solved
    val declared = (
      f.receiver.iterator.map(_.named) ++ f.args.iterator.map(_.named) ++
        f.moduleCaptures.iterator.map(_.named) ++ f.termCaptures.iterator.map(_.named) ++
        f.collectAll[p.Stmt].iterator.collect { case p.Stmt.Var(n, _, _) => n }
    ).map(n => n.symbol -> n).toMap
    val respace: Map[String, p.Type.Space] = declared.iterator.flatMap { case (symbol, named) =>
      for {
        declaredSpace <- AddressRefinement.spaceOf(named.tpe)
        refinedSpace  <- analysis.refinedSpace(named)
        if declaredSpace != refinedSpace
      } yield symbol -> refinedSpace
    }.toMap
    def reN(n: p.Named): p.Named =
      respace.get(n.symbol).fold(n)(s => n.copy(tpe = AddressRefinement.withSpace(n.tpe, s)))
    def reT(t: p.Term, s: p.Type.Space): p.Term = t match {
      case select: p.Term.Select                => select.copy(tpe = AddressRefinement.withSpace(select.tpe, s))
      case p.Term.NullPtrConst(comp, _, region) => p.Term.NullPtrConst(comp, s, region)
      case other                                => other
    }
    def reE(e: p.Expr, s: p.Type.Space): p.Expr = e match {
      // When provenance repairs a pointer loaded from a generic aggregate field, repair the source term as
      // well as the result. Loading it as Global and only then casting to Local creates an AS0 -> AS3 round
      // trip which loses the field's real storage space and can crash NVPTX's address-space combiner.
      case p.Expr.Alias(from)                      => p.Expr.Alias(reT(from, s))
      case p.Expr.Cast(from, as)                   => p.Expr.Cast(reT(from, s), AddressRefinement.withSpace(as, s))
      case p.Expr.RefTo(lhs, idx, comp, _, region) => p.Expr.RefTo(reT(lhs, s), idx, comp, s, region)
      case other                                   => other
    }
    def reNullComparison(e: p.Expr): p.Expr = e match {
      case p.Expr.IntrOp(p.Intr.LogicEq(x, n: p.Term.NullPtrConst)) =>
        AddressRefinement.spaceOf(x.tpe).fold(e)(s => p.Expr.IntrOp(p.Intr.LogicEq(x, reT(n, s))))
      case p.Expr.IntrOp(p.Intr.LogicEq(n: p.Term.NullPtrConst, y)) =>
        AddressRefinement.spaceOf(y.tpe).fold(e)(s => p.Expr.IntrOp(p.Intr.LogicEq(reT(n, s), y)))
      case p.Expr.IntrOp(p.Intr.LogicNeq(x, n: p.Term.NullPtrConst)) =>
        AddressRefinement.spaceOf(x.tpe).fold(e)(s => p.Expr.IntrOp(p.Intr.LogicNeq(x, reT(n, s))))
      case p.Expr.IntrOp(p.Intr.LogicNeq(n: p.Term.NullPtrConst, y)) =>
        AddressRefinement.spaceOf(y.tpe).fold(e)(s => p.Expr.IntrOp(p.Intr.LogicNeq(reT(n, s), y)))
      case other => other
    }
    def rewriteStmt(stmt: p.Stmt): p.Stmt = stmt match {
      case p.Stmt.Var(n, e, m) if respace.contains(n.symbol) =>
        p.Stmt.Var(reN(n), e.map(reE(_, respace(n.symbol))), m)
      case p.Stmt.Mut(p.Term.Select(n, Nil, t), e) if respace.contains(n.symbol) =>
        val s = respace(n.symbol)
        p.Stmt.Mut(p.Term.Select(reN(n), Nil, AddressRefinement.withSpace(t, s)), reE(e, s))
      // Generic C++ aggregate fields and control-flow pointer merges retain their declared pointer space, while
      // individual writes may have proven Local/Constant provenance. Keep each assignment internally well-typed;
      // the backend converts to the storage slot's representation and later loads use the tracked provenance.
      case p.Stmt.Mut(lhs @ p.Term.Select(_, _, t), e) if adaptPointerStores =>
        (AddressRefinement.spaceOf(t), AddressRefinement.spaceOf(e.tpe)) match {
          case (Some(lhsSpace), Some(rhsSpace)) if lhsSpace != rhsSpace =>
            p.Stmt.Mut(lhs.copy(tpe = AddressRefinement.withSpace(t, rhsSpace)), e)
          case _ => p.Stmt.Mut(lhs, e)
        }
      case other => other
    }
    val prepared = f
      .modifyAll[p.Term] {
        // bare use re-types its own slot; a stepped use keeps the leaf type
        case p.Term.Select(n, Nil, _) if respace.contains(n.symbol)   => val rn = reN(n); p.Term.Select(rn, Nil, rn.tpe)
        case p.Term.Select(n, steps, t) if respace.contains(n.symbol) => p.Term.Select(reN(n), steps, t)
        case t                                                        => t
      }
      .modifyAll[p.Expr](reNullComparison)
    val rooted = prepared.copy(body = mapStmtsRec(prepared.body)(stmt => List(rewriteStmt(stmt))))
    (rooted, respace.size)
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val (entry, ec) = program.entry
      .map(run(program, _))
      .map((function, count) => Some(function) -> count)
      .getOrElse(None -> 0)
    val (functions, fcs) = program.functions.map(run(program, _)).unzip
    val total            = ec + fcs.sum
    if (total > 0) log.info(s"respaced $total rooted pointer(s) to their resource's address space")
    program.copy(entry = entry, functions = functions)
  }
}
