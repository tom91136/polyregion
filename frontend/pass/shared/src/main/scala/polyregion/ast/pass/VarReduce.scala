package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// copy-propagates immutable val-aliases (`val b = a`) to a fixed point: forwards `a`'s root+steps into
// every Select of `b`, then drops the alias decl
// edge cases:
//   either side later mutated -> not propagated; forwarding spans the whole function, so a `Mut` on
//                                `a` after the alias would make a `b` read see the post-mutation value
object VarReduce extends ProgramPass {

  private def filterStmts(body: List[p.Stmt])(drop: Set[p.Stmt]): List[p.Stmt] = body.flatMap {
    case s if drop.contains(s)             => Nil
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, filterStmts(t)(drop), filterStmts(f)(drop)) :: Nil
    case p.Stmt.While(c, b)                => p.Stmt.While(c, filterStmts(b)(drop)) :: Nil
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, filterStmts(b)(drop)) :: Nil
    case p.Stmt.Annotated(inner, pos, c)   => filterStmts(inner :: Nil)(drop).map(p.Stmt.Annotated(_, pos, c))
    case other                             => other :: Nil
  }

  private def run(f: p.Function, log: Log) = {
    // mutatedNames is constant across iterations (Mut sites are never rewritten), so scan once
    val mutatedNames: Set[p.Named] =
      f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(name, _, _), _) => name }.toSet
    val (n, reduced) = doUntilNotEq(f) { (_, f) =>
      f.collectFirst_[p.Stmt] {
        case source @ p.Stmt.Var(name, Some(p.Expr.Alias(that: p.Term.Select)), false)
            if !mutatedNames.contains(name) && !mutatedNames.contains(that.root) =>
          (source, name, that)
      } match {
        case Some((source, name, that)) =>
          val modified = f.modifyAll[p.Term] {
            case p.Term.Select(`name`, ys, tpe) => p.Term.Select(that.root, that.steps ::: ys, tpe)
            case x                              => x
          }
          log.info(s"Delete `${source.repr}`")
          modified.copy(body = filterStmts(modified.body)(Set(source)))
        case None => f
      }
    }
    log.info(s"Var reduction is stable after ${n} passes")
    reduced
  }

  override def apply(program: p.Program, log: Log): p.Program =
    program.copy(
      entry = run(program.entry, log.subLog(s"VarReduce on ${program.entry.name}")),
      functions = program.functions.map(f => run(f, log.subLog(s"VarReduce on ${f.name}")))
    )

}
