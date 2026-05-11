package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

object UnitExprElision extends ProgramPass {

  // Drop a Stmt entirely if its RHS is a no-op unit expression *and* the name is unused.
  // Otherwise the Return/Mut elsewhere still references it and dropping breaks the IR.
  private def isUnusedUnit(s: p.Stmt, referenced: Set[p.Named]): Boolean = s match {
    case p.Stmt.Var(n @ p.Named(_, p.Type.Unit0), None, _)                  => !referenced.contains(n)
    case p.Stmt.Var(n @ p.Named(_, p.Type.Unit0), Some(p.Expr.Alias(_)), _) => !referenced.contains(n)
    case _                                                                  => false
  }

  private def filter(body: List[p.Stmt], referenced: Set[p.Named]): List[p.Stmt] = body.flatMap {
    case s if isUnusedUnit(s, referenced)  => Nil
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, filter(t, referenced), filter(f, referenced)) :: Nil
    case p.Stmt.While(c, b)                => p.Stmt.While(c, filter(b, referenced)) :: Nil
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, filter(b, referenced)) :: Nil
    case p.Stmt.Annotated(inner, pos, c)   => filter(inner :: Nil, referenced).map(p.Stmt.Annotated(_, pos, c))
    case other                             => other :: Nil
  }

  private def run(f: p.Function) = {
    val referenced: Set[p.Named] = f.body
      .flatMap(_.collectWhere[p.Term] { case p.Term.Select(root, _, _) =>
        root
      })
      .toSet
    f.copy(body = filter(f.body, referenced))
  }

  override def apply(program: p.Program, log: Log): p.Program =
    p.Program(run(program.entry), program.functions.map(run(_)), program.defs, program.phase)

}
