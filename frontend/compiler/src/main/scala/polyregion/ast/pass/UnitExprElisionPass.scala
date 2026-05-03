package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

object UnitExprElisionPass extends ProgramPass {

  // An expression that has no observable effects when evaluated -- safe to drop
  // when its result (a Unit value) is unused.
  private def isSideEffectFree(e: p.Expr): Boolean = e match {
    case p.Expr.Unit0Const         => true
    case p.Expr.Select(_, _)       => true
    case p.Expr.Annotated(x, _, _) => isSideEffectFree(x)
    case _                         => false
  }

  private def run(f: p.Function) = f
    .modifyAll[p.Stmt] {
      case s @ p.Stmt.Var(p.Named(_, p.Type.Unit0), None) =>
        p.Stmt.Comment(s"removed unit expr ${s.repr}")
      case s @ p.Stmt.Var(p.Named(_, p.Type.Unit0), Some(t)) if t.tpe == p.Type.Unit0 && isSideEffectFree(t) =>
        p.Stmt.Comment(s"removed unit expr ${s.repr}")
      case x => x
    }
    .modifyAll[p.Expr] {
      case s @ p.Expr.Select(_, _) if s.tpe == p.Type.Unit0 => p.Expr.Unit0Const
      case x                                                => x
    }

  override def apply(program: p.Program, log: Log): p.Program = {
    println(">UnitExprElisionPass")
    p.Program(run(program.entry), program.functions.map(run(_)), program.defs)
  }

}
