package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, given, *}

object UnitExprElisionPass extends ProgramPass {

  private def run(f: p.Function) = f
    .modifyAll[p.Stmt] {
      case s @ p.Stmt.Var(p.Named(_, p.Type.Unit), None) =>
        p.Stmt.Comment(s"removed unit expr ${s.repr}")
      case s @ p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(t))) if t.tpe == p.Type.Unit =>
        p.Stmt.Comment(s"removed unit expr ${s.repr}")
      case x => x
    }
    .modifyAll[p.Term] {
      case s @ p.Term.Select(_, _) if s.tpe == p.Type.Unit => p.Term.UnitConst
      case x                                               => x
    }

  override def apply(program: p.Program, log: Log): (p.Program, Log) = {
    println(">UnitExprElisionPass")
    (p.Program(run(program.entry), program.functions.map(run(_)), program.defs), log)
  }

}
