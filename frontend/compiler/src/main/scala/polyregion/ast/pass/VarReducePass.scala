package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, given, *}
import polyregion.ast.Traversal.*

object VarReducePass extends ProgramPass {

  private def run(f: p.Function) = {

    // Remove intermediate assignments:
    // var a: T = expr
    // var b: T = a
    // use b

    // var a: T = expr
    // use a

    val allVars = f.collectWhere[p.Stmt] { case v @ p.Stmt.Var(_, _) => v }

    val reduced = allVars.foldLeft(f.body) { case (stmts, source @ p.Stmt.Var(name, _)) =>
      val SourceName = p.Term.Select(Nil, name)
      val allSelectToSource = f.body.collectWhere[p.Term] {
        case x @ SourceName                    => x
        case x @ p.Term.Select(`name` :: _, _) => x
      }
      allSelectToSource match {
        case _ :: Nil =>
          // find vars with RHS to source
          val candidates = f.body.collectWhere[p.Stmt] { case r @ p.Stmt.Var(_, Some(p.Expr.Alias(SourceName))) => r }
          candidates match {
            case alias :: Nil => // single, inline rhs and delete
              stmts.modifyAll[p.Stmt] {
                case `source` => p.Stmt.Comment(s"${source}") // source is only used here, delete
                case `alias`  => alias.copy(expr = source.expr)
                case x        => x
              }
            case _ => stmts
          }
        case _ => stmts
      }
    }

    f.copy(body = reduced)
  }

  override def apply(program: p.Program, log: Log): (p.Program, Log) = {
    println(">VarReducePass")
    (p.Program(run(program.entry), program.functions.map(run(_)), program.defs), log)
  }

}
