package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

object VarReducePass extends ProgramPass {

  private def run(f: p.Function) = {

    // Remove intermediate assignments:
    // var a: T = expr
    // var b: T = a
    // use b

    // var a: T = expr
    // use a

    val allVars = f.body.flatMap(x =>
      x.acc {
        case v @ p.Stmt.Var(_, _) => v :: Nil
        case _                    => Nil
      }
    )

    val reduced = allVars.foldLeft(f.body) { case (stmts, source @ p.Stmt.Var(name, _)) =>
      val SourceName = p.Term.Select(Nil, name)
      val allSelectToSource = f.body.flatMap(s =>
        s.accTerm(
          {
            case x @ SourceName => x :: Nil
            case x @ p.Term.Select(`name` :: _, _)  => x:: Nil
            case _              => Nil
          }
        )
      )
      allSelectToSource match {
        case _ :: Nil =>
          val candidates = f.body.flatMap(_.acc { // find vars with RHS to source
            case r @ p.Stmt.Var(_, Some(p.Expr.Alias(SourceName))) => r :: Nil
            case _                                                 => Nil
          })
          candidates match {
            case alias :: Nil => // single, inline rhs and delete
              stmts.flatMap(_.map {
                case `source` => Nil // source is only used here, delete
                case `alias`  => alias.copy(expr = source.expr) :: Nil
                case x        => x :: Nil
              })
            case _ => stmts
          }
        case _ => stmts
      }
    }

    f.copy(body = reduced)
  }

  override def apply(program: p.Program, log: Log): (p.Program, Log) =
    (p.Program(run(program.entry), program.functions.map(run(_)), program.defs), log)

}
