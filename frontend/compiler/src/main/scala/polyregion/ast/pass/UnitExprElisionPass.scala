package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

object UnitExprElisionPass extends ProgramPass {

  private def run(f: p.Function) = f.copy(body =
    f.body.flatMap(s =>
      s.map(_.map {
        case p.Stmt.Var(p.Named(_, p.Type.Unit), Some(p.Expr.Alias(p.Term.UnitConst)) | None) => Nil
        case x                                                                                => x :: Nil
      })
    )
  )

  override def apply(program: p.Program, log: Log): (p.Program, Log) =
    (p.Program(run(program.entry), program.functions.map(run(_)), program.defs), log)

}
