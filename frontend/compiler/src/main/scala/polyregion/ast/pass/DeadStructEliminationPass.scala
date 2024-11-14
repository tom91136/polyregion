package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{ScalaSRR as p, *}

object DeadStructEliminationPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program =
    // program.entry.args

    // val selects = program.entry.body.flatMap { s =>
    //    s.accTerm(s => s :: Nil, _ => Nil)
    // }

    // program.entry.body.arg

    // (program, log)

    // (p.Program(run(program.entry), program.functions.map(run(_)), program.defs), log)
    ???

}
