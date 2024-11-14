package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{ScalaSRR as p, *, given}

object DeadArgEliminationPass extends ProgramPass {

  inline def run(f: p.Function): p.Function = {
    val topLevelRefs = f.body.flatMap { s =>
      s.collectWhere[p.Term] {
        case p.Term.Select(Nil, x)    => x
        case p.Term.Select(x :: _, _) => x
      }
    }.toSet
    f.copy(
      receiver = f.receiver.filter(arg => topLevelRefs.contains(arg.named)),
      args = f.args.filter(arg => topLevelRefs.contains(arg.named)),
      moduleCaptures = f.moduleCaptures.filter(arg => topLevelRefs.contains(arg.named)),
      termCaptures = f.termCaptures.filter(arg => topLevelRefs.contains(arg.named))
    )
  }

  override def apply(program: p.Program, log: Log): (p.Program) =
    program.copy(entry = run(program.entry))

}
