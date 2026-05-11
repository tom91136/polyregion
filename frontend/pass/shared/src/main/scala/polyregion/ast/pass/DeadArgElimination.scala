package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

object DeadArgElimination extends ProgramPass {

  private def referencedRoots(f: p.Function): Set[p.Named] =
    f.body.flatMap { s =>
      s.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root }
    }.toSet

  private def cleanModuleCaptures(f: p.Function): p.Function =
    f.copy(moduleCaptures = f.moduleCaptures.filter(arg => referencedRoots(f).contains(arg.named)))

  private def cleanEntryFully(f: p.Function): p.Function = {
    val refs = referencedRoots(f)
    f.copy(
      receiver = f.receiver.filter(arg => refs.contains(arg.named)),
      moduleCaptures = f.moduleCaptures.filter(arg => refs.contains(arg.named)),
      termCaptures = f.termCaptures.filter(arg => refs.contains(arg.named))
    )
  }

  override def apply(program: p.Program, log: Log): p.Program =
    program.copy(
      entry = cleanEntryFully(program.entry),
      functions = program.functions.map(cleanModuleCaptures)
    )

}
