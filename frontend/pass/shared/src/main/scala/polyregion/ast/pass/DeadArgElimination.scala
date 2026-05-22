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

  // XXX The entry's params are the kernel ABI seen by the JVM macro, which packs fnValues
  // before this pass runs. Dropping captures here desyncs host marshalling from the kernel.
  override def apply(program: p.Program, log: Log): p.Program =
    program.copy(functions = program.functions.map(cleanModuleCaptures))

}
