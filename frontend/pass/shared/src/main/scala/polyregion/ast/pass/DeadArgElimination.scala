package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

// drops moduleCaptures never used as the root of a Select in the body
// examples:
//   fn f(cap m, cap n) { m.x }  ->  fn f(cap m) { m.x }     // n unreferenced
//   fn f(cap m) { use(m) }      ->  fn f(cap m) { use(m) }  // kept: bare m is Select(m, Nil)
// edge cases:
//   entry params untouched      ->  codegen already packed that ABI; dropping here breaks host mirroring
object DeadArgElimination extends ProgramPass {

  private def cleanModuleCaptures(f: p.Function): p.Function = {
    val roots = f.body.flatMap(_.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root }).toSet
    f.copy(moduleCaptures = f.moduleCaptures.filter(arg => roots.contains(arg.named)))
  }

  // XXX The entry's params are the kernel ABI seen by the JVM macro, which packs fnValues
  // before this pass runs. Dropping captures here desyncs host mirroring from the kernel.
  override def apply(program: p.Program, log: Log): p.Program =
    program.copy(functions = program.functions.map(cleanModuleCaptures))

}
