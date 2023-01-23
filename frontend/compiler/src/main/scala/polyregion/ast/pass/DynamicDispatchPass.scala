package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}

object DynamicDispatchPass extends ProgramPass {

  inline def run(f: p.Function): p.Function = {
    // TODO 
    f
  }

  override def apply(program: p.Program, log: Log): p.Program =
    program.copy(entry = run(program.entry))

}
