package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// drops Mut statements that assign through a #empty_struct_storage padding field
// examples:
//   s.#empty_struct_storage = 0; f()  ->  f()
//   s.x.#empty_struct_storage = 0     ->  (dropped: last step is the padding field)
//   s.x = 0                           ->  s.x = 0   // kept: not a padding field
// edge cases:
//   only the last path step is checked  ->  a #empty_struct_storage hop mid-path does not match
object EmptyStorageElision extends ProgramPass {

  private def isPaddingMut(s: p.Stmt): Boolean = s match {
    case p.Stmt.Mut(p.Term.Select(_, steps, _), _) =>
      steps.lastOption.exists {
        case p.PathStep.Field(f) => f == p.Conventions.EmptyStructStorageField; case _ => false
      }
    case _ => false
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    def run(f: p.Function): p.Function =
      if (!f.collectAll[p.Stmt].exists(isPaddingMut)) f
      else f.copy(body = mapStmtsRec(f.body)(s => if (isPaddingMut(s)) Nil else List(s)))
    program.copy(entry = run(program.entry), functions = program.functions.map(run))
  }
}
