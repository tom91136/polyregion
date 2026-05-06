package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

object VarReducePass extends ProgramPass {

  private def run(f: p.Function, log: Log) = {
    // Remove intermediate assignments, so the following:
    //    var a: T = expr
    //    var b: T = a
    //    f(b)
    // becomes
    //    var a: T = expr
    //    f(a)
    val (n, reduced) = doUntilNotEq(f) { (_, f) =>
      f.collectFirst_[p.Stmt] {
        // Find the first var declaration that points to an alias
        case source @ p.Stmt.Var(name, Some(that: p.Expr.Select)) => (source, name, that)
      } match {
        case Some((source, name, that)) =>
          // Then  replace all references to that var with the alias itself
          val modified = f.modifyAll[p.Expr] {
            case x @ p.Expr.Select(`name` :: ys, y) => p.Expr.Select(that.init ::: that.last :: ys, y)
            case x @ p.Expr.Select(Nil, `name`)     => that
            case x                                  => x
          }
          // Whether or not any reference was rewritten, the rhs is a pure Select so an unused
          // alias is a dead store either way; drop the source var declaration in both cases.
          log.info(s"Delete `${source.repr}`")
          modified.modifyAll[p.Stmt] {
            case `source` => p.Stmt.Comment(s"[VarReducePass] ${source.repr}")
            case x        => x
          }
        case None => f
      }
    }
    log.info(s"Var reduction is stable after ${n} passes")
    reduced
  }

  override def apply(program: p.Program, log: Log): p.Program =
    p.Program(
      run(program.entry, log.subLog(s"VarReducePass on ${program.entry.name}")),
      program.functions.map(f => run(f, log.subLog(s"VarReducePass on ${f.name}"))),
      program.defs
    )

}
