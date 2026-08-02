package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

import scala.annotation.tailrec

// drops Functions unreachable from the entry or an Exported one, following Type.FnRef
// examples:
//   exported a -> shared; orphan        ->  a, shared    (orphan reached by neither)
//   exported a, b both -> shared        ->  a, b, shared
//   of those, only a left Exported      ->  a, shared    (b dropped; the seed narrows with visibility)
//   exported a -> b -> a                ->  a, b         (cycle terminates, revisits are subtracted)
// edge cases:
//   entry is always a root              ->  a library's synthetic root reaches nothing, but a kernel entry
//                                           keeps its callees, so the pass is safe in any pipeline
//   name-keyed, not signature-keyed     ->  every overload of a reached name survives for Specialisation to pick
object DeadFunctionElimination extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {
    val seeds: Set[p.Sym] =
      program.functions.filter(_.visibility == p.Function.Visibility.Exported).map(_.name).toSet ++
        program.entry.collectWhere[p.Type] { case f: p.Type.FnRef => f.name }

    val byName = program.functions.groupBy(_.name)

    @tailrec def reach(frontier: Set[p.Sym], live: Set[p.Sym]): Set[p.Sym] = {
      val next = frontier.flatMap { s =>
        byName.getOrElse(s, Nil).flatMap(_.collectWhere[p.Type] { case f: p.Type.FnRef => f.name })
      } -- live
      if (next.isEmpty) live else reach(next, live ++ next)
    }

    val live = reach(seeds, seeds)
    log.info("kept", live.toSeq.map(_.repr).sorted*)
    log.info("dropped", program.functions.map(_.name).filterNot(live.contains).map(_.repr).sorted*)
    program.copy(functions = program.functions.filter(f => live.contains(f.name)))
  }

}
