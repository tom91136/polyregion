package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

import scala.annotation.tailrec

object DeadStructEliminationPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {

    val roots: Set[p.Sym] =
      (program.entry :: program.functions)
        .flatMap(_.collectWhere[p.Type] { case s: p.Type.Struct => s.name })
        .toSet

    val byName = program.defs.map(d => d.name -> d).toMap

    @tailrec def reach(frontier: Set[p.Sym], live: Set[p.Sym]): Set[p.Sym] = {
      val next = frontier.flatMap { s =>
        byName.get(s).toList.flatMap { d =>
          d.parents.map(_.name) ++ d.members.flatMap(
            _.tpe.collectWhere[p.Type] { case ts: p.Type.Struct => ts.name }
          )
        }
      } -- live
      if (next.isEmpty) live else reach(next, live ++ next)
    }

    val live = reach(roots, roots)
    log.info("kept", live.toSeq.map(_.repr).sorted*)
    log.info("dropped", program.defs.map(_.name).filterNot(live.contains).map(_.repr).sorted*)
    program.copy(defs = program.defs.filter(d => live.contains(d.name)))
  }

}
