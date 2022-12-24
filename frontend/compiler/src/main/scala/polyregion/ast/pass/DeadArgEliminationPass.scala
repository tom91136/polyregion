package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, given, *}
import polyregion.ast.Traversal.*


object DeadArgEliminationPass extends ProgramPass {

  inline def run(f: p.Function): p.Function = {

    val topLevelRefs = f.body.flatMap { s =>
      s.collect[p.Term].collect {
        case p.Term.Select(Nil, x)      => x
        case p.Term.Select(x :: Nil, _) => x
      }
    }.toSet

    // println(s"aaa = ${topLevelRefs.toList.mkString("\n")}")
    // pprint.pprintln(f.body)

    f.copy(
      receiver = f.receiver.filter(topLevelRefs.contains(_)),
      args = f.args.filter(topLevelRefs.contains(_)),
      captures = f.captures.filter(topLevelRefs.contains(_))
    )
  }

  override def apply(program: p.Program, log: Log): (p.Program, Log) =
    (program.copy(entry = run(program.entry)), log)

}
