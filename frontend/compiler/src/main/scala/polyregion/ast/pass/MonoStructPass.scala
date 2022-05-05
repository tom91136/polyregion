package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

object MonoStructPass {

  def run(program: p.Program)(log: Log): (p.Program, Log) = {

    val structs = program.entry.body.flatMap(_.accType {
      case s: p.Type.Struct => s :: Nil
      case x                => Nil
    })

    val monoStructDefs = for {
      sdef   <- program.defs
      struct <- structs.distinct
      if struct.name == sdef.name
      table = struct.tpeVars.zip(struct.args).toMap
      name  = p.Sym(struct.monomorphicName)
    } yield struct -> p.StructDef(
      name = name,
      tpeVars = Nil,
      members = sdef.members.map(_.mapType {
        case p.Type.Var(name) => table(name)
        case x                => x
      })
    )

    val replacementTable = monoStructDefs.toMap

    def doReplacement(t: p.Type) = t match {
      case s: p.Type.Struct => replacementTable.get(s).map(x => p.Type.Struct(x.name, Nil, Nil)).getOrElse(s)
      case x                => x
    }

    val body     = program.entry.body.flatMap(_.mapType(doReplacement(_)))
    val args     = program.entry.args.map(_.mapType(doReplacement(_)))
    val receiver = program.entry.receiver.map(_.mapType(doReplacement(_)))
    val captures = program.entry.captures.map(_.mapType(doReplacement(_)))
    

    (
      p.Program(
        entry = program.entry.copy(
          body = body,
          // receiver = receiver,
          captures = captures
        ),
        functions = Nil,
        defs = monoStructDefs.map(_._2)
      ),
      log
    )

  }

}
