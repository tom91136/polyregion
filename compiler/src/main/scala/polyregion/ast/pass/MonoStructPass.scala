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
      name  = p.Sym(struct.monomorphicName),
    } yield struct -> p.StructDef(
      name = name,
      tpeVars = Nil,
      members = sdef.members.map(_.mapType {
        case p.Type.Var(name) => table(name)
        case x                => x
      })
    )

    val replacementTable = monoStructDefs.toMap

    val body = program.entry.body.flatMap(s =>
      s.mapType {
        case s: p.Type.Struct => replacementTable.get(s).map(x => p.Type.Struct(x.name, Nil, Nil)).getOrElse(s)
        case x                => x
      }
    )

    (p.Program(program.entry.copy(body = body), Nil, monoStructDefs.map(_._2)), log)

  }

}
