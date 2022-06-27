package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

object MonoStructPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): (p.Program, Log) = {

    val structInSignature =
      (program.entry.receiver.toList ::: program.entry.captures ::: program.entry.args).map(_.tpe).collect {
        case s: p.Type.Struct => s
      }
    val structsInBody = program.entry.body.flatMap(_.accType {
      case s: p.Type.Struct => s :: Nil
      case x                => Nil
    })

    val structInFunction = structInSignature ::: structsInBody

    println(s"In f = ${structInFunction} ${program.defs}")
    val monoStructDefs = for {
      sdef   <- program.defs
      struct <- structInFunction.distinct // XXX this may be empty
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

    def typeIsNotDeleted(t: p.Type) = t match {
      case s: p.Type.Struct => monoStructDefs.contains(s)
      case _                => true
    }

    val body = program.entry.body.flatMap(_.mapType(doReplacement(_)))

    val args     = program.entry.args.map(_.mapType(doReplacement(_)))     // .filter(x => typeIsNotDeleted(x.tpe))
    val receiver = program.entry.receiver.map(_.mapType(doReplacement(_))) // .filter(x => typeIsNotDeleted(x.tpe))
    val captures = program.entry.captures.map(_.mapType(doReplacement(_))) // .filter(x => typeIsNotDeleted(x.tpe))

    (
      p.Program(
        entry = program.entry.copy(
          body = body,
          args = args,
          receiver = receiver,
          captures = captures
        ),
        functions = Nil,
        defs = monoStructDefs.map(_._2)
      ),
      log
    )

  }

}
