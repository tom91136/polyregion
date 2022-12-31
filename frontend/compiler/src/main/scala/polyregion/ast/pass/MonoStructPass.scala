package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.prism.Prism

object MonoStructPass extends BoundaryPass[Map[p.Sym, p.Sym]] {

  override def apply(program: p.Program, log: Log): (Map[p.Sym, p.Sym], p.Program, Log) = {
    println(">MonoStructPass")

    val structsInFunction: List[p.Type.Struct] =
      program.entry
        .collectWhere[p.Type] { case s: p.Type.Struct => s }
        .distinct

    println(s"In f = ${structsInFunction} ${program.defs}")
    val monoStructDefs = for {
      sdef   <- program.defs
      struct <- structsInFunction // XXX this may be empty
      if struct.name == sdef.name
      table = struct.tpeVars.zip(struct.args).toMap
    } yield struct -> p.StructDef(
      name = p.Sym(struct.monomorphicName),
      reference = false,
      tpeVars = Nil,
      members = sdef.members.modifyAll[p.Type](_.mapLeaf {
        case p.Type.Var(name) => table(name)
        case x                => x
      })
    )

    val replacementTable = monoStructDefs.toMap

    println(s"[Rep] Table:\n${replacementTable.map((k, v) => s"\t${k.repr} => ${v.repr}").mkString("\n")}")

    // do the replacement outside in
    def doReplacement(t: p.Type): p.Type = t match {
      case s @ p.Type.Struct(name, tpeVars, args) =>
        println(s"[Rep] ${s.repr} => ${replacementTable.get(s)}")
        replacementTable.get(s) match {
          case Some(sdef) => p.Type.Struct(sdef.name, Nil, Nil)
          case None       => p.Type.Struct(name, tpeVars, args.map(doReplacement(_)))
        }
      case a @ p.Type.Array(c) => p.Type.Array(doReplacement(c))
      case a                   => a
    }

    val rootStructDefs = monoStructDefs
      .map(_._2) // make sure we handle nested structs
      .map(s => s.copy(members = s.members.modifyAll[p.Type](doReplacement(_))))

    val referencedStructDefs = rootStructDefs
      .collectWhere[p.Type] { t =>
        def findLeafStructDefs(t: p.Type): List[p.StructDef] = t match {
          case p.Type.Struct(name, _, xs) => xs.flatMap(findLeafStructDefs(_)) ::: program.defs.filter(_.name == name)
          case p.Type.Array(component)    => findLeafStructDefs(component)
          case _                          => Nil
        }
        findLeafStructDefs(t)
      }
      .flatten

    (
      // Create reverse lookup table for finding the original name of a monomorphic name.
      // This is required for associating the term prism of a mirror during pickling.
      replacementTable.map((struct, sdef) => sdef.name -> struct.name),
      p.Program(
        entry = program.entry.modifyAll[p.Type](doReplacement(_)),
        functions = program.functions,
        defs = rootStructDefs ++ referencedStructDefs
      ),
      log
    )

  }

}
