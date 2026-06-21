package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

// monomorphises StructDefs the way Specialisation does functions: one concrete StructDef per distinct
// applied-type-args use, members type-substituted, parents and every Type reference rewritten to the
// zero-arg monomorphic struct. Returns the monomorphic-name -> original-name map (the boundary value)
// examples:
//   Box[A]{A v}; uses Box[Int], Box[Float]   ->  Box_Int{Int v}, Box_Float{Float v}  // + map *_Int->Box etc.
//   Pair[A]{Box[A] b}; uses Pair[Int]        ->  Pair_Int{Box_Int b}                 // member Box[Int] -> Box_Int
//   Dog[A] extends Animal[A]; uses Dog[Int]  ->  Dog_Int extends Animal_Int          // parent -> monomorphic
// edge cases:
//   doReplacement lookup -> match a use by its original key, then by already-renamed args, then
//                           structurally (same name + same recursively-replaced args) for a nested use
object MonoStruct extends BoundaryPass[Map[p.Sym, p.Sym]] {

  override def phase = p.PassPhase.PostMono

  override def apply(program: p.Program, log: Log): (Map[p.Sym, p.Sym], p.Program) = {

    val structsInFunction: List[p.Type.Struct] =
      (program.entry :: program.functions)
        .flatMap(_.collectWhere[p.Type] { case s: p.Type.Struct => s })
        .distinct

    val sdefByName = program.defs.map(d => d.name -> d).toMap

    log.info("uses", structsInFunction.map(_.repr)*)
    log.info("defs", program.defs.map(_.repr)*)
    val monoStructDefs = for {
      sdef   <- program.defs
      struct <- structsInFunction
      if struct.name == sdef.name
      table = sdef.tpeVars.zip(struct.args).toMap
    } yield struct -> p.StructDef(
      name = p.Sym(struct.monomorphicName),
      tpeVars = Nil,
      members = sdef.members.modifyAll[p.Type](
        _.mapLeaf {
          case p.Type.Var(name) => table(name)
          case x                => x
        }
      ),
      parents = sdef.parents
    )

    val nameTable = monoStructDefs.map((s, sdef) => s.name -> sdef.name).toMap
    val replacementTable =
      monoStructDefs
        .map((s, sdef) =>
          s -> sdef.copy(parents = sdef.parents.flatMap { parent =>
            nameTable.get(parent.name).map(newName => p.Type.Struct(newName, Nil))
          })
        )
        .toMap

    log.info("rename table", replacementTable.map((k, v) => s"${k.repr} => ${v.repr}").toSeq*)

    def doReplacement(t: p.Type): p.Type = t match {
      case s @ p.Type.Struct(name, args) =>
        val newArgs                        = args.map(doReplacement(_))
        val withRenamedArgs: p.Type.Struct = p.Type.Struct(name, newArgs)
        val byOriginal                     = replacementTable.get(s)
        val byRenamed                      = replacementTable.get(withRenamedArgs)
        val byName =
          if (byOriginal.isDefined || byRenamed.isDefined) None
          else
            replacementTable.collectFirst {
              case (key, sdef)
                  if key.name == name && key.args.size == newArgs.size &&
                    key.args.map(doReplacement) == newArgs =>
                sdef
            }
        byOriginal.orElse(byRenamed).orElse(byName) match {
          case Some(sdef) => p.Type.Struct(sdef.name, Nil)
          case None       => withRenamedArgs
        }
      case p.Type.Ptr(c, s)    => p.Type.Ptr(doReplacement(c), s)
      case p.Type.Arr(c, l, s) => p.Type.Arr(doReplacement(c), l, s)
      case a                   => a
    }

    val rootStructDefs = monoStructDefs
      .map(_._2)
      .map(s => s.copy(members = s.members.modifyAll[p.Type](doReplacement(_))))

    val referencedStructDefs = rootStructDefs
      .collectWhere[p.Type] { t =>
        def findLeafStructDefs(t: p.Type): List[p.StructDef] = t match {
          case p.Type.Struct(name, xs) =>
            xs.flatMap(findLeafStructDefs(_)) ::: program.defs.filter(_.name == name)
          case p.Type.Ptr(component, _)    => findLeafStructDefs(component)
          case p.Type.Arr(component, _, _) => findLeafStructDefs(component)
          case _                           => Nil
        }
        findLeafStructDefs(t)
      }
      .flatten

    (
      replacementTable.map((struct, sdef) => sdef.name -> struct.name),
      program.copy(
        entry = program.entry.modifyAll[p.Type](doReplacement(_)),
        functions = program.functions.map(_.modifyAll[p.Type](doReplacement(_))),
        defs = rootStructDefs ++ referencedStructDefs,
        phase = p.PassPhase.PostMono
      )
    )

  }

}
