package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

object MonoStructPass extends BoundaryPass[Map[p.Sym, p.Sym]] {

  override def apply(program: p.Program, log: Log): (Map[p.Sym, p.Sym], p.Program) = {

    val structsInFunction: List[p.Type.Struct] =
      (program.entry :: program.functions)
        .flatMap(_.collectWhere[p.Type] { case s: p.Type.Struct => s })
        .distinct

    log.info("uses", structsInFunction.map(_.repr)*)
    log.info("defs", program.defs.map(_.repr)*)
    val monoStructDefs = for {
      sdef   <- program.defs
      struct <- structsInFunction // XXX this may be empty
      if struct.name == sdef.name
      table = struct.tpeVars.zip(struct.args).toMap
    } yield struct -> p.StructDef(
      name = p.Sym(struct.monomorphicName),
      tpeVars = Nil,
      members = sdef.members.modifyAll[p.Type](
        _.mapLeaf {
          case p.Type.Var(name) => table(name)
          case x                => x
        }
      ),
      parents = struct.parents
    )

    // Make sure we rename the parents too
    val nameTable = monoStructDefs.map((s, sdef) => s.name -> sdef.name).toMap
    val replacementTable =
      monoStructDefs.map((s, sdef) => s -> sdef.copy(parents = sdef.parents.flatMap(nameTable.get(_)))).toMap

    log.info("rename table", replacementTable.map((k, v) => s"${k.repr} => ${v.repr}").toSeq*)

    // do the replacement inside-out: recurse into args first, then look up. This way structs whose
    // args were captured pre-rename in structsInFunction (e.g. `ListBuffer[A=Float2_orig]` while
    // the table key has the same `ListBuffer[A=Float2_orig]` shape) resolve when called inside
    // a function body whose receiver type already reflects post-replacement args
    // (`ListBuffer[A=Float2_mono]`). Replacing args first canonicalises the key for lookup, then
    // we also try the post-replacement shape against the table built from pre-replacement keys.
    def doReplacement(t: p.Type): p.Type = t match {
      case s @ p.Type.Struct(name, tpeVars, args, parents) =>
        val newArgs                        = args.map(doReplacement(_))
        val withRenamedArgs: p.Type.Struct = p.Type.Struct(name, tpeVars, newArgs, parents)
        // First try the original shape (matches when the body still has pre-rename args).
        val byOriginal = replacementTable.get(s)
        // Then try the renamed shape (matches when the body already has post-rename args).
        val byRenamed = replacementTable.get(withRenamedArgs)
        // Finally try matching by name+arity, scanning the table for a keyed struct whose
        // post-replacement args equal newArgs (handles cases where the table key has yet
        // a different inner-arg shape but represents the same monomorphic instance).
        val byName =
          if (byOriginal.isDefined || byRenamed.isDefined) None
          else
            replacementTable.collectFirst {
              case (key, sdef)
                  if key.name == name && key.args.size == newArgs.size &&
                    key.args.map(applyTypeReplacement) == newArgs =>
                sdef
            }
        byOriginal.orElse(byRenamed).orElse(byName) match {
          case Some(sdef) => p.Type.Struct(sdef.name, Nil, Nil, sdef.parents)
          case None       => withRenamedArgs
        }
      case a @ p.Type.Ptr(c, l, s) => p.Type.Ptr(doReplacement(c), l, s)
      case a                       => a
    }
    def applyTypeReplacement(t: p.Type): p.Type = doReplacement(t)

    val rootStructDefs = monoStructDefs
      .map(_._2) // make sure we handle nested structs
      .map(s => s.copy(members = s.members.modifyAll[p.Type](doReplacement(_))))

    val referencedStructDefs = rootStructDefs
      .collectWhere[p.Type] { t =>
        def findLeafStructDefs(t: p.Type): List[p.StructDef] = t match {
          case p.Type.Struct(name, _, xs, _) =>
            xs.flatMap(findLeafStructDefs(_)) ::: program.defs.filter(_.name == name)
          case p.Type.Ptr(component, _, _) => findLeafStructDefs(component)
          case _                           => Nil
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
        functions = program.functions.map(_.modifyAll[p.Type](doReplacement(_))),
        defs = rootStructDefs ++ referencedStructDefs
      )
    )

  }

}
