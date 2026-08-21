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

    val rootStructs: List[p.Type.Struct] =
      ((program.entry :: program.functions).flatMap(_.collectWhere[p.Type] { case s: p.Type.Struct => s }) ++
        program.defs
          .filter(_.tpeVars.isEmpty)
          .flatMap(_.collectWhere[p.Type] { case s: p.Type.Struct => s })).distinct

    val sdefByName = program.defs.map(d => d.name -> d).toMap

    def instantiate(struct: p.Type.Struct): Option[p.StructDef] = sdefByName.get(struct.name).map { sdef =>
      val table = sdef.tpeVars.zip(struct.args).toMap
      def subst(t: p.Type, env: Map[String, p.Type] = table): p.Type = t match {
        case variable @ p.Type.Var(name)     => env.getOrElse(name, variable)
        case p.Type.Struct(name, args)       => p.Type.Struct(name, args.map(subst(_, env)))
        case p.Type.Ptr(comp, space)         => p.Type.Ptr(subst(comp, env), space)
        case p.Type.Arr(comp, length, space) => p.Type.Arr(subst(comp, env), length, space)
        case p.Type.Exec(tpeVars, args, rtn) =>
          val nested = env -- tpeVars
          p.Type.Exec(tpeVars, args.map(subst(_, nested)), subst(rtn, nested))
        case x => x
      }
      def substStruct(struct: p.Type.Struct): p.Type.Struct = subst(struct) match {
        case result: p.Type.Struct => result
        case _ => throw AssertionError(s"struct substitution changed ${struct.repr} to a non-struct type")
      }
      p.StructDef(
        name = if sdef.tpeVars.isEmpty then sdef.name else p.Sym(struct.monomorphicName),
        tpeVars = Nil,
        members = sdef.members.map(member => member.copy(tpe = subst(member.tpe))),
        parents = sdef.parents.map(substStruct),
        isUnion = sdef.isUnion
      )
    }

    case class Pending(struct: p.Type.Struct, ancestry: Map[p.Sym, List[p.Type.Struct]])

    def typeComplexity(tpe: p.Type): Int = tpe match {
      case p.Type.Struct(_, args)    => 1 + args.map(typeComplexity).sum
      case p.Type.Ptr(comp, _)       => 1 + typeComplexity(comp)
      case p.Type.Arr(comp, _, _)    => 1 + typeComplexity(comp)
      case p.Type.Exec(_, args, rtn) => 1 + args.map(typeComplexity).sum + typeComplexity(rtn)
      case _                         => 1
    }

    @annotation.tailrec
    def closeStructs(
        pending: List[Pending],
        seen: Set[p.Type.Struct] = Set.empty,
        result: List[p.Type.Struct] = Nil
    ): List[p.Type.Struct] = pending match {
      case Nil                                        => result
      case Pending(struct, _) :: tail if seen(struct) => closeStructs(tail, seen, result)
      case Pending(struct, ancestry) :: tail =>
        val history = ancestry.getOrElse(struct.name, Nil)
        history.takeRight(2) match {
          case previousPrevious :: previous :: Nil
              if typeComplexity(struct) > typeComplexity(previous) &&
                typeComplexity(previous) > typeComplexity(previousPrevious) =>
            throw IllegalStateException(
              s"MonoStruct detected expanding polymorphic recursion in ${struct.name.repr}: " +
                s"${previousPrevious.repr} -> ${previous.repr} -> ${struct.repr}"
            )
          case _ => ()
        }
        val nestedAncestry = ancestry.updated(struct.name, history :+ struct)
        val dependencies = instantiate(struct).toList.flatMap(
          _.collectWhere[p.Type] { case dependency: p.Type.Struct => dependency }
        )
        closeStructs(tail ::: dependencies.map(Pending(_, nestedAncestry)), seen + struct, result :+ struct)
    }

    val structUses = closeStructs(rootStructs.map(Pending(_, Map.empty)))

    log.info("uses", structUses.map(_.repr)*)
    log.info("defs", program.defs.map(_.repr)*)
    val monoStructDefs   = structUses.flatMap(struct => instantiate(struct).map(struct -> _))
    val replacementTable = monoStructDefs.toMap

    log.info("rename table", replacementTable.map((k, v) => s"${k.repr} => ${v.repr}").toSeq*)

    val replacementsByName: Map[p.Sym, List[(p.Type.Struct, p.StructDef)]] = replacementTable.toList.groupBy(_._1.name)

    val replacementCache                 = scala.collection.mutable.HashMap.empty[p.Type, p.Type]
    def doReplacement(t: p.Type): p.Type = replacementCache.getOrElseUpdate(t, computeReplacement(t))
    def computeReplacement(t: p.Type): p.Type = t match {
      case s @ p.Type.Struct(name, args) =>
        val newArgs                        = args.map(doReplacement(_))
        val withRenamedArgs: p.Type.Struct = p.Type.Struct(name, newArgs)
        val byOriginal                     = replacementTable.get(s)
        val byRenamed                      = replacementTable.get(withRenamedArgs)
        val byName =
          if (byOriginal.isDefined || byRenamed.isDefined) None
          else
            replacementsByName.getOrElse(name, Nil).collectFirst {
              case (key, sdef) if key.args.size == newArgs.size && key.args.map(doReplacement) == newArgs =>
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
    def replaceStruct(struct: p.Type.Struct): p.Type.Struct = doReplacement(struct) match {
      case result: p.Type.Struct => result
      case _ => throw AssertionError(s"struct replacement changed ${struct.repr} to a non-struct type")
    }

    val rootStructDefs = monoStructDefs
      .map(_._2)
      .map(s =>
        s.copy(
          members = s.members.modifyAll[p.Type](doReplacement(_)),
          parents = s.parents.map(replaceStruct)
        )
      )

    val concreteStructDefs = program.defs.filter(_.tpeVars.isEmpty).map(_.modifyAll[p.Type](doReplacement(_)))

    (
      replacementTable.map((struct, sdef) => sdef.name -> struct.name),
      program.copy(
        entry = program.entry.modifyAll[p.Type](doReplacement(_)),
        functions = program.functions.map(_.modifyAll[p.Type](doReplacement(_))),
        defs = (rootStructDefs ++ concreteStructDefs).distinctBy(_.name),
        phase = p.PassPhase.PostMono
      )
    )

  }

}
