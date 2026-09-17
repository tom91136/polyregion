package polyregion.ast.pass

import scala.collection.mutable.ListBuffer

import polyregion.ast.Traversal.*
import polyregion.ast.{Log, PolyAST as p, *, given}

// chooses the source-language storage representation before rendering. erased callable members retain their ABI
// byte, empty aggregate members are elided, and address-taken scalar values are materialised as private lvalues.
// an elided empty base is represented by its enclosing object's address followed by an explicit pointer Cast
// examples:
//   struct Capture { FnRef f }  ->  struct Capture { u8 f }
//   p = &1                     ->  tmp = 1; p = &tmp
//   &derived.#base_empty       ->  (Empty*)&derived
// edge cases:
//   elideRecursivelyEmptyAggregates  ->  also elides aggregates whose complete member tree is empty (required by Metal)
//   ForeignCall and diagnostic source names  ->  unaffected
final case class SourceStorageLegalise(elideRecursivelyEmptyAggregates: Boolean = false) extends ProgramPass
    derives PassArgCodec {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  override def apply(program: p.Program, log: Log): p.Program = {
    val definitions   = program.defs.map(definition => definition.name -> definition).toMap
    val directlyEmpty = program.defs.collect { case definition if definition.members.isEmpty => definition.name }.toSet
    val empty =
      if (!elideRecursivelyEmptyAggregates) directlyEmpty
      else {
        def containsOnlyEmptyMembers(definition: p.StructDef, known: Set[p.Sym]): Boolean =
          definition.members.forall {
            case p.Named(_, p.Type.Struct(name, _), _) => known(name)
            case _                                     => false
          }

        @annotation.tailrec
        def close(known: Set[p.Sym]): Set[p.Sym] = {
          val refined = known ++ program.defs.filter(containsOnlyEmptyMembers(_, known)).map(_.name)
          if (refined == known) known else close(refined)
        }
        close(directlyEmpty)
      }

    def emptyMember(member: p.Named): Boolean = member.tpe match {
      case p.Type.Struct(name, _) => empty(name)
      case _                      => false
    }

    def selectedType(root: p.Type, steps: List[p.PathStep]): p.Type = steps.foldLeft(root) {
      case (current, p.PathStep.Field(field)) =>
        val structure = current match {
          case value: p.Type.Struct                => value
          case p.Type.Ptr(value: p.Type.Struct, _) => value
          case other => throw IllegalArgumentException(s"field $field selected on non-struct type ${other.repr}")
        }
        definitions
          .get(structure.name)
          .flatMap(_.members.find(_.symbol == field))
          .map(_.tpe)
          .getOrElse(throw IllegalArgumentException(s"field $field not found on ${structure.name.fqcn}"))
      case (p.Type.Ptr(component, _), p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn) => component
      case (p.Type.Arr(component, _, _), _: p.PathStep.Index | _: p.PathStep.IndexDyn)                 => component
      case (current, step) => throw IllegalArgumentException(s"cannot apply ${step.repr} to ${current.repr}")
    }

    def trailingElidedBase(select: p.Term.Select): Option[(List[p.PathStep], p.Type)] = {
      val prefixes = select.steps.indices.map(i => select.steps.take(i) -> select.steps(i)).toList
      val flags = prefixes.map {
        case (prefix, p.PathStep.Field(field)) if field.startsWith(p.Conventions.BaseFieldPrefix) =>
          selectedType(select.root.tpe, prefix :+ p.PathStep.Field(field)) match {
            case p.Type.Struct(name, _) => empty(name)
            case _                      => false
          }
        case _ => false
      }
      val cut = flags.lastIndexWhere(!_)
      Option.when(flags.nonEmpty && flags.last)(
        select.steps.take(cut + 1) -> selectedType(select.root.tpe, select.steps.take(cut + 1))
      )
    }

    def lvalueSpace(select: p.Term.Select, indexed: Boolean): p.Type.Space = {
      val initialSpace =
        if (indexed && select.steps.isEmpty)
          select.root.tpe match {
            case p.Type.Ptr(_, space)    => space
            case p.Type.Arr(_, _, space) => space
            case _                       => p.Type.Space.Private
          }
        else p.Type.Space.Private
      select.steps
        .foldLeft(select.root.tpe -> initialSpace) {
          case ((current, space), p.PathStep.Field(field)) =>
            val (owner, ownerSpace) = current match {
              case p.Type.Ptr(component, value) => component -> value
              case _                            => current   -> space
            }
            selectedType(owner, List(p.PathStep.Field(field))) -> ownerSpace
          case ((current, space), p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn) =>
            current match {
              case p.Type.Ptr(component, value)    => component -> value
              case p.Type.Arr(component, _, value) => component -> value
              case _                               => current   -> space
            }
        }
        ._2
    }

    def legalise(function: p.Function): p.Function = {
      val occupied = scala.collection.mutable.Set.from(
        (function.receiver.toList ::: function.args ::: function.moduleCaptures ::: function.termCaptures)
          .map(_.named.symbol) :::
          function.collectAll[p.Stmt].flatMap {
            case p.Stmt.Var(name, _, _)                 => List(name.symbol)
            case p.Stmt.ForRange(induction, _, _, _, _) => List(induction.symbol)
            case _                                      => Nil
          }
      )
      val candidates = Iterator.from(0).map(index => s"#source_storage_$index")
      def fresh(tpe: p.Type): p.Named = {
        val symbol = candidates.find(occupied.add).get
        p.Named(symbol, tpe)
      }
      def select(name: p.Named): p.Term.Select = p.Term.Select(name, Nil, name.tpe)

      def lowerExpr(expression: p.Expr, prefix: ListBuffer[p.Stmt]): p.Expr = expression match {
        case ref @ p.Expr.RefTo(lhs: p.Term.Select, index, comp, _, region) =>
          val space = lvalueSpace(lhs, index.nonEmpty)
          trailingElidedBase(lhs) match {
            case Some((steps, parentType)) =>
              val parent    = lhs.copy(steps = steps, tpe = parentType)
              val pointer   = p.Type.Ptr(parentType, space)
              val temporary = fresh(pointer)
              prefix += p.Stmt.Var(temporary, Some(p.Expr.RefTo(parent, None, parentType, space, region)))
              p.Expr.Cast(select(temporary), p.Type.Ptr(comp, space))
            case None => ref.copy(space = space, region = p.Region.Rooted(lhs.root))
          }
        case ref @ p.Expr.RefTo(lhs, _, comp, _, _)
            if !lhs.isInstanceOf[p.Term.Select] &&
              !lhs.isInstanceOf[p.Term.StringConst] && !lhs.tpe.isInstanceOf[p.Type.Ptr] && !lhs.tpe
                .isInstanceOf[p.Type.Arr] =>
          val temporary = fresh(comp)
          prefix += p.Stmt.Var(temporary, Some(p.Expr.Alias(lhs)), isMutable = true)
          ref.copy(lhs = select(temporary), space = p.Type.Space.Private, region = p.Region.Rooted(temporary))
        case other => other
      }

      def lowerBlock(body: List[p.Stmt]): List[p.Stmt] = body.flatMap {
        case variable @ p.Stmt.Var(_, expression, _) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = expression.map(lowerExpr(_, prefix))
          prefix.toList :+ variable.copy(expr = lowered)
        case mutation @ p.Stmt.Mut(_, expression) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerExpr(expression, prefix)
          prefix.toList :+ mutation.copy(expr = lowered)
        case returned @ p.Stmt.Return(value) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerExpr(value, prefix)
          prefix.toList :+ returned.copy(value = lowered)
        case loop @ p.Stmt.While(_, body)             => List(loop.copy(body = lowerBlock(body)))
        case loop @ p.Stmt.ForRange(_, _, _, _, body) => List(loop.copy(body = lowerBlock(body)))
        case branch @ p.Stmt.Cond(_, trueBr, falseBr) =>
          List(branch.copy(trueBr = lowerBlock(trueBr), falseBr = lowerBlock(falseBr)))
        case annotated @ p.Stmt.Annotated(inner, _, _) =>
          lowerBlock(List(inner)) match {
            case init :+ last => init :+ annotated.copy(inner = last)
            case Nil          => Nil
            case other        => other
          }
        case tried @ p.Stmt.Try(body, handlers, fin) =>
          List(
            tried.copy(
              body = lowerBlock(body),
              handlers = handlers.map(h => h.copy(body = lowerBlock(h.body))),
              fin = lowerBlock(fin)
            )
          )
        case raised @ p.Stmt.Raise(_, _, cleanup) => List(raised.copy(cleanup = lowerBlock(cleanup)))
        case other                                => List(other)
      }

      function.copy(body = lowerBlock(function.body))
    }

    val physicalDefs = program.defs.map { definition =>
      definition.copy(members = definition.members.filterNot(emptyMember).map { member =>
        if (member.tpe.isInstanceOf[p.Type.FnRef]) member.copy(tpe = p.Type.IntU8) else member
      })
    }
    program.copy(
      entry = program.entry.map(legalise),
      functions = program.functions.map(legalise),
      defs = physicalDefs
    )
  }
}
