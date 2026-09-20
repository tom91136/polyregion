package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// legalises pointer representations for source dialects without a generic address space. it refines rooted pointer
// spaces, turns an immediately-read mixed-space pointer merge into a scalar merge, and specialises aggregate layouts
// whose pointer fields have a concrete storage space. any remaining mixed-space pointer is rejected instead of guessed
// by the source emitter
// examples:
//   p = cond ? &global[i] : &local[j]; x = *p  ->  x = cond ? global[i] : local[j]
//   Box{T* p}; a.p = global; b.p = local       ->  Box{global T* p}; Box_asl{local T* p}
//   q = &localPtr                              ->  q is a private slot containing a local pointer
// edge cases:
//   mixed pointer escapes / mixed aggregate slot  ->  rejected for strict source dialects
//   indexed address &p[i]                         ->  retains p's element type, not pointer-slot type
//   non-strict C11                                ->  retains representable generic pointer forms
final case class SourcePointerLegalise(requiresConcreteSpaces: Boolean = true) extends ProgramPass
    derives PassArgCodec {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private final case class PointerMerge(symbol: String, scalar: p.Type, aliases: Set[String], readIndex: p.Term)

  private type FieldPath   = List[String]
  private type Requirement = Map[FieldPath, p.Type.Space]

  private def structOf(tpe: p.Type): Option[p.Type.Struct] = tpe match {
    case structure: p.Type.Struct                   => Some(structure)
    case p.Type.Ptr(structure: p.Type.Struct, _)    => Some(structure)
    case p.Type.Arr(structure: p.Type.Struct, _, _) => Some(structure)
    case _                                          => None
  }

  private def retypeStruct(tpe: p.Type, replacement: p.Type.Struct): p.Type = tpe match {
    case _: p.Type.Struct                          => replacement
    case p.Type.Ptr(_: p.Type.Struct, space)       => p.Type.Ptr(replacement, space)
    case p.Type.Arr(_: p.Type.Struct, size, space) => p.Type.Arr(replacement, size, space)
    case other                                     => other
  }

  private def spaceCode(space: p.Type.Space): String = space match {
    case p.Type.Space.Global   => "g"
    case p.Type.Space.Constant => "c"
    case p.Type.Space.Local    => "l"
    case p.Type.Space.Private  => "p"
  }

  private final class AggregateSpecialiser(program: p.Program) {
    private val originalDefs = program.defs.map(definition => definition.name -> definition).toMap
    private val definitions  = scala.collection.mutable.LinkedHashMap.from(originalDefs)
    private val clones       = scala.collection.mutable.LinkedHashMap.empty[(p.Sym, Requirement), p.Type.Struct]
    private val takenNames   = scala.collection.mutable.Set.from(program.defs.map(_.name.fqn.mkString(".")))

    private def memberType(owner: p.Type, name: String): Option[p.Type] = structOf(owner).flatMap { structure =>
      definitions.get(structure.name).flatMap(_.members.find(_.symbol == name).map(_.tpe))
    }

    private def element(tpe: p.Type): p.Type = tpe match {
      case p.Type.Ptr(component, _)    => component
      case p.Type.Arr(component, _, _) => component
      case other                       => other
    }

    def typeAt(root: p.Type, steps: List[p.PathStep]): p.Type = steps.foldLeft(root) {
      case (current, p.PathStep.Field(name)) => memberType(current, name).getOrElse(p.Type.Nothing)
      case (current, p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn) => element(current)
    }

    private def fieldPath(steps: List[p.PathStep]): FieldPath = steps.collect { case p.PathStep.Field(name) => name }

    private def expressionSpace(
        expression: p.Expr,
        solution: AddressRefinement.Solution
    ): Option[p.Type.Space] = {
      val spaces = solution.result(expression).spaces
      Option
        .when(spaces.size == 1)(spaces.head)
        .orElse {
          expression match {
            case p.Expr.RefTo(_, _, _, space, _)     => Some(space)
            case p.Expr.Alias(_: p.Term.StringConst) => Some(p.Type.Space.Constant)
            case p.Expr.Alias(term)                  => AddressRefinement.spaceOf(term.tpe)
            case other                               => AddressRefinement.spaceOf(other.tpe)
          }
        }
    }

    private def pointerLeaves(tpe: p.Type, seen: Set[p.Sym] = Set.empty): List[p.Type.Space] = tpe match {
      case p.Type.Ptr(_, space)        => List(space)
      case p.Type.Arr(component, _, _) => pointerLeaves(component, seen)
      case p.Type.Struct(symbol, _) if !seen(symbol) =>
        definitions.get(symbol).toList.flatMap(_.members.flatMap(member => pointerLeaves(member.tpe, seen + symbol)))
      case _ => Nil
    }

    private def freshName(original: p.Sym, suffix: String): p.Sym = {
      val base   = original.fqn.lastOption.getOrElse("") + "_as" + suffix
      val prefix = original.fqn.dropRight(1)
      val fqn = (Iterator.single(base) ++ Iterator.from(1).map(base + _))
        .map(prefix :+ _)
        .find(name => takenNames.add(name.mkString(".")))
        .get
      p.Sym(fqn)
    }

    private def cloneStruct(structure: p.Type.Struct, requirement: Requirement): p.Type.Struct = {
      if (requirement.isEmpty) return structure
      clones.getOrElseUpdate(
        structure.name -> requirement, {
          val definition = definitions.getOrElse(
            structure.name,
            throw IllegalArgumentException(s"missing definition for ${structure.name.fqn.mkString(".")}")
          )
          val members = definition.members.map { member =>
            val direct = requirement.get(List(member.symbol))
            val nested = requirement.collect {
              case (head :: tail, space) if head == member.symbol && tail.nonEmpty => tail -> space
            }
            val retyped = (member.tpe, direct) match {
              case (p.Type.Ptr(component, old), Some(space)) if old != space => p.Type.Ptr(component, space)
              case _                                                         => member.tpe
            }
            val nestedRetyped = structOf(retyped).filter(_ => nested.nonEmpty).fold(retyped) { child =>
              retypeStruct(retyped, cloneStruct(child, nested))
            }
            member.copy(tpe = nestedRetyped)
          }
          val changed = members != definition.members
          if (!changed) structure
          else {
            val signature            = members.flatMap(member => pointerLeaves(member.tpe)).map(spaceCode).mkString
            val name                 = freshName(structure.name, signature)
            val clone: p.Type.Struct = p.Type.Struct(name, structure.args)
            def retargetSelf(tpe: p.Type): p.Type = tpe match {
              case value: p.Type.Struct if value == structure => clone
              case p.Type.Ptr(component, space)               => p.Type.Ptr(retargetSelf(component), space)
              case p.Type.Arr(component, size, space)         => p.Type.Arr(retargetSelf(component), size, space)
              case other                                      => other
            }
            definitions(name) = definition.copy(
              name = name,
              members = members.map(member => member.copy(tpe = retargetSelf(member.tpe)))
            )
            clone
          }
        }
      )
    }

    private def declarations(function: p.Function): Map[String, p.Named] =
      (function.receiver.iterator.map(_.named) ++ function.args.iterator.map(_.named) ++
        function.moduleCaptures.iterator.map(_.named) ++ function.termCaptures.iterator.map(_.named) ++
        function.collectAll[p.Stmt].iterator.collect { case p.Stmt.Var(name, _, _) => name })
        .map(name => name.symbol -> name)
        .toMap

    private def requirements(function: p.Function): Map[String, Requirement] = {
      val solution = AddressRefinement.solve(program, function)
      val declared = declarations(function)
      val grouped = solution.slots.toList
        .flatMap { case (AddressRefinement.Query.Slot(root, steps), value) =>
          declared.get(root).flatMap { named =>
            val path = fieldPath(steps)
            typeAt(named.tpe, steps) match {
              case _: p.Type.Ptr if path.nonEmpty && value.spaces.nonEmpty => Some((root -> path) -> value.spaces)
              case _                                                       => None
            }
          }
        }
        .groupMap(_._1)(_._2)
        .view
        .mapValues(_.flatten.toSet)
        .toMap
      val stores = function
        .collectAll[p.Stmt]
        .flatMap {
          case p.Stmt.Mut(p.Term.Select(root, steps, _), expression) if steps.nonEmpty =>
            val path = fieldPath(steps)
            Option
              .when(path.nonEmpty && AddressRefinement.isPtr(typeAt(root.tpe, steps))) {
                expressionSpace(expression, solution).map(space => (root.symbol -> path) -> space)
              }
              .flatten
          case _ => None
        }
        .groupMap(_._1)(_._2)
        .view
        .mapValues(_.toSet)
        .toMap
      val effective = (grouped.keySet ++ stores.keySet).iterator.map { slot =>
        slot -> stores.get(slot).filter(_.nonEmpty).orElse(grouped.get(slot)).getOrElse(Set.empty)
      }.toMap
      val ambiguous = effective.collect {
        case ((root, path), spaces) if spaces.size > 1 => s"$root.${path.mkString(".")}"
      }
      if (requiresConcreteSpaces && ambiguous.nonEmpty)
        throw IllegalArgumentException(
          s"aggregate pointer slot has incompatible address spaces: ${ambiguous.toList.sorted.mkString(", ")}"
        )
      effective.iterator
        .collect { case ((root, path), spaces) if spaces.size == 1 => root -> (path -> spaces.head) }
        .toList
        .groupMap(_._1)(_._2)
        .view
        .mapValues(_.toMap)
        .toMap
    }

    private def selectedType(select: p.Term.Select, types: Map[String, p.Type]): p.Type =
      typeAt(types.getOrElse(select.root.symbol, select.root.tpe), select.steps)

    private def indexedComponent(selected: p.Type, fallback: p.Type): p.Type =
      structOf(selected)
        .filterNot(structure => originalDefs.contains(structure.name))
        .fold(fallback)(retypeStruct(fallback, _))

    private def referenceSpace(
        localStorage: Set[String],
        select: p.Term.Select,
        index: Option[p.Term],
        selected: p.Type,
        fallback: p.Type.Space
    ): p.Type.Space =
      if (select.steps.isEmpty && index.isEmpty) p.Type.Space.Private
      else if (localStorage(select.root.symbol) && !AddressRefinement.isPtr(select.root.tpe))
        select.root.tpe match {
          case p.Type.Arr(_, _, p.Type.Space.Global) => p.Type.Space.Private
          case _ => AddressRefinement.spaceOf(select.root.tpe).getOrElse(p.Type.Space.Private)
        }
      else if (select.steps.isEmpty) AddressRefinement.spaceOf(selected).getOrElse(fallback)
      else fallback

    private def inferredTypes(function: p.Function, requirement: Map[String, Requirement]): Map[String, p.Type] = {
      val declared     = declarations(function)
      val localStorage = function.collectAll[p.Stmt].collect { case p.Stmt.Var(name, _, _) => name.symbol }.toSet
      val initial = declared.iterator.map { case (symbol, named) =>
        val retyped = structOf(named.tpe)
          .flatMap { structure =>
            requirement.get(symbol).map(req => retypeStruct(named.tpe, cloneStruct(structure, req)))
          }
          .getOrElse(named.tpe)
        symbol -> retyped
      }.toMap
      def propagatePointerStruct(
          types: Map[String, p.Type],
          name: p.Named,
          source: p.Term.Select,
          followSourceType: Boolean
      ): Map[String, p.Type] = {
        val targetType = types.getOrElse(name.symbol, name.tpe)
        val sourceType = selectedType(source, types)
        val specialised = List(targetType, sourceType)
          .flatMap(structOf)
          .find(structure => !originalDefs.contains(structure.name))
        specialised match {
          case Some(structure) =>
            val target     = retypeStruct(targetType, structure)
            val withTarget = types.updated(name.symbol, target)
            if (source.steps.isEmpty && structOf(sourceType).nonEmpty)
              withTarget.updated(source.root.symbol, retypeStruct(sourceType, structure))
            else withTarget
          case None =>
            sourceType match {
              case pointer: p.Type.Ptr if followSourceType => types.updated(name.symbol, pointer)
              case _                                       => types
            }
        }
      }
      doUntilNotEq(initial) { (_, known) =>
        function.collectAll[p.Stmt].foldLeft(known) {
          case (types, p.Stmt.Var(name @ p.Named(_, _: p.Type.Ptr, _), Some(p.Expr.Alias(select: p.Term.Select)), _)) =>
            propagatePointerStruct(types, name, select, followSourceType = true)
          case (
                types,
                p.Stmt.Var(
                  name @ p.Named(_, _: p.Type.Ptr, _),
                  Some(p.Expr.Cast(select: p.Term.Select, _: p.Type.Ptr)),
                  _
                )
              ) =>
            propagatePointerStruct(types, name, select, followSourceType = false)
          case (
                types,
                p.Stmt.Var(
                  name @ p.Named(_, _: p.Type.Ptr, _),
                  Some(ref @ p.Expr.RefTo(select: p.Term.Select, index, _, space, _)),
                  _
                )
              ) =>
            val selected = selectedType(select, types)
            val component =
              if (select.steps.isEmpty && index.isEmpty) types.getOrElse(select.root.symbol, select.root.tpe)
              else if (index.nonEmpty) indexedComponent(selected, ref.comp)
              else structOf(selected).fold(ref.comp)(_ => selected)
            types.updated(
              name.symbol,
              p.Type.Ptr(component, referenceSpace(localStorage, select, index, selected, space))
            )
          case (types, p.Stmt.Var(name, Some(p.Expr.Alias(select: p.Term.Select)), _)) if structOf(name.tpe).nonEmpty =>
            val source = selectedType(select, types)
            if (structOf(source).nonEmpty) types.updated(name.symbol, source) else types
          case (types, p.Stmt.Var(name, Some(p.Expr.RefTo(select: p.Term.Select, _, _, space, _)), _))
              if AddressRefinement.isPtr(name.tpe) && structOf(name.tpe).nonEmpty =>
            structOf(selectedType(select, types)).fold(types)(structure =>
              types.updated(name.symbol, p.Type.Ptr(structure, space))
            )
          case (types, p.Stmt.Mut(p.Term.Select(target, Nil, _), p.Expr.Alias(source: p.Term.Select)))
              if structOf(target.tpe).nonEmpty =>
            val sourceType = selectedType(source, types)
            if (structOf(sourceType).nonEmpty) types.updated(target.symbol, sourceType) else types
          case (types, _) => types
        }
      }._2
    }

    def rewrite(function: p.Function): p.Function = {
      val types        = inferredTypes(function, requirements(function))
      val localStorage = function.collectAll[p.Stmt].collect { case p.Stmt.Var(name, _, _) => name.symbol }.toSet
      def retype(name: p.Named): p.Named = types.get(name.symbol).fold(name)(tpe => name.copy(tpe = tpe))
      def retypeArg(arg: p.Arg): p.Arg   = arg.copy(named = retype(arg.named))
      def conform(expression: p.Expr, expected: p.Type): p.Expr = expression match {
        case cast @ p.Expr.Cast(_, _: p.Type.Ptr) if AddressRefinement.isPtr(expected) => cast.copy(as = expected)
        case other                                                                     => other
      }
      val rewritten = function
        .modifyAll[p.Term] {
          case select: p.Term.Select =>
            val root     = retype(select.root)
            val resolved = typeAt(root.tpe, select.steps)
            val selected =
              if (select.steps.isEmpty) root.tpe
              else
                resolved match {
                  case _: p.Type.Ptr | _: p.Type.Struct | _: p.Type.Arr => resolved
                  case _                                                => select.tpe
                }
            p.Term.Select(root, select.steps, selected)
          case term => term
        }
        .modifyAll[p.Expr] {
          case ref @ p.Expr.RefTo(select: p.Term.Select, _, _, _, _) =>
            val resolved = typeAt(select.root.tpe, select.steps)
            val component =
              if (select.steps.isEmpty && ref.idx.isEmpty) select.root.tpe
              else if (ref.idx.nonEmpty) indexedComponent(resolved, ref.comp)
              else structOf(resolved).fold(ref.comp)(_ => resolved)
            ref.copy(
              lhs = select,
              comp = component,
              space = referenceSpace(localStorage, select, ref.idx, select.root.tpe, ref.space)
            )
          case expression => expression
        }
        .modifyAll[p.Stmt] {
          case variable: p.Stmt.Var =>
            val name = retype(variable.name)
            variable.copy(name = name, expr = variable.expr.map(conform(_, name.tpe)))
          case mutation @ p.Stmt.Mut(target, expression) =>
            mutation.copy(expr = conform(expression, typeAt(target.root.tpe, target.steps)))
          case range: p.Stmt.ForRange => range.copy(induction = retype(range.induction))
          case statement              => statement
        }
      val returns = rewritten.collectAll[p.Stmt].collect { case p.Stmt.Return(value) => value.tpe }.distinct
      rewritten.copy(
        decl = rewritten.decl.copy(
          receiver = rewritten.receiver.map(retypeArg),
          args = rewritten.args.map(retypeArg),
          moduleCaptures = rewritten.moduleCaptures.map(retypeArg),
          termCaptures = rewritten.termCaptures.map(retypeArg),
          rtn = returns.headOption.getOrElse(rewritten.rtn)
        )
      )
    }

    def result(entry: Option[p.Function], functions: List[p.Function]): p.Program =
      program.copy(
        entry = entry,
        functions = functions,
        defs = originalDefs.values.toList ++ definitions.iterator
          .filterNot((name, _) => originalDefs.contains(name))
          .map(_._2)
          .toList
          .sortBy(_.name.fqn.mkString("."))
      )
  }

  private def isZero(term: p.Term): Boolean = term match {
    case p.Term.IntU8Const(value)  => value == 0
    case p.Term.IntU16Const(value) => value == 0
    case p.Term.IntU32Const(value) => value == 0
    case p.Term.IntU64Const(value) => value == 0
    case p.Term.IntS8Const(value)  => value == 0
    case p.Term.IntS16Const(value) => value == 0
    case p.Term.IntS32Const(value) => value == 0
    case p.Term.IntS64Const(value) => value == 0
    case _                         => false
  }

  private def load(candidate: PointerMerge, expr: p.Expr): Option[p.Expr] = expr match {
    case p.Expr.Alias(ref) if AddressRefinement.isPtr(ref.tpe) =>
      Some(p.Expr.Index(ref, candidate.readIndex, candidate.scalar))
    case p.Expr.RefTo(lhs, Some(index), _, _, _) => Some(p.Expr.Index(lhs, index, candidate.scalar))
    case p.Expr.RefTo(lhs, None, _, _, _)        => Some(p.Expr.Alias(lhs))
    case _                                       => None
  }

  private def plan(function: p.Function, symbol: String): Option[PointerMerge] = {
    val statements = function.collectAll[p.Stmt]
    val variables  = statements.collect { case variable: p.Stmt.Var => variable }
    val mutations  = statements.collect { case mutation: p.Stmt.Mut => mutation }
    val declaration = variables.collect {
      case p.Stmt.Var(name @ p.Named(`symbol`, p.Type.Ptr(component, _), _), None, _) => name -> component
    }
    if (declaration.size != 1) return None
    val scalar = declaration.head._2

    val aliases = doUntilNotEq(Set(symbol)) { (_, known) =>
      known ++ variables.collect {
        case p.Stmt.Var(
              p.Named(alias, p.Type.Ptr(component, _), _),
              Some(p.Expr.Alias(p.Term.Select(root, Nil, _))),
              _
            ) if component == scalar && known(root.symbol) =>
          alias
      }
    }._2
    def rooted(select: p.Term.Select): Boolean = aliases(select.root.symbol)

    val assignments = mutations.collect {
      case mutation @ p.Stmt.Mut(p.Term.Select(root, Nil, _), _) if root.symbol == symbol => mutation
    }
    if (assignments.size != 2) return None

    val aliasInitialisers = variables.count {
      case p.Stmt.Var(name, Some(p.Expr.Alias(select: p.Term.Select)), _) =>
        name.symbol != symbol && aliases(name.symbol) && select.steps.isEmpty && rooted(select)
      case _ => false
    }
    val indexedReads = function.collectAll[p.Expr].collect {
      case index @ p.Expr.Index(select: p.Term.Select, _, component)
          if select.steps.isEmpty && rooted(select) && component == scalar =>
        index
    }
    val dereferenceReads = function.collectAll[p.Expr].collect {
      case alias @ p.Expr.Alias(select: p.Term.Select)
          if select.steps == List(p.PathStep.Deref) && rooted(select) && select.tpe == scalar =>
        alias
    }
    if (indexedReads.size + dereferenceReads.size != 1) return None

    val selectionCount = function.collectAll[p.Term].count {
      case select: p.Term.Select => rooted(select)
      case _                     => false
    }
    if (selectionCount != assignments.size + aliasInitialisers + indexedReads.size + dereferenceReads.size) return None

    val readIndex = indexedReads.headOption.map(_.idx).getOrElse(p.Term.IntS64Const(0))
    if (
      !isZero(readIndex) || assignments
        .exists(mutation => load(PointerMerge(symbol, scalar, aliases, readIndex), mutation.expr).isEmpty)
    )
      return None

    def assigns(body: List[p.Stmt]): Int = body.count {
      case p.Stmt.Mut(p.Term.Select(root, Nil, _), _) => root.symbol == symbol
      case _                                          => false
    }
    def containsRead(statement: p.Stmt): Boolean = statement.collectAll[p.Expr].exists {
      case p.Expr.Index(select: p.Term.Select, _, _) => rooted(select)
      case p.Expr.Alias(select: p.Term.Select)       => select.steps.nonEmpty && rooted(select)
      case _                                         => false
    }
    def aliasInitialiser(statement: p.Stmt): Boolean = statement match {
      case p.Stmt.Var(name, Some(p.Expr.Alias(select: p.Term.Select)), _) =>
        name.symbol != symbol && aliases(name.symbol) && select.steps.isEmpty && rooted(select)
      case _ => false
    }
    def pureVariable(statement: p.Stmt): Boolean = statement match {
      case variable: p.Stmt.Var =>
        !variable.collectAll[p.Expr].exists {
          case _: p.Expr.Invoke | _: p.Expr.ForeignCall | _: p.Expr.SpecOp | _: p.Expr.Alloc => true
          case _                                                                             => false
        }
      case _ => false
    }
    def safeEnvelope(body: List[p.Stmt]): Boolean = {
      def tailAssigns(branch: List[p.Stmt]): Boolean = branch.lastOption.exists {
        case p.Stmt.Mut(p.Term.Select(root, Nil, _), _) => root.symbol == symbol
        case _                                          => false
      }
      def search(rest: List[p.Stmt], declared: Boolean): Boolean = rest match {
        case Nil                                                        => false
        case p.Stmt.Var(name, None, _) :: tail if name.symbol == symbol => search(tail, declared = true)
        case p.Stmt.Cond(_, trueBr, falseBr) :: tail =>
          val local = declared && tailAssigns(trueBr) && tailAssigns(falseBr) && assigns(trueBr) == 1 && assigns(
            falseBr
          ) == 1 && tail.iterator
            .takeWhile(statement => containsRead(statement) || aliasInitialiser(statement) || pureVariable(statement))
            .exists(containsRead)
          local || safeEnvelope(trueBr) || safeEnvelope(falseBr) || search(tail, declared)
        case p.Stmt.While(_, loop) :: tail             => safeEnvelope(loop) || search(tail, declared)
        case p.Stmt.ForRange(_, _, _, _, loop) :: tail => safeEnvelope(loop) || search(tail, declared)
        case p.Stmt.Annotated(inner, _, _) :: tail     => safeEnvelope(List(inner)) || search(tail, declared)
        case _ :: tail                                 => search(tail, declared)
      }
      search(body, declared = false)
    }
    Option.when(safeEnvelope(function.body))(PointerMerge(symbol, scalar, aliases, readIndex))
  }

  private def demote(function: p.Function, symbols: Set[String]): p.Function = {
    val candidates = symbols.toList.sorted.flatMap(plan(function, _))
    val byRoot     = candidates.map(candidate => candidate.symbol -> candidate).toMap
    val byAlias    = candidates.flatMap(candidate => candidate.aliases.map(_ -> candidate)).toMap
    if (candidates.isEmpty) function
    else {
      def tracked(name: p.Named): Option[PointerMerge] =
        byAlias
          .get(name.symbol)
          .filter(candidate =>
            name.tpe match {
              case p.Type.Ptr(component, _) => component == candidate.scalar
              case _                        => false
            }
          )
      def retype(name: p.Named): p.Named = tracked(name).fold(name)(candidate => name.copy(tpe = candidate.scalar))
      val rewritten = function
        .modifyAll[p.Stmt] {
          case p.Stmt.Mut(p.Term.Select(root, Nil, _), expr) if byRoot.contains(root.symbol) =>
            val candidate = byRoot(root.symbol)
            p.Stmt.Mut(p.Term.Select(root, Nil, candidate.scalar), load(candidate, expr).get)
          case statement => statement
        }
        .modifyAll[p.Expr] {
          case p.Expr.Index(p.Term.Select(root, Nil, _), _, _) if tracked(root).nonEmpty =>
            p.Expr.Alias(p.Term.Select(root, Nil, tracked(root).get.scalar))
          case expression => expression
        }
        .modifyAll[p.Term] {
          case p.Term.Select(root, steps, _)
              if tracked(root).nonEmpty &&
                (steps.isEmpty || steps == List(p.PathStep.Deref)) =>
            p.Term.Select(retype(root), Nil, tracked(root).get.scalar)
          case term => term
        }
        .modifyAll[p.Stmt] {
          case variable: p.Stmt.Var => variable.copy(name = retype(variable.name))
          case statement            => statement
        }
      def retypeArg(arg: p.Arg): p.Arg = arg.copy(named = retype(arg.named))
      rewritten.copy(
        decl = rewritten.decl.copy(
          receiver = rewritten.receiver.map(retypeArg),
          args = rewritten.args.map(retypeArg),
          moduleCaptures = rewritten.moduleCaptures.map(retypeArg),
          termCaptures = rewritten.termCaptures.map(retypeArg)
        )
      )
    }
  }

  private def legalise(program: p.Program, function: p.Function): p.Function = {
    val analysis  = AddressRefinement.solve(program, function)
    val conflicts = analysis.refinedSpaces.collect { case (symbol, spaces) if spaces.size > 1 => symbol }.toSet
    val demoted   = demote(function, conflicts)
    val remaining = AddressRefinement
      .solve(
        program.copy(
          entry = Option.when(program.entry.contains(function))(demoted),
          functions = program.functions.map {
            case `function` => demoted
            case other      => other
          }
        ),
        demoted
      )
      .refinedSpaces
      .collect { case (symbol, spaces) if spaces.size > 1 => symbol -> spaces }
    if (requiresConcreteSpaces && remaining.nonEmpty) {
      val details = remaining.toList.sortBy(_._1).map { case (symbol, spaces) =>
        s"$symbol=${spaces.toList.map(_.toString).sorted.mkString("|")}"
      }
      throw IllegalArgumentException(
        s"cross-address-space pointer merge escapes read-only use in `${function.name.fqn.mkString(".")}`: ${details.mkString(", ")}"
      )
    }
    demoted
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    def respace(function: p.Function): (p.Function, Int) =
      RegionRespace.run(
        program,
        function,
        requireSolved = false,
        adaptPointerStores = requiresConcreteSpaces
      )

    val (entry0, entryCount) = program.entry
      .map(respace)
      .map((function, count) => Some(function) -> count)
      .getOrElse(None -> 0)
    val (functions0, counts) = program.functions.map(respace).unzip
    val respaced             = program.copy(entry = entry0, functions = functions0)
    val total                = entryCount + counts.sum
    if (total > 0) log.info(s"respaced $total rooted pointer(s) during source legalisation")
    val entry        = respaced.entry.map(legalise(respaced, _))
    val functions    = respaced.functions.map(legalise(respaced, _))
    val pointerLegal = respaced.copy(entry = entry, functions = functions)
    if (!requiresConcreteSpaces) pointerLegal
    else {
      val aggregates     = AggregateSpecialiser(pointerLegal)
      val finalEntry     = pointerLegal.entry.map(aggregates.rewrite)
      val finalFunctions = pointerLegal.functions.map(aggregates.rewrite)
      aggregates.result(finalEntry, finalFunctions)
    }
  }
}
