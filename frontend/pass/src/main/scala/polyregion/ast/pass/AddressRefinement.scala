package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

/** Constraint-based address refinement.
  *
  * The PolyAST remains the source tree. Refinements are composed with it in a typed side table and are valid only for
  * that exact tree. Missing evidence is retained as a diagnostic obligation rather than invented as an address.
  */
private[pass] object AddressRefinement {

  enum Encoding {
    case Absolute, ArenaRelative
  }

  enum Provenance {
    case ArenaRoot(capture: String)
    case Parameter(symbol: String)
    case Local(symbol: String, path: List[p.PathStep] = Nil)
    case Allocation
    case StaticStorage
  }

  /** One possible abstract address. Keeping the encoding, provenance and address space together prevents joins from
    * inventing combinations which did not occur on either incoming path. An arena base is simply an absolute address
    * with `ArenaRoot` provenance; it is not a third encoding.
    */
  enum AbstractAddress {
    case Null
    case Absolute(provenance: Option[Provenance], space: Option[p.Type.Space])
    case ArenaRelative(arena: Option[String])

    def encoding: Option[Encoding] = this match {
      case Null             => None
      case Absolute(_, _)   => Some(Encoding.Absolute)
      case ArenaRelative(_) => Some(Encoding.ArenaRelative)
    }

    def provenanceOption: Option[Provenance] = this match {
      case Null                         => None
      case Absolute(provenance, _)      => provenance
      case ArenaRelative(Some(capture)) => Some(Provenance.ArenaRoot(capture))
      case ArenaRelative(None)          => None
    }

    def spaceOption: Option[p.Type.Space] = this match {
      case Null               => None
      case Absolute(_, space) => space
      case ArenaRelative(_)   => Some(p.Type.Space.Global)
    }
  }

  enum Coercion {
    case Preserve, ResolveRelative, EncodeRelative
  }

  enum AddressModel {
    case Physical, Logical
  }

  private enum StorageRegion {
    case Local, Arena, External, Undetermined
  }

  enum Query[A] {
    case Binding(symbol: String)                    extends Query[AddressValue]
    case Slot(root: String, path: List[p.PathStep]) extends Query[AddressValue]
    case TermValue(term: p.Term)                    extends Query[AddressValue]
    case ExprResult(expr: p.Expr)                   extends Query[AddressValue]
    case AddressOf(term: p.Term)                    extends Query[Set[Provenance]]

    def label: String = this match {
      case Binding(symbol)  => symbol
      case Slot(root, path) => s"$root.${path.mkString(".")}"
      case TermValue(term)  => term.repr
      case ExprResult(expr) => expr.repr
      case AddressOf(term)  => term.repr
    }
  }

  final case class AddressValue(
      alternatives: Set[AbstractAddress] = Set.empty,
      references: Set[Query[AddressValue]] = Set.empty,
      obligations: Set[String] = Set.empty,
      encoding: Option[Encoding] = None
  ) {
    def encodings: Set[Encoding]     = alternatives.flatMap(_.encoding)
    def provenances: Set[Provenance] = alternatives.flatMap(_.provenanceOption)
    def spaces: Set[p.Type.Space]    = alternatives.flatMap(_.spaceOption)
    def includesNull: Boolean        = alternatives.contains(AbstractAddress.Null)
    def hasArenaRoot: Boolean = alternatives.exists {
      case AbstractAddress.Absolute(Some(_: Provenance.ArenaRoot), _) => true
      case _                                                          => false
    }
    def hasNonArenaAbsolute: Boolean = alternatives.exists {
      case AbstractAddress.Absolute(Some(_: Provenance.ArenaRoot), _) => false
      case _: AbstractAddress.Absolute                                => true
      case _                                                          => false
    }

    private[pass] def join(that: AddressValue): AddressValue =
      AddressValue(
        alternatives ++ that.alternatives,
        references ++ that.references,
        obligations ++ that.obligations
      )

    private[AddressRefinement] def inEncoding(encoding: Encoding): AddressValue = copy(
      alternatives = alternatives.map {
        case AbstractAddress.Null                                               => AbstractAddress.Null
        case address: AbstractAddress.Absolute if encoding == Encoding.Absolute => address
        case AbstractAddress.ArenaRelative(arena) if encoding == Encoding.Absolute =>
          AbstractAddress.Absolute(arena.map(Provenance.ArenaRoot(_)), Some(p.Type.Space.Global))
        case AbstractAddress.Absolute(Some(Provenance.ArenaRoot(capture)), _) =>
          AbstractAddress.ArenaRelative(Some(capture))
        case address: AbstractAddress.ArenaRelative => address
        // Conversion validation rejects non-arena absolute addresses before they can be stored relatively.
        case address => address
      },
      encoding = Some(encoding)
    )

    private[AddressRefinement] def read: AddressValue =
      encoding.fold(this)(inEncoding)

    private[AddressRefinement] def mapAbsolute(
        f: (Option[Provenance], Option[p.Type.Space]) => AbstractAddress
    ): AddressValue = copy(alternatives = alternatives.map {
      case AbstractAddress.Absolute(provenance, space) => f(provenance, space)
      case other                                       => other
    })
  }

  object AddressValue {
    val empty: AddressValue     = AddressValue()
    val nullValue: AddressValue = AddressValue(Set(AbstractAddress.Null))

    def absolute(
        provenance: Option[Provenance] = None,
        space: Option[p.Type.Space] = None
    ): AddressValue = AddressValue(Set(AbstractAddress.Absolute(provenance, space)))

    def absolutes(provenances: Iterable[Provenance], space: Option[p.Type.Space]): AddressValue =
      if (provenances.isEmpty) absolute(space = space)
      else AddressValue(provenances.iterator.map(p => AbstractAddress.Absolute(Some(p), space)).toSet)

    def arenaRoot(capture: String): AddressValue =
      AddressValue(
        Set(AbstractAddress.Absolute(Some(Provenance.ArenaRoot(capture)), Some(p.Type.Space.Global)))
      )

    def arenaRelative(captures: Iterable[String]): AddressValue = {
      val arenas = Option.when(captures.nonEmpty)(captures.iterator.map(Some(_)).toSet).getOrElse(Set(None))
      AddressValue(arenas.map(AbstractAddress.ArenaRelative(_)))
    }

    def unresolved(obligations: String*): AddressValue = AddressValue(obligations = obligations.toSet)
  }

  final class FactTable private[AddressRefinement] (private val entries: Map[Query[?], Any]) {
    def get[A](token: Query[A]): Option[A] = entries.get(token).map(_.asInstanceOf[A])
  }

  final case class CoercionSite(
      index: Int,
      target: Query[AddressValue],
      source: AddressValue,
      coercion: Coercion,
      context: String
  )

  final case class Diagnostic(code: String, message: String, evidence: List[String] = Nil) {
    override def toString: String =
      if (evidence.isEmpty) s"$code: $message" else s"$code: $message\n${evidence.map("  " + _).mkString("\n")}"
  }

  final case class Solution(
      function: p.Function,
      model: AddressModel,
      facts: FactTable,
      bindings: Map[String, AddressValue],
      slots: Map[Query.Slot, AddressValue],
      refinedSpaces: Map[String, Set[p.Type.Space]],
      coercions: List[CoercionSite],
      returned: AddressValue,
      diagnostics: List[Diagnostic]
  ) {
    def value(term: p.Term): AddressValue            = facts.get(Query.TermValue(term)).getOrElse(AddressValue())
    def result(expr: p.Expr): AddressValue           = facts.get(Query.ExprResult(expr)).getOrElse(AddressValue())
    def provenancesOf(term: p.Term): Set[Provenance] = facts.get(Query.AddressOf(term)).getOrElse(Set.empty)
    def refinedSpace(name: p.Named): Option[p.Type.Space] =
      refinedSpaces.get(name.symbol).flatMap(spaces => Option.when(spaces.size == 1)(spaces.head))

    def requireSolved: Solution = {
      if (diagnostics.nonEmpty) throw IllegalArgumentException(diagnostics.mkString("\n"))
      this
    }
  }

  type Field = (p.Sym, String)

  final case class LogicalAddressPlan(
      arenaStructs: Set[p.Sym],
      identityFields: Set[Field],
      fieldAliases: Map[p.Named, (Field, p.Term)],
      localPointerKeys: Map[p.Named, String],
      directLocalPointerKeys: Set[String]
  )

  private final case class StorageInfo(tpe: p.Type, region: StorageRegion)
  private final case class Pending(
      index: Int,
      target: Query[AddressValue],
      slot: Option[(p.Named, List[p.PathStep])],
      expr: p.Expr,
      context: String
  )
  private final case class SlotCopy(
      target: p.Named,
      targetPrefix: List[p.PathStep],
      source: p.Named,
      sourcePrefix: List[p.PathStep]
  )
  private final case class CallKey(function: String, boundary: List[(String, AddressValue)], model: AddressModel)
  private final class InferenceContext {
    val calls = scala.collection.mutable.Map.empty[CallKey, AddressValue]
  }

  def isPtr(t: p.Type): Boolean = t match {
    case _: p.Type.Ptr => true
    case _             => false
  }

  def reassignedIn(function: p.Function): Set[String] =
    function.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(name, Nil, _), _) => name.symbol }.toSet

  def spaceOf(tpe: p.Type): Option[p.Type.Space] = tpe match {
    case p.Type.Ptr(_, space)    => Some(space)
    case p.Type.Arr(_, _, space) => Some(space)
    case _                       => None
  }

  def withSpace(tpe: p.Type, space: p.Type.Space): p.Type = tpe match {
    case p.Type.Ptr(component, _)       => p.Type.Ptr(component, space)
    case p.Type.Arr(component, size, _) => p.Type.Arr(component, size, space)
    case other                          => other
  }

  /** Whether `prefix` denotes an abstract storage prefix of `path`.
    *
    * A dynamic array subscript is one summary location for address-refinement purposes: RecursionLower's explicit
    * stack, for example, writes a frame through `sp` and later reads it through `ci`. The value-level analysis proves
    * which element is live; this overlay only needs to retain the encoding shared by those slots.
    */
  def slotPrefix(prefix: List[p.PathStep], path: List[p.PathStep]): Boolean =
    prefix.size <= path.size && prefix.zip(path).forall {
      case (_: p.PathStep.IndexDyn, _: p.PathStep.Index)    => true
      case (_: p.PathStep.IndexDyn, _: p.PathStep.IndexDyn) => true
      case (_: p.PathStep.Index, _: p.PathStep.IndexDyn)    => true
      case (left, right)                                    => left == right
    }

  /** Plans the absolute-address identity values which a logical-address backend may safely encode.
    *
    * Arena offsets and identity-only local pointers are deliberately separate: the latter may be stored and compared,
    * but never dereferenced. Keeping that proof beside address refinement prevents a lowering from independently
    * guessing which absolute addresses are safe to encode.
    */
  def logicalAddressPlan(program: p.Program, analysis: Solution): LogicalAddressPlan = {
    require(analysis.model == AddressModel.Logical, "logical address planning requires the logical model")
    val function = analysis.function
    val members  = program.defs.iterator.map(d => d.name -> d.members).toMap

    def pointee(tpe: p.Type): p.Type = tpe match {
      case p.Type.Ptr(component, _) => component
      case other                    => other
    }
    def elem(tpe: p.Type): p.Type = tpe match {
      case p.Type.Ptr(component, _)    => component
      case p.Type.Arr(component, _, _) => component
      case other                       => other
    }
    def isArray(tpe: p.Type): Boolean = tpe match {
      case _: p.Type.Arr => true
      case _             => false
    }
    def fieldsAt(rootTpe: p.Type, steps: List[p.PathStep]): List[(Field, p.Type)] = {
      def member(symbol: p.Sym, field: String): Option[p.Type] =
        members.get(symbol).flatMap(_.find(_.symbol == field).map(_.tpe))
      steps
        .foldLeft((rootTpe, List.empty[(Field, p.Type)])) {
          case ((current, found), p.PathStep.Field(field)) =>
            val owner = pointee(current) match {
              case p.Type.Struct(symbol, _) => Some(symbol)
              case _                        => None
            }
            owner.flatMap(symbol => member(symbol, field).map(tpe => tpe -> (symbol -> field, tpe))) match {
              case Some((tpe, resolved)) => tpe     -> (found :+ resolved)
              case None                  => current -> found
            }
          case ((current, found), p.PathStep.Deref)       => pointee(current) -> found
          case ((current, found), _: p.PathStep.Index)    => elem(current)    -> found
          case ((current, found), _: p.PathStep.IndexDyn) => elem(current)    -> found
        }
        ._2
    }
    def staticIndex(index: Option[p.Term]): Option[String] = index match {
      case None                        => Some("")
      case Some(p.Term.IntU8Const(v))  => Some(java.lang.Byte.toUnsignedInt(v).toString)
      case Some(p.Term.IntU16Const(v)) => Some(v.toInt.toString)
      case Some(p.Term.IntU32Const(v)) => Some(java.lang.Integer.toUnsignedLong(v).toString)
      case Some(p.Term.IntU64Const(v)) => Some(java.lang.Long.toUnsignedString(v))
      case Some(p.Term.IntS8Const(v))  => Some(v.toString)
      case Some(p.Term.IntS16Const(v)) => Some(v.toString)
      case Some(p.Term.IntS32Const(v)) => Some(v.toString)
      case Some(p.Term.IntS64Const(v)) => Some(v.toString)
      case _                           => None
    }
    def staticPathKey(steps: List[p.PathStep]): Option[String] =
      if (steps.exists { case _: p.PathStep.IndexDyn => true; case _ => false }) None
      else
        Some(
          steps
            .map {
              case p.PathStep.Field(name) => s"field:$name"
              case p.PathStep.Deref       => "deref"
              case p.PathStep.Index(idx)  => s"index:$idx"
              case _: p.PathStep.IndexDyn => throw IllegalStateException("dynamic step in static arena path")
            }
            .mkString("/")
        )
    def localReferenceKey(base: p.Term.Select, index: Option[p.Term]): Option[String] =
      for {
        path <- staticPathKey(base.steps)
        idx  <- staticIndex(index)
      } yield s"${base.root.symbol}:$path:$idx"
    def referencedStructs(tpe: p.Type): Set[p.Sym] = tpe match {
      case p.Type.Struct(symbol, args) => Set(symbol) ++ args.flatMap(referencedStructs)
      case p.Type.Ptr(component, _)    => referencedStructs(component)
      case p.Type.Arr(component, _, _) => referencedStructs(component)
      case p.Type.Exec(_, args, rtn)   => args.flatMap(referencedStructs).toSet ++ referencedStructs(rtn)
      case _                           => Set.empty
    }
    val arenaStructs = doUntilNotEq(captureRoot(function).map(_._2.name).toSet) { (_, seen) =>
      seen ++ seen.flatMap(symbol => members.getOrElse(symbol, Nil).flatMap(member => referencedStructs(member.tpe)))
    }._2
    def isLocal(term: p.Term): Boolean = analysis.provenancesOf(term).exists {
      case _: Provenance.Local => true
      case _                   => false
    }
    val reassignedPointers = function
      .collectAll[p.Stmt]
      .collect { case p.Stmt.Mut(p.Term.Select(name, Nil, _: p.Type.Ptr), _) => name }
      .toSet
    val localPointerKeys = function.collectAll[p.Stmt].foldLeft(Map.empty[p.Named, String]) {
      case (known, p.Stmt.Var(name, Some(p.Expr.RefTo(base: p.Term.Select, index, _, _, _)), _))
          if isPtr(name.tpe) && !reassignedPointers(name) && isLocal(base) =>
        localReferenceKey(base, index).fold(known)(key => known.updated(name, key))
      case (known, p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), _))
          if isPtr(name.tpe) && !reassignedPointers(name) && !reassignedPointers(root) && known.contains(root) =>
        known.updated(name, known(root))
      case (known, p.Stmt.Var(name, Some(p.Expr.Cast(source @ p.Term.Select(root, Nil, _), _: p.Type.Ptr)), _))
          if isPtr(name.tpe) && !reassignedPointers(name) && !reassignedPointers(root) &&
            (known.contains(root) || (isArray(root.tpe) && isLocal(source))) =>
        known.updated(name, known.getOrElse(root, s"${root.symbol}:array"))
      case (known, _) => known
    }
    val directLocalPointerKeys = function
      .collectAll[p.Expr]
      .collect {
        case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) if isLocal(base) =>
          localReferenceKey(base, index)
      }
      .flatten
      .toSet
    def localIdentity(term: p.Term): Boolean = term match {
      case _: p.Term.NullPtrConst      => true
      case p.Term.Select(root, Nil, _) => isLocal(term) || localPointerKeys.contains(root)
      case _                           => false
    }
    def directSelectedField(term: p.Term): Option[Field] = term match {
      case p.Term.Select(root, steps, _: p.Type.Ptr) if steps.nonEmpty =>
        fieldsAt(root.tpe, steps).lastOption.collect { case (field, p.Type.Ptr(_, p.Type.Space.Global)) =>
          field
        }
      case _ => None
    }
    val fieldAliases = doUntilNotEq(Map.empty[p.Named, (Field, p.Term)]) { (_, known) =>
      val discovered = function
        .collectAll[p.Stmt]
        .collect { case p.Stmt.Var(name, Some(p.Expr.Alias(source)), false) =>
          directSelectedField(source)
            .map(field => name -> (field -> source))
            .orElse(source match {
              case p.Term.Select(root, Nil, _) =>
                known.get(root).map { case (field, original) =>
                  name -> (field -> original)
                }
              case _ => None
            })
        }
        .flatten
        .toMap
      known ++ discovered
    }._2
    def selectedField(term: p.Term): Option[Field] =
      directSelectedField(term).orElse(term match {
        case p.Term.Select(root, Nil, _: p.Type.Ptr) => fieldAliases.get(root).map(_._1)
        case _                                       => None
      })
    def identityWriteSource(expr: p.Expr): Option[Either[Field, Boolean]] = expr match {
      case p.Expr.Alias(rhs)               => Some(selectedField(rhs).toLeft(localIdentity(rhs)))
      case p.Expr.Cast(rhs, _: p.Type.Ptr) => Some(selectedField(rhs).toLeft(localIdentity(rhs)))
      case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) =>
        Some(Right(isLocal(base) && localReferenceKey(base, index).nonEmpty))
      case _ => None
    }
    val writes = function
      .collectAll[p.Stmt]
      .collect { case p.Stmt.Mut(target: p.Term.Select, expr) =>
        selectedField(target).flatMap(field => identityWriteSource(expr).map(field -> _))
      }
      .flatten
    def traversedPointerFields(term: p.Term): List[Field] = term match {
      case p.Term.Select(root, steps, _) if steps.nonEmpty =>
        fieldsAt(root.tpe, steps).collect { case (field, _: p.Type.Ptr) => field }
      case _ => Nil
    }
    def tally(fields: List[Field]): Map[Field, Int] = fields.groupMapReduce(identity)(_ => 1)(_ + _)
    val totalUses                                   = tally(function.collectAll[p.Term].flatMap(traversedPointerFields))
    val identityUses = tally(
      function.collectAll[p.Stmt].flatMap {
        case p.Stmt.Mut(target: p.Term.Select, p.Expr.Alias(rhs)) =>
          directSelectedField(target).toList ++ directSelectedField(rhs).toList
        case p.Stmt.Mut(target: p.Term.Select, _)         => directSelectedField(target).toList
        case p.Stmt.Var(_, Some(p.Expr.Alias(source)), _) => directSelectedField(source).toList
        case _                                            => Nil
      } ::: function.collectAll[p.Expr].flatMap {
        case p.Expr.IntrOp(p.Intr.LogicEq(x, y))  => List(x, y).flatMap(directSelectedField)
        case p.Expr.IntrOp(p.Intr.LogicNeq(x, y)) => List(x, y).flatMap(directSelectedField)
        case _                                    => Nil
      }
    )
    val writesByField = writes.groupMap(_._1)(_._2)
    val candidates = doUntilNotEq(Set.empty[Field]) { (_, known) =>
      writesByField.collect {
        case (field @ (owner, _), sources)
            if !arenaStructs(owner) && sources.nonEmpty && totalUses.get(field) == identityUses.get(field) &&
              sources.forall {
                case Right(ok)  => ok
                case Left(from) => from == field || known(from)
              } && sources.exists {
                case Right(ok)  => ok
                case Left(from) => from != field && known(from)
              } =>
          field
      }.toSet
    }._2
    def comparable(term: p.Term): Boolean = term match {
      case _: p.Term.NullPtrConst                                         => true
      case p.Term.Select(root, Nil, _) if localPointerKeys.contains(root) => true
      case selected if selectedField(selected).exists(candidates)         => true
      case _                                                              => false
    }
    val unsupported = function
      .collectAll[p.Expr]
      .flatMap {
        case p.Expr.IntrOp(p.Intr.LogicEq(x, y))  => List(x -> y, y -> x)
        case p.Expr.IntrOp(p.Intr.LogicNeq(x, y)) => List(x -> y, y -> x)
        case _                                    => Nil
      }
      .flatMap { case (selected, other) =>
        selectedField(selected).filter(candidates).filter(_ => !comparable(other))
      }
      .toSet
    LogicalAddressPlan(arenaStructs, candidates -- unsupported, fieldAliases, localPointerKeys, directLocalPointerKeys)
  }

  private def isAddressInt(t: p.Type): Boolean = t == p.Type.IntU64 || t == p.Type.IntS64

  private def isCarrier(t: p.Type): Boolean = isPtr(t) || isAddressInt(t)

  private def carrierResult(e: p.Expr): Boolean = isCarrier(e.tpe)

  private def tokenOf(root: p.Named): Query.Binding = Query.Binding(root.symbol)

  def solve(program: p.Program, function: p.Function, model: AddressModel = AddressModel.Physical): Solution =
    solve(program, function, model, Map.empty, Nil, InferenceContext())

  private def solve(
      program: p.Program,
      function: p.Function,
      model: AddressModel,
      boundaryFacts: Map[String, AddressValue],
      activeCalls: List[String],
      context: InferenceContext
  ): Solution = {
    val capture    = captureRoot(function).map(_._1)
    val statements = function.collectAll[p.Stmt]
    val declarations = (
      function.receiver.iterator.map(_.named) ++
        function.args.iterator.map(_.named) ++
        function.moduleCaptures.iterator.map(_.named) ++
        function.termCaptures.iterator.map(_.named) ++
        statements.iterator.collect { case p.Stmt.Var(n, _, _) => n }
    ).map(n => n.symbol -> n).toMap
    val localAggregates = statements.iterator.collect {
      case p.Stmt.Var(n, _, _) if !isPtr(n.tpe) => n.symbol
    }.toSet
    val members = program.defs.iterator.map(d => d.name -> d.members.map(m => m.symbol -> m.tpe).toMap).toMap
    def pointerSlotDepth(tpe: p.Type, seen: Set[p.Sym] = Set.empty): Int = tpe match {
      case _: p.Type.Ptr => 0
      case p.Type.Struct(symbol, _) if !seen(symbol) =>
        members
          .getOrElse(symbol, Map.empty)
          .valuesIterator
          .map {
            case _: p.Type.Ptr => 1
            case member =>
              val nested = pointerSlotDepth(member, seen + symbol)
              if (nested > 0) nested + 1 else 0
          }
          .maxOption
          .getOrElse(0)
      case p.Type.Arr(component, _, _) =>
        val nested = pointerSlotDepth(component, seen)
        if (nested > 0) nested + 1 else 0
      case _ => 0
    }
    val syntacticSlotDepth = function
      .collectAll[p.Term]
      .iterator
      .collect { case p.Term.Select(_, steps, _) => steps.size }
      .maxOption
      .getOrElse(0)
    val maxSlotDepth =
      math.max(syntacticSlotDepth, declarations.valuesIterator.map(n => pointerSlotDepth(n.tpe)).maxOption.getOrElse(0))
    val bound                    = (function.receiver.iterator ++ function.args.iterator).map(_.named).toList
    val logicalArenaAddressViews = LogicalArenaViewAbi.arenaAddressBindings(bound)

    def directSlot(root: p.Named, steps: List[p.PathStep], tpe: p.Type): Option[(Query.Slot, StorageInfo)] =
      if (steps.isEmpty || !isPtr(tpe)) None
      else if (capture.exists(_.symbol == root.symbol))
        Some(Query.Slot(root.symbol, steps) -> StorageInfo(tpe, StorageRegion.Arena))
      else if (localAggregates(root.symbol))
        Some(Query.Slot(root.symbol, steps)    -> StorageInfo(tpe, StorageRegion.Local))
      else Some(Query.Slot(root.symbol, steps) -> StorageInfo(tpe, StorageRegion.Undetermined))

    val pending = statements.zipWithIndex.flatMap { case (stmt, index) =>
      stmt match {
        case p.Stmt.Var(n, Some(expr), _) if isCarrier(n.tpe) =>
          Some(Pending(index, Query.Binding(n.symbol), None, expr, s"initialiser of ${n.symbol}"))
        case p.Stmt.Mut(p.Term.Select(n, Nil, tpe), expr) if isCarrier(tpe) =>
          Some(Pending(index, Query.Binding(n.symbol), None, expr, s"assignment to ${n.symbol}"))
        case p.Stmt.Mut(p.Term.Select(root, steps, tpe), expr) =>
          directSlot(root, steps, tpe).map { case (slot, _) =>
            Pending(index, slot, Some(root -> steps), expr, s"assignment to ${slot.label}")
          }
        case _ => None
      }
    }

    val bindingInfo = declarations.valuesIterator.collect {
      case n if isCarrier(n.tpe) =>
        val domain =
          if (boundaryFacts.contains(n.symbol)) StorageRegion.Undetermined
          else if (capture.exists(_.symbol == n.symbol)) StorageRegion.Arena
          else if (isPtr(n.tpe) && bound.exists(_.symbol == n.symbol)) StorageRegion.External
          else StorageRegion.Local
        Query.Binding(n.symbol) -> StorageInfo(n.tpe, domain)
    }.toMap
    val slotInfo = statements.iterator.flatMap {
      case p.Stmt.Mut(p.Term.Select(root, steps, tpe), _) => directSlot(root, steps, tpe)
      case _                                              => None
    }.toMap
    val slotCopies = statements.flatMap {
      case p.Stmt.Var(target, Some(p.Expr.Alias(p.Term.Select(source, prefix, _))), _) if !isPtr(target.tpe) =>
        Some(SlotCopy(target, Nil, source, prefix))
      case p.Stmt.Mut(
            p.Term.Select(target, targetPrefix, tpe),
            p.Expr.Alias(p.Term.Select(source, sourcePrefix, _))
          ) if !isPtr(tpe) =>
        Some(SlotCopy(target, targetPrefix, source, sourcePrefix))
      case _ => None
    }
    val storage: Map[Query[AddressValue], StorageInfo] =
      (bindingInfo.iterator.map((token, info) => (token: Query[AddressValue]) -> info) ++
        slotInfo.iterator.map((token, info) => (token: Query[AddressValue]) -> info)).toMap

    val seeds = storage.iterator.map { case (token, info) =>
      val fact = token match {
        case Query.Binding(symbol) if boundaryFacts.contains(symbol) => boundaryFacts(symbol)
        case Query.Binding(symbol) if capture.exists(_.symbol == symbol) =>
          AddressValue.arenaRoot(symbol)
        case Query.Binding(symbol) if info.region == StorageRegion.External =>
          AddressValue.absolute(Some(Provenance.Parameter(symbol)), spaceOf(info.tpe))
        case _: Query.Slot if info.region == StorageRegion.Arena =>
          AddressValue.arenaRelative(capture.map(_.symbol))
        case _ => AddressValue()
      }
      token -> fact
    }.toMap

    def classifyProvenance(origins: Set[Provenance]): (Boolean, Boolean) = {
      val arena = origins.exists { case _: Provenance.ArenaRoot => true; case _ => false }
      arena -> origins.exists { case _: Provenance.ArenaRoot => false; case _ => true }
    }

    def domainOf(
        token: Query[AddressValue],
        state: Map[Query[AddressValue], AddressValue]
    ): StorageRegion = storage.get(token).map(_.region).getOrElse(StorageRegion.Undetermined) match {
      case StorageRegion.Undetermined =>
        token match {
          case Query.Slot(root, _) if localAggregates(root) => StorageRegion.Local
          case Query.Slot(root, _) =>
            val origins        = state.getOrElse(Query.Binding(root), AddressValue()).provenances
            val (arena, other) = classifyProvenance(origins)
            if (arena && !other) StorageRegion.Arena
            else if (origins.exists { case _: Provenance.Local => true; case _ => false } && !arena) StorageRegion.Local
            else if (other && !arena) StorageRegion.External
            else StorageRegion.Undetermined
          case _ => StorageRegion.Undetermined
        }
      case known => known
    }

    def select(
        token: Query[AddressValue],
        fact: AddressValue,
        state: Map[Query[AddressValue], AddressValue]
    ): AddressValue = {
      val domain = domainOf(token, state)
      val isSlot = token match {
        case _: Query.Slot => true
        case _             => false
      }
      val isCaptureBinding = token match {
        case Query.Binding(symbol) => capture.exists(_.symbol == symbol)
        case _                     => false
      }
      val encoding = domain match {
        case StorageRegion.Arena if isSlot => Some(Encoding.ArenaRelative)
        case StorageRegion.External        => Some(Encoding.Absolute)
        case StorageRegion.Local if isSlot && model == AddressModel.Physical =>
          Some(Encoding.Absolute)
        case StorageRegion.Local if isSlot =>
          if (fact.encodings.contains(Encoding.Absolute)) Some(Encoding.Absolute)
          else if (fact.encodings.contains(Encoding.ArenaRelative)) Some(Encoding.ArenaRelative)
          else None
        case StorageRegion.Local if fact.includesNull && fact.encodings.isEmpty =>
          if (model == AddressModel.Physical) Some(Encoding.Absolute) else Some(Encoding.ArenaRelative)
        case StorageRegion.Local if fact.hasArenaRoot =>
          if (model == AddressModel.Physical) Some(Encoding.Absolute)
          else if (fact.hasNonArenaAbsolute) Some(Encoding.Absolute)
          else Some(Encoding.ArenaRelative)
        case StorageRegion.Arena if isCaptureBinding         => Some(Encoding.Absolute)
        case _ if fact.encodings.contains(Encoding.Absolute) => Some(Encoding.Absolute)
        case _ if fact.encodings.contains(Encoding.ArenaRelative) =>
          Some(Encoding.ArenaRelative)
        case _ => None
      }
      fact.copy(encoding = encoding)
    }

    def refineEncodings(state: Map[Query[AddressValue], AddressValue]): Map[Query[AddressValue], AddressValue] =
      state.iterator.map { case (token, fact) => token -> select(token, fact, state) }.toMap

    def selectValue(fact: AddressValue): AddressValue = {
      val encoding =
        if (fact.encodings.contains(Encoding.Absolute)) Some(Encoding.Absolute)
        else if (fact.encodings.contains(Encoding.ArenaRelative)) Some(Encoding.ArenaRelative)
        else fact.encoding
      fact.copy(encoding = encoding)
    }

    def read(state: Map[Query[AddressValue], AddressValue], token: Query[AddressValue]): AddressValue =
      state.get(token).map(_.read).getOrElse(AddressValue.unresolved(s"no facts for ${token.label}"))

    def pointerFromMemory(base: AddressValue, tpe: p.Type, context: String): AddressValue = {
      val (arena, other) = classifyProvenance(base.provenances)
      if (arena && other)
        AddressValue.unresolved(s"$context has both arena and non-arena origins")
      else if (arena)
        AddressValue.arenaRelative(base.provenances.collect { case Provenance.ArenaRoot(capture) => capture })
      else if (other)
        AddressValue.absolutes(base.provenances, spaceOf(tpe))
      else
        AddressValue(obligations = base.obligations + s"$context has no known memory origin")
    }

    def canonicalSlot(
        state: Map[Query[AddressValue], AddressValue],
        root: p.Named,
        steps: List[p.PathStep]
    ): Query.Slot = {
      val locals = state
        .getOrElse(Query.Binding(root.symbol), AddressValue())
        .provenances
        .collect { case Provenance.Local(symbol, path) => symbol -> path }
      if (locals.size == 1) {
        val (storage, prefix) = locals.head
        val suffix =
          if (!declarations.get(storage).exists(n => isPtr(n.tpe)) && steps.headOption.contains(p.PathStep.Deref))
            steps.tail
          else steps
        Query.Slot(storage, prefix ++ suffix)
      } else Query.Slot(root.symbol, steps)
    }

    def slotFact(
        state: Map[Query[AddressValue], AddressValue],
        root: p.Named,
        steps: List[p.PathStep]
    ): Option[(Int, AddressValue)] = {
      val slot = canonicalSlot(state, root, steps)
      val matching = state.iterator.collect {
        case (Query.Slot(slotRoot, path), fact) if slotRoot == slot.root && slotPrefix(path, slot.path) =>
          path.size -> fact.read
      }.toList
      matching
        .map(_._1)
        .maxOption
        .map(depth => depth -> matching.iterator.collect { case (`depth`, fact) => fact }.reduce(_.join(_)))
    }

    def termFact(state: Map[Query[AddressValue], AddressValue], term: p.Term): AddressValue = term match {
      case p.Term.NullPtrConst(_, _, _) => AddressValue.nullValue
      case _: p.Term.StringConst =>
        AddressValue.absolute(Some(Provenance.StaticStorage), Some(p.Type.Space.Constant))
      case p.Term.IntU64Const(value) => if (value == 0) AddressValue.nullValue else AddressValue.empty
      case p.Term.IntS64Const(value) => if (value == 0) AddressValue.nullValue else AddressValue.empty
      case p.Term.Defer(tpe) if isPtr(tpe) =>
        AddressValue.unresolved(s"deferred pointer ${tpe.repr}")
      case p.Term.Select(root, Nil, tpe) if isCarrier(tpe) => read(state, tokenOf(root))
      case p.Term.Select(root, steps, tpe) if isPtr(tpe) =>
        val slot = canonicalSlot(state, root, steps)
        slotFact(state, root, steps) match {
          case Some((depth, fact)) if depth == slot.path.size => fact
          case Some((_, fact)) => pointerFromMemory(fact, tpe, s"pointer field ${slot.label}")
          case None if localAggregates(slot.root) =>
            val known = state.keysIterator
              .collect {
                case candidate: Query.Slot if candidate.root == slot.root => candidate.label
              }
              .toList
              .sorted
            AddressValue(obligations =
              Set(s"local aggregate slot ${slot.label} has no address constraints") ++
                Option.when(known.nonEmpty)(s"known slots for ${slot.root}: ${known.mkString(", ")}")
            )
          case None => pointerFromMemory(read(state, tokenOf(root)), tpe, s"pointer field ${slot.label}")
        }
      case _ if isPtr(term.tpe) => AddressValue.unresolved(s"unsupported pointer term ${term.repr}")
      case _                    => AddressValue()
    }

    def addressFact(
        state: Map[Query[AddressValue], AddressValue],
        term: p.Term,
        pointsToBinding: Boolean,
        materialisesStorageAddress: Boolean,
        space: p.Type.Space
    ): AddressValue =
      term match {
        // Pointer arithmetic on a pointer-valued field follows the stored pointer. Only a RefTo whose component is
        // the pointer itself materialises the address of the field's storage.
        case selected: p.Term.Select if selected.steps.nonEmpty && isPtr(selected.tpe) && !pointsToBinding =>
          termFact(state, selected).copy(references = Set.empty)
        case p.Term.Select(root, Nil, tpe) if isPtr(tpe) && pointsToBinding =>
          AddressValue
            .absolute(Some(Provenance.Local(root.symbol)), Some(space))
            .copy(references = Set(tokenOf(root)))
        case p.Term.Select(root, Nil, tpe) if isPtr(tpe) =>
          state.get(Query.Binding(root.symbol)) match {
            case Some(fact) => fact.copy(references = Set.empty)
            case None =>
              AddressValue.absolute(Some(Provenance.Parameter(root.symbol)), spaceOf(root.tpe))
          }
        case p.Term.Select(root, steps, _) =>
          slotFact(state, root, steps).map(_._2.copy(references = Set.empty)).getOrElse {
            if (localAggregates(root.symbol) || !isPtr(root.tpe))
              AddressValue.absolute(
                Some(Provenance.Local(root.symbol, steps)),
                Some(
                  if (materialisesStorageAddress && localAggregates(root.symbol))
                    root.tpe match {
                      case p.Type.Arr(_, _, declared) if declared != p.Type.Space.Global => declared
                      case _                                                             => p.Type.Space.Private
                    }
                  else
                    root.tpe match {
                      case _: p.Type.Struct => p.Type.Space.Private
                      case _                => spaceOf(root.tpe).getOrElse(space)
                    }
                )
              )
            else {
              val base              = read(state, tokenOf(root))
              val (arena, nonArena) = classifyProvenance(base.provenances)
              if (arena && nonArena)
                AddressValue.unresolved(s"address through ${root.symbol} has mixed memory origins")
              else if (arena)
                AddressValue.arenaRelative(
                  base.provenances.collect { case Provenance.ArenaRoot(capture) => capture }
                )
              else if (nonArena)
                base.mapAbsolute { (provenance, addressSpace) =>
                  AbstractAddress.Absolute(
                    provenance.map {
                      case Provenance.Local(symbol, prefix) =>
                        val suffix = if (steps.headOption.contains(p.PathStep.Deref)) steps.tail else steps
                        Provenance.Local(symbol, prefix ++ suffix)
                      case other => other
                    },
                    addressSpace
                  )
                }
              else AddressValue(obligations = base.obligations + s"address through ${root.symbol} has no known origin")
            }
          }
        case _ =>
          AddressValue.absolute(Some(Provenance.Local(s"materialised ${term.repr}")), Some(space))
      }

    def exprFact(state: Map[Query[AddressValue], AddressValue], expr: p.Expr): AddressValue = expr match {
      case p.Expr.Alias(term) => termFact(state, term)
      case p.Expr.Cast(term: p.Term.Select, p.Type.Ptr(_, space)) if !isCarrier(term.tpe) =>
        val materialisesStorageAddress = term.tpe match {
          case _: p.Type.Arr => true
          case _             => false
        }
        addressFact(state, term, pointsToBinding = false, materialisesStorageAddress, space)
      case p.Expr.Cast(term, _) => termFact(state, term)
      case p.Expr.RefTo(
            p.Term.Select(root, Nil, p.Type.Ptr(component, _)),
            None,
            comp,
            _,
            _
          ) if capture.exists(_.symbol == root.symbol) && component == comp =>
        AddressValue.arenaRoot(root.symbol)
      case p.Expr.RefTo(term, index, comp, space, _) =>
        addressFact(
          state,
          term,
          index.isEmpty && isPtr(term.tpe) && comp == term.tpe,
          materialisesStorageAddress = true,
          space
        )
      case p.Expr.Alloc(_, _, space, _) =>
        AddressValue.absolute(Some(Provenance.Allocation), Some(space))
      case p.Expr.Index(p.Term.Select(root, _, _), _, comp)
          if isAddressInt(comp) && logicalArenaAddressViews(root.symbol) =>
        AddressValue.arenaRelative(Set(root.symbol))
      case p.Expr.Index(base, _, comp) if isPtr(comp) =>
        val fact = termFact(state, base)
        if (fact.references.nonEmpty)
          fact.references.iterator.map(read(state, _)).foldLeft(AddressValue())(_.join(_))
        else pointerFromMemory(fact, comp, s"pointer load from ${base.repr}")
      case p.Expr.IntrOp(p.Intr.Add(x, y, _)) if carrierResult(expr) =>
        val left  = termFact(state, x)
        val right = termFact(state, y)
        if (isScalarConstant(y)) left
        else if (isScalarConstant(x)) right
        else if (left.spaces.nonEmpty && right.spaces.isEmpty) left
        else if (right.spaces.nonEmpty && left.spaces.isEmpty) right
        else left.join(right)
      case p.Expr.IntrOp(p.Intr.Sub(x, y, _)) if carrierResult(expr) =>
        val left  = termFact(state, x)
        val right = termFact(state, y)
        if (isScalarConstant(y)) left
        else if (left.spaces.nonEmpty && right.spaces.nonEmpty)
          AddressValue.absolute()
        else if (left.spaces.nonEmpty) left
        else AddressValue.absolute()
      case _: p.Expr.ForeignCall if carrierResult(expr) =>
        AddressValue.unresolved(s"foreign call ${expr.repr} has no pointer summary")
      case invoke: p.Expr.Invoke if carrierResult(expr) =>
        val candidates = invoke.calleeSym.toList.flatMap { name =>
          (program.entry.toList ::: program.functions).filter(_.name == name)
        }
        val actuals = invoke.receiver.toList ::: invoke.args
        val callee = candidates
          .find { candidate =>
            val formals = candidate.receiver.toList.map(_.named) ::: candidate.args.map(_.named)
            formals.map(_.tpe) == actuals.map(_.tpe) && candidate.rtn == invoke.rtn
          }
          .orElse(Option.when(candidates.size == 1)(candidates.head))
        callee match {
          case None => AddressValue.unresolved(s"call ${expr.repr} has no pointer summary")
          case Some(target) if activeCalls.contains(target.signatureKey) =>
            AddressValue.unresolved(s"recursive call ${expr.repr} has no converged pointer summary")
          case Some(target) =>
            val formals     = target.receiver.toList.map(_.named) ::: target.args.map(_.named)
            val actualFacts = actuals.map(term => selectValue(termFact(state, term)))
            val inherited = (target.moduleCaptures.iterator ++ target.termCaptures.iterator)
              .map(_.named)
              .flatMap(formal => state.get(Query.Binding(formal.symbol)).map(formal.symbol -> _))
              .toMap
            val overrides = inherited ++ formals.iterator.map(_.symbol).zip(actualFacts).toMap
            val key       = CallKey(target.signatureKey, overrides.toList.sortBy(_._1), model)
            context.calls.getOrElseUpdate(
              key,
              solve(program, target, model, overrides, target.signatureKey :: activeCalls, context).returned
            )
        }
      case p.Expr.SpecOp(_: p.Spec.RemoteAlloc) =>
        AddressValue.absolute(Some(Provenance.Allocation), Some(p.Type.Space.Global))
      case _ if isAddressInt(expr.tpe) => AddressValue()
      case _ if carrierResult(expr)    => AddressValue.unresolved(s"unsupported pointer result ${expr.repr}")
      case _                           => AddressValue()
    }

    def isScalarConstant(term: p.Term): Boolean = term match {
      case _: p.Term.IntU64Const | _: p.Term.IntS64Const => true
      case _                                             => false
    }

    val initial = refineEncodings(seeds)
    val limit   = (pending.size + slotCopies.size + storage.size + 1) * 4

    def targetOf(
        assignment: Pending,
        state: Map[Query[AddressValue], AddressValue]
    ): Query[AddressValue] = assignment.slot match {
      case Some((root, steps)) => canonicalSlot(state, root, steps)
      case None                => assignment.target
    }

    @annotation.tailrec
    def close(
        current: Map[Query[AddressValue], AddressValue],
        remaining: Int
    ): (Map[Query[AddressValue], AddressValue], Map[String, String]) = {
      // This is a constraint closure, not a control-flow transfer. Retain facts learned by earlier
      // iterations so canonical slot aliases can only add evidence rather than oscillating as an
      // alias becomes known and changes where the same constraint is recorded.
      val assigned = pending.foldLeft(current) { (facts, assignment) =>
        val target = targetOf(assignment, current)
        // Missing-evidence messages describe the current approximation, not lattice facts. Carrying them through
        // closure makes a resolved pointer graph alternate between "not known yet" and its eventual origin.
        // Re-evaluate those obligations against the closed refinement below instead.
        val source = exprFact(current, assignment.expr).copy(obligations = Set.empty)
        facts.updated(
          target,
          facts.getOrElse(target, AddressValue()).join(source)
        )
      }
      val nextRaw = slotCopies.foldLeft(assigned) { (facts, copy) =>
        val source = canonicalSlot(current, copy.source, copy.sourcePrefix)
        val target = canonicalSlot(current, copy.target, copy.targetPrefix)
        current.iterator
          .collect {
            case (Query.Slot(root, path), fact) if root == source.root && slotPrefix(source.path, path) =>
              Query.Slot(target.root, target.path ++ path.drop(source.path.size)) -> fact.copy(obligations = Set.empty)
          }
          .filter { case (slot, _) => slot.path.size <= maxSlotDepth }
          .foldLeft(facts) { case (acc, (slot, fact)) =>
            acc.updated(slot, acc.getOrElse(slot, AddressValue()).join(fact))
          }
      }
      val next = refineEncodings(nextRaw)
      if (next == current) next -> Map.empty
      else if (remaining == 0)
        next -> (next.keySet ++ current.keySet).iterator
          .filter(token => next.get(token) != current.get(token))
          .map(token => token.label -> s"before=${current.get(token)}; after=${next.get(token)}")
          .toMap
      else close(next, remaining - 1)
    }
    val (solved, unstable) = close(initial, limit)

    def coercion(target: Query[AddressValue], source: AddressValue): Either[Diagnostic, Coercion] = {
      val targetFact = solved.getOrElse(target, AddressValue())
      val sourceRep  = source.encoding.orElse(source.encodings.headOption)
      (targetFact.encoding, sourceRep) match {
        case (_, None) if source.includesNull                        => Right(Coercion.Preserve)
        case (Some(Encoding.Absolute), Some(Encoding.ArenaRelative)) => Right(Coercion.ResolveRelative)
        case (Some(Encoding.ArenaRelative), Some(Encoding.Absolute)) =>
          val (arena, other) = classifyProvenance(source.provenances)
          if (arena && !other) Right(Coercion.EncodeRelative)
          else
            Left(
              Diagnostic(
                "absolute-to-arena-slot",
                s"${target.label} stores an absolute address unrelated to its capture arena",
                source.provenances.toList.map(x => s"source origin: $x")
              )
            )
        case (Some(x), Some(y)) if x != y =>
          Left(Diagnostic("incompatible-encoding", s"${target.label} requires $x but its source is $y"))
        case _ => Right(Coercion.Preserve)
      }
    }

    val assignmentResults = pending.map { item =>
      val target = targetOf(item, solved)
      val source = selectValue(exprFact(solved, item.expr))
      (item, target) -> (source -> coercion(target, source))
    }
    val coercions = assignmentResults.collect { case ((item, target), (source, Right(c))) =>
      CoercionSite(item.index, target, source, c, item.context)
    }
    val conversionDiagnostics = assignmentResults.collect { case (_, (_, Left(diagnostic))) => diagnostic }

    def pointerSlots(tpe: p.Type, prefix: List[p.PathStep], remaining: Int): List[(List[p.PathStep], p.Type)] =
      if (remaining == 0) Nil
      else
        tpe match {
          case p.Type.Struct(symbol, _) =>
            members.getOrElse(symbol, Map.empty).toList.flatMap { case (name, memberType) =>
              val path = prefix :+ p.PathStep.Field(name)
              if (isPtr(memberType)) (path -> memberType) :: Nil
              else pointerSlots(memberType, path, remaining - 1)
            }
          case p.Type.Arr(component, size, _) =>
            (0 until size).toList.flatMap(index =>
              pointerSlots(component, prefix :+ p.PathStep.Index(index), remaining - 1)
            )
          case _ => Nil
        }

    val aggregateParameterSeeds = bound.iterator
      .filterNot(n => isPtr(n.tpe))
      .flatMap { root =>
        pointerSlots(root.tpe, Nil, maxSlotDepth).map { case (path, tpe) =>
          (Query.Slot(root.symbol, path): Query[AddressValue]) ->
            AddressValue.absolute(Some(Provenance.Parameter(root.symbol)), spaceOf(tpe))
        }
      }
      .toMap

    def storedFact(target: Query[AddressValue], source: AddressValue): AddressValue = {
      val encoding = solved.get(target).flatMap(_.encoding).orElse(selectValue(source).encoding)
      encoding.fold(source)(source.inEncoding)
    }

    def mergeFacts(left: AddressValue, right: AddressValue): AddressValue = {
      val joined = left.join(right)
      joined.copy(encoding =
        if (left.encoding == right.encoding) left.encoding
        else selectValue(joined).encoding
      )
    }

    def mergeStates(
        left: Map[Query[AddressValue], AddressValue],
        right: Map[Query[AddressValue], AddressValue]
    ): Map[Query[AddressValue], AddressValue] =
      (left.keySet ++ right.keySet).iterator.map { token =>
        val fact = (left.get(token), right.get(token)) match {
          case (Some(x), Some(y)) => mergeFacts(x, y)
          case (Some(x), None)    => x
          case (None, Some(y))    => y
          case _                  => AddressValue()
        }
        token -> fact
      }.toMap

    def mergeSummaries(
        left: Map[String, AddressValue],
        right: Map[String, AddressValue]
    ): Map[String, AddressValue] =
      (left.keySet ++ right.keySet).iterator.map { symbol =>
        val fact = (left.get(symbol), right.get(symbol)) match {
          case (Some(x), Some(y)) => mergeFacts(x, y)
          case (Some(x), None)    => x
          case (None, Some(y))    => y
          case _                  => AddressValue()
        }
        symbol -> fact
      }.toMap

    def copySlots(
        state: Map[Query[AddressValue], AddressValue],
        targetRoot: p.Named,
        targetPrefix: List[p.PathStep],
        sourceRoot: p.Named,
        sourcePrefix: List[p.PathStep]
    ): Map[Query[AddressValue], AddressValue] = {
      val source = canonicalSlot(state, sourceRoot, sourcePrefix)
      val target = canonicalSlot(state, targetRoot, targetPrefix)
      val copied = state.iterator.collect {
        case (Query.Slot(root, path), fact)
            if root == source.root && slotPrefix(source.path, path) &&
              target.path.size + path.size - source.path.size <= maxSlotDepth =>
          Query.Slot(target.root, target.path ++ path.drop(source.path.size)) -> fact
      }.toMap
      state ++ copied
    }

    type FlowState = Map[Query[AddressValue], AddressValue]
    type Summary   = Map[String, AddressValue]

    def unresolvedFact(label: String, fact: AddressValue, context: String): Option[Diagnostic] =
      if (model == AddressModel.Logical && fact.spaces.size > 1)
        Some(
          Diagnostic(
            "incompatible-address-spaces",
            s"$label may denote pointers in incompatible address spaces in $context",
            fact.alternatives.toList
              .map {
                case address @ AbstractAddress.Absolute(Some(Provenance.Local(symbol, _)), _) =>
                  val declared = declarations.get(symbol).map(_.tpe.repr).getOrElse("<unknown>")
                  s"possible address: $address (storage type: $declared)"
                case address => s"possible address: $address"
              }
              .sortBy(identity)
          )
        )
      else
        Option.when(fact.obligations.nonEmpty || fact.encoding.isEmpty) {
          Diagnostic(
            "unresolved-pointer-use",
            s"cannot determine the address encoding of $label in $context",
            fact.obligations.toList.sorted
          )
        }

    def unresolvedUse(state: FlowState, term: p.Term, context: String): Option[Diagnostic] =
      unresolvedFact(term.repr, selectValue(termFact(state, term)), context)

    def pointerOperand(term: p.Term): Option[p.Term] = term match {
      case pointer if isPtr(pointer.tpe)                => Some(pointer)
      case p.Term.Select(root, _, _) if isPtr(root.tpe) => Some(p.Term.Select(root, Nil, root.tpe))
      case _                                            => None
    }

    def pointee(tpe: p.Type): p.Type = tpe match {
      case p.Type.Ptr(component, _)    => component
      case p.Type.Arr(component, _, _) => component
      case other                       => other
    }
    def stepType(current: p.Type, step: p.PathStep): p.Type = step match {
      case p.PathStep.Field(name) =>
        pointee(current) match {
          case p.Type.Struct(symbol, _) => members.get(symbol).flatMap(_.get(name)).getOrElse(p.Type.Nothing)
          case _                        => p.Type.Nothing
        }
      case p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn => pointee(current)
    }

    def traversedPointers(term: p.Term.Select): List[p.Term] = {
      val (_, _, pointers) =
        term.steps.zipWithIndex.foldLeft((term.root.tpe, List.empty[p.PathStep], List.empty[p.Term])) {
          case ((current, prefix, found), (step, index)) =>
            val nextPrefix = prefix :+ step
            val nextType   = stepType(current, step)
            val next =
              if (isPtr(nextType) && index < term.steps.size - 1)
                found :+ p.Term.Select(term.root, nextPrefix, nextType)
              else found
            (nextType, nextPrefix, next)
        }
      pointers
    }

    def diagnosticsIn(stmt: p.Stmt, state: FlowState): List[Diagnostic] = {
      val terms = stmt.collectAll[p.Term].flatMap {
        case term @ p.Term.Select(root, steps, _) if steps.nonEmpty =>
          val rootUse = Option.when(isPtr(root.tpe))(p.Term.Select(root, Nil, root.tpe))
          (rootUse.toList ++ traversedPointers(term))
            .flatMap(pointer => unresolvedUse(state, pointer, s"stepped select ${term.repr}"))
        case _ => None
      }
      val expressions = stmt.collectAll[p.Expr].flatMap {
        case p.Expr.Index(base, _, _) => pointerOperand(base).flatMap(unresolvedUse(state, _, "index"))
        case p.Expr.SpecOp(p.Spec.GpuAtomicRMW(_, ptr, _, _, _, _)) =>
          unresolvedUse(state, ptr, "atomic operation")
        case p.Expr.SpecOp(p.Spec.GpuAtomicCAS(ptr, _, _, _, _, _)) =>
          unresolvedUse(state, ptr, "atomic operation")
        case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(ptr, _))  => unresolvedUse(state, ptr, "volatile load")
        case p.Expr.SpecOp(p.Spec.GpuVolatileStore(ptr, _)) => unresolvedUse(state, ptr, "volatile store")
        case _                                              => None
      }
      val direct = stmt match {
        case p.Stmt.Mut(p.Term.Select(name, Nil, tpe), p.Expr.Alias(value))
            if isAddressInt(tpe) && isScalarConstant(value) &&
              state
                .get(Query.Binding(name.symbol))
                .exists(_.provenances.exists { case _: Provenance.ArenaRoot => true; case _ => false }) =>
          List(
            Diagnostic(
              "incompatible-address-encoding",
              s"address encoding changes between an arena-relative address and a scalar in ${name.symbol}"
            )
          )
        case p.Stmt.Update(lhs, _, _) => pointerOperand(lhs).flatMap(unresolvedUse(state, _, "update")).toList
        case p.Stmt.Return(value) if isPtr(function.rtn) =>
          unresolvedFact(value.repr, selectValue(exprFact(state, value)), "exported return").toList
        case _ => Nil
      }
      terms ::: expressions ::: direct
    }

    def transfer(stmt: p.Stmt, state: FlowState): (FlowState, Summary) = stmt match {
      case p.Stmt.Var(name, Some(expr), _) if isCarrier(name.tpe) =>
        val target = Query.Binding(name.symbol)
        val value  = storedFact(target, selectValue(exprFact(state, expr)))
        state.updated(target, value) -> Option.when(isPtr(name.tpe))(name.symbol -> value).toMap
      case p.Stmt.Mut(p.Term.Select(name, Nil, tpe), expr) if isCarrier(tpe) =>
        val target = Query.Binding(name.symbol)
        val value  = storedFact(target, selectValue(exprFact(state, expr)))
        state.updated(target, value) -> Option.when(isPtr(tpe))(name.symbol -> value).toMap
      case p.Stmt.Mut(p.Term.Select(root, steps, tpe), expr) if steps.nonEmpty && isPtr(tpe) =>
        val target = canonicalSlot(state, root, steps)
        state.updated(target, storedFact(target, selectValue(exprFact(state, expr)))) -> Map.empty
      case p.Stmt.Var(target, Some(p.Expr.Alias(p.Term.Select(source, prefix, _))), _) if !isPtr(target.tpe) =>
        copySlots(state, target, Nil, source, prefix) -> Map.empty
      case p.Stmt.Mut(
            p.Term.Select(target, targetPrefix, tpe),
            p.Expr.Alias(p.Term.Select(source, sourcePrefix, _))
          ) if !isPtr(tpe) =>
        copySlots(state, target, targetPrefix, source, sourcePrefix) -> Map.empty
      case _ => state -> Map.empty
    }

    def analyse(stmts: List[p.Stmt], initial: FlowState): (FlowState, Summary, Boolean, List[Diagnostic]) =
      stmts.foldLeft((initial, Map.empty[String, AddressValue], true, List.empty[Diagnostic])) {
        case ((state, summary, converged, diagnostics), stmt) =>
          val (next, added, stable, found) = stmt match {
            case p.Stmt.Cond(_, whenTrue, whenFalse) =>
              val (trueState, trueSummary, trueStable, trueDiagnostics)     = analyse(whenTrue, state)
              val (falseState, falseSummary, falseStable, falseDiagnostics) = analyse(whenFalse, state)
              (
                mergeStates(trueState, falseState),
                mergeSummaries(trueSummary, falseSummary),
                trueStable && falseStable,
                trueDiagnostics ::: falseDiagnostics
              )
            case p.Stmt.While(_, body)             => flowLoop(body, state)
            case p.Stmt.ForRange(_, _, _, _, body) => flowLoop(body, state)
            case p.Stmt.Try(body, handlers, fin) =>
              val (bodyState, bodySummary, bodyStable, bodyDiagnostics) = analyse(body, state)
              val (handledState, handledSummary, handledStable, handledDiagnostics) = handlers.foldLeft(
                (bodyState, bodySummary, bodyStable, bodyDiagnostics)
              ) { case ((currentState, currentSummary, currentStable, currentDiagnostics), handler) =>
                val (handlerState, handlerSummary, handlerStable, handlerDiagnostics) =
                  analyse(handler.body, mergeStates(state, bodyState))
                (
                  mergeStates(currentState, handlerState),
                  mergeSummaries(currentSummary, handlerSummary),
                  currentStable && handlerStable,
                  currentDiagnostics ::: handlerDiagnostics
                )
              }
              val (finalState, finalSummary, finalStable, finalDiagnostics) = analyse(fin, handledState)
              (
                finalState,
                mergeSummaries(handledSummary, finalSummary),
                handledStable && finalStable,
                handledDiagnostics ::: finalDiagnostics
              )
            case p.Stmt.Annotated(inner, _, _) => analyse(List(inner), state)
            case p.Stmt.Raise(_, _, cleanup)   => analyse(cleanup, state)
            case leaf =>
              val (leafState, leafSummary) = transfer(leaf, state)
              (leafState, leafSummary, true, diagnosticsIn(leaf, state))
          }
          (next, mergeSummaries(summary, added), converged && stable, diagnostics ::: found)
      }

    def flowLoop(body: List[p.Stmt], entry: FlowState): (FlowState, Summary, Boolean, List[Diagnostic]) = {
      @annotation.tailrec
      def loop(
          current: FlowState,
          summary: Summary,
          diagnostics: List[Diagnostic],
          remaining: Int
      ): (FlowState, Summary, Boolean, List[Diagnostic]) = {
        val (bodyState, bodySummary, nestedStable, bodyDiagnostics) = analyse(body, current)
        val next                                                    = mergeStates(entry, bodyState)
        val nextSummary                                             = mergeSummaries(summary, bodySummary)
        val nextDiagnostics                                         = (diagnostics ::: bodyDiagnostics).distinct
        if (next == current) (next, nextSummary, nestedStable, nextDiagnostics)
        else if (remaining == 0) (next, nextSummary, false, nextDiagnostics)
        else loop(next, nextSummary, nextDiagnostics, remaining - 1)
      }
      loop(entry, Map.empty, Nil, limit)
    }

    val flowSeeds = seeds.iterator.map { case (token, fact) =>
      // StructuredExit payload slots are a tagged union: they are only read after the matching tag is written.
      val pendingExceptionPayload = token match {
        case Query.Binding(symbol) => symbol.startsWith(StructuredExit.ExceptionSlotPrefix)
        case _                     => false
      }
      val initialised =
        fact.encodings.nonEmpty || fact.provenances.nonEmpty || fact.includesNull || pendingExceptionPayload
      token -> Option
        .when(!initialised) {
          fact.copy(obligations = Set(s"${token.label} may be uninitialised on this path"))
        }
        .getOrElse(fact)
    }.toMap
    val flowInitial                                      = refineEncodings(flowSeeds ++ aggregateParameterSeeds)
    val (_, flowSummary, flowConverged, flowDiagnostics) = analyse(function.body, flowInitial)
    val refinedSpaces = flowSummary.iterator.collect {
      case (symbol, fact) if fact.spaces.nonEmpty => symbol -> fact.spaces
    }.toMap
    val carrierDiagnostics = solved.iterator.collect {
      case (token, fact)
          if storage.get(token).exists(info => isAddressInt(info.tpe)) &&
            fact.encodings.contains(Encoding.Absolute) &&
            fact.encodings.contains(Encoding.ArenaRelative) =>
        Diagnostic(
          "incompatible-address-encoding",
          s"address encoding changes between an arena-relative address and a scalar in ${token.label}"
        )
    }.toList
    val logicalDiagnostics =
      if (model != AddressModel.Logical) Nil
      else
        solved.iterator.collect {
          case (token, fact)
              if fact.hasNonArenaAbsolute &&
                (fact.hasArenaRoot || fact.encodings.contains(Encoding.ArenaRelative)) =>
            Diagnostic(
              "logical-mixed-encoding",
              s"${token.label} cannot hold both an arena-relative and an absolute address"
            )
        }.toList
    val convergenceDiagnostics =
      Option
        .when(unstable.nonEmpty) {
          Diagnostic(
            "address-refinement-did-not-converge",
            s"address refinement exceeded $limit iterations in ${function.name.repr}",
            unstable.toList.sortBy(_._1).map { case (label, evidence) => s"unstable storage: $label ($evidence)" }
          )
        }
        .toList ::: Option
        .when(!flowConverged) {
          Diagnostic(
            "address-flow-did-not-converge",
            s"address-flow refinement exceeded $limit iterations in ${function.name.repr}"
          )
        }
        .toList

    val diagnostics = (
      convergenceDiagnostics ++ conversionDiagnostics ++ carrierDiagnostics ++ logicalDiagnostics ++
        flowDiagnostics
    ).distinct
    val valueEntries = function.collectAll[p.Term].iterator.map { term =>
      (Query.TermValue(term): Query[?]) -> selectValue(termFact(solved, term))
    }
    val producerEntries = function.collectAll[p.Expr].iterator.map { expr =>
      (Query.ExprResult(expr): Query[?]) -> selectValue(exprFact(solved, expr))
    }
    val addressEntries = function.collectAll[p.Term].iterator.map { term =>
      val origins = term match {
        case p.Term.Select(root, steps, _) if localAggregates(root.symbol) =>
          Set[Provenance](Provenance.Local(root.symbol, steps))
        case p.Term.Select(root, _, _) =>
          solved.getOrElse(Query.Binding(root.symbol), AddressValue()).provenances
        case _ => Set.empty[Provenance]
      }
      (Query.AddressOf(term): Query[?]) -> origins
    }
    val entries: Map[Query[?], Any] = (
      solved.iterator.map { case (token, fact) => (token: Query[?]) -> fact } ++
        valueEntries ++ producerEntries ++ addressEntries
    ).toMap
    val bindings = solved.iterator.collect { case (Query.Binding(symbol), fact) => symbol -> fact }.toMap
    val slots    = solved.iterator.collect { case (slot: Query.Slot, fact) => slot -> fact }.toMap
    val returned = function
      .collectAll[p.Stmt]
      .collect { case stmt: p.Stmt.Return => stmt }
      .iterator
      .map(stmt => selectValue(exprFact(solved, stmt.value)))
      .reduceOption(_.join(_))
      .map(selectValue)
      .getOrElse(AddressValue())
    Solution(function, model, FactTable(entries), bindings, slots, refinedSpaces, coercions, returned, diagnostics)
  }
}
