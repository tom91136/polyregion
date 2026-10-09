package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ListBuffer

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

private[pass] object LogicalArenaViewAbi {
  private val Global = p.Type.Space.Global

  val bindings: List[p.Named] =
    List(
      p.Type.IntS8,
      p.Type.IntS16,
      p.Type.IntS32,
      p.Type.IntS64,
      p.Type.Float32,
      p.Type.Float64,
      p.Type.Float16
    ).zipWithIndex
      .map((tpe, index) => p.Named(s"#av$index", p.Type.Ptr(tpe, Global)))

  val addressBinding: p.Named = bindings
    .collectFirst { case binding @ p.Named(_, p.Type.Ptr(p.Type.IntS64, p.Type.Space.Global), _) =>
      binding
    }
    .getOrElse(throw IllegalStateException("logical arena-view ABI lacks its address view"))

  def arenaAddressBindings(bound: Iterable[p.Named]): Set[String] = {
    val actual = bound.iterator.map(binding => binding.symbol -> binding.tpe).toMap
    Option
      .when(bindings.forall(binding => actual.get(binding.symbol).contains(binding.tpe)))(Set(addressBinding.symbol))
      .getOrElse(Set.empty)
  }
}

// generic single-arena lowering for logical SPIR-V (Vulkan glcompute): no flat address space, no int<->ptr
// cast, no pointer load/store/offset. the capture sits at arena offset 0 and every pointer is an i64 byte
// offset; each deref reads/writes through a fixed roster of typed scalar "view" descriptors indexed by
// `offset / sizeof(elem)`, the only legal access form. pointer struct fields are retyped to i64 so an arena
// object's `_M_p` and a local iterator's `_M_current` are uniform
// examples:
//   cap                         ->  0                                      // capture is arena offset 0
//   cap.x   (scalar field)      ->  view_i32[offsetof(cap, x) / 4]         // read/write via the typed view
//   p[i]    (p arena offset)    ->  view_T[(p + i*sizeof T) / sizeof T]    // arena-relative deref
//   it._M_current               ->  the i64 offset directly (field retyped)
//   struct value read           ->  local copy, scalar leaves filled from views (loadAgg)
//   local `p = &x; s.ptr = p`   ->  stable i64 identity token when s.ptr is only stored/compared
// edge cases:
//   pointer Rooted at a stack local  ->  stays a real pointer (e.g. inlined std::min over two locals)
//   local pointer field dereference  ->  not tokenised; only identity-only fields are eligible
//   Float16 field                    ->  own f16 view (numeric Cast can't bitcast; i16 view would convert)
//   ForRange bound / Cond cond       ->  stepped Select hoisted into a Var first (hoistInlineTerms)
//   reduction scratch arg            ->  kept leading, a real workgroup pointer ahead of the views
object ArenaView extends ProgramPass {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private val ctr = new AtomicLong(0L)

  private val Global                     = p.Type.Space.Global
  private def isPtr(t: p.Type): Boolean  = t match { case _: p.Type.Ptr => true; case _ => false }
  private def pointee(t: p.Type): p.Type = t match { case p.Type.Ptr(c, _) => c; case _ => t }
  private def elem(t: p.Type): p.Type = t match {
    case p.Type.Ptr(c, _) => c; case p.Type.Arr(c, _, _) => c; case _ => t
  }
  // a Global pointer (or array-of-pointer) struct field holds an arena byte offset, so retype to i64
  // (same layout). Private pointer fields belong to stack-local aggregates and remain real pointers.
  private def i64ify(t: p.Type): p.Type = t match {
    case p.Type.Ptr(_, p.Type.Space.Global) => I64
    case p.Type.Ptr(c, s)                   => p.Type.Ptr(i64ify(c), s)
    case p.Type.Arr(c, n, s)                => p.Type.Arr(i64ify(c), n, s)
    case _                                  => t
  }

  private type Field = (p.Sym, String)

  private def fieldsAt(
      rootTpe: p.Type,
      steps: List[p.PathStep],
      members: Map[p.Sym, List[p.Named]]
  ): List[(Field, p.Type)] = {
    def member(sym: p.Sym, field: String): Option[p.Type] =
      members.get(sym).flatMap(_.find(_.symbol == field).map(_.tpe))
    steps
      .foldLeft((rootTpe, List.empty[(Field, p.Type)])) {
        case ((cur, found), p.PathStep.Field(field)) =>
          val owner = pointee(cur) match { case p.Type.Struct(sym, _) => Some(sym); case _ => None }
          owner.flatMap(sym => member(sym, field).map(t => (t, (sym -> field, t)))) match {
            case Some((t, resolved)) => (t, found :+ resolved)
            case None                => (cur, found)
          }
        case ((cur, found), p.PathStep.Deref)       => (pointee(cur), found)
        case ((cur, found), p.PathStep.Index(_))    => (elem(cur), found)
        case ((cur, found), p.PathStep.IndexDyn(_)) => (elem(cur), found)
      }
      ._2
  }

  private def fieldAt(rootTpe: p.Type, steps: List[p.PathStep], members: Map[p.Sym, List[p.Named]]): Option[Field] =
    fieldsAt(rootTpe, steps, members).lastOption.map(_._1)

  private def staticIndex(index: Option[p.Term]): Option[String] = index match {
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

  private def staticPathKey(steps: List[p.PathStep]): Option[String] =
    if (steps.exists(_.isInstanceOf[p.PathStep.IndexDyn])) None
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

  private def localReferenceKey(base: p.Term.Select, index: Option[p.Term]): Option[String] =
    for {
      path <- staticPathKey(base.steps)
      idx  <- staticIndex(index)
    } yield s"${base.root.symbol}:$path:$idx"

  private def unsupportedNativeSlots(
      entry: p.Function,
      members: Map[p.Sym, List[p.Named]],
      identityFields: Set[Field],
      analysis: AddressRefinement.Solution
  ): List[(p.Named, String)] = {
    import AddressRefinement.{Encoding, Query}

    val locals = entry.collectAll[p.Stmt].collect { case p.Stmt.Var(n, _, _) => n.symbol -> n }.toMap
    analysis.slots.iterator
      .flatMap { case (Query.Slot(root, path), fact) =>
        for {
          local <- locals.get(root)
          field <- fieldsAt(local.tpe, path, members).lastOption
          if field._2 match {
            case p.Type.Ptr(_, p.Type.Space.Global) => true
            case _                                  => false
          }
          if fact.encoding.contains(Encoding.Absolute)
          if !identityFields(field._1)
        } yield local -> s"$root.${path.mkString(".")}"
      }
      .toList
      .sortBy(_._2)
  }

  // A pointer that is only ever the address of a local (sub)aggregate and only reaches fields is that aggregate: C++
  // constructors and member calls write through `this` this way. Resolving it keeps the aggregate's address untaken.
  private def resolveSelfPointers(entry: p.Function, members: Map[p.Sym, List[p.Named]]): p.Function = {
    val arguments  = entry.args.map(_.named).toSet
    val reassigned = AddressRefinement.reassignedIn(entry)
    val stmts      = entry.collectAll[p.Stmt].map(core)
    object AliasOf {
      def unapply(expr: p.Expr): Option[p.Term.Select] = expr match {
        case p.Expr.Alias(select @ p.Term.Select(_, Nil, _))                           => Some(select)
        case p.Expr.Cast(select @ p.Term.Select(_, Nil, _: p.Type.Ptr), _: p.Type.Ptr) => Some(select)
        case _                                                                         => None
      }
    }
    val candidates = doUntilNotEq(Map.empty[p.Named, (p.Named, List[p.PathStep])]) { (_, known) =>
      known ++ stmts.collect {
        case p.Stmt.Var(pointer, Some(p.Expr.RefTo(p.Term.Select(root, path, _), None, _, _, _)), _)
            if !reassigned(pointer.symbol) && !arguments(root) =>
          pointer -> (root -> path)
        case p.Stmt.Var(pointer, Some(AliasOf(source)), _)
            if !reassigned(pointer.symbol) && known.contains(source.root) =>
          pointer -> known(source.root)
      }
    }._2
    // a pointer reached through casts only stands for the aggregate when it is read as that aggregate's own type
    def readsAsTarget(pointer: p.Named, target: (p.Named, List[p.PathStep])) = pointer.tpe match {
      case p.Type.Ptr(component, _) => typeAt(target._1.tpe, target._2, members).contains(component)
      case _                        => false
    }
    val resolved = doUntilNotEq(candidates) { (_, known) =>
      val aliasSources = stmts.collect {
        case p.Stmt.Var(pointer, Some(AliasOf(source)), _) if known.contains(pointer) => source
      }.toSet
      val invalid = entry
        .collectAll[p.Term]
        .collect {
          case select @ p.Term.Select(root, steps, _) if known.contains(root) =>
            val fieldAccess = steps.headOption.exists {
              case _: p.PathStep.Field => true
              case _                   => false
            }
            Option.when(!(fieldAccess && readsAsTarget(root, known(root))) && !aliasSources.contains(select))(root)
        }
        .flatten
        .toSet
      known.filter { case (pointer, (root, _)) => !invalid(pointer) && !known.contains(root) }
    }._2
    if (resolved.isEmpty) entry
    else {
      def rewrite(term: p.Term): p.Term = term match {
        case p.Term.Select(root, steps, tpe) if resolved.contains(root) && steps.nonEmpty =>
          val (target, path) = resolved(root)
          p.Term.Select(target, path ++ steps, tpe)
        case other => other
      }
      def drop(stmts: List[p.Stmt]): List[p.Stmt] = stmts.filter(stmt =>
        core(stmt) match {
          case p.Stmt.Var(name, _, _) => !resolved.contains(name)
          case _                      => true
        }
      )
      val body = drop(entry.body)
        .modifyAll[p.Stmt] {
          case p.Stmt.Mut(target, expr) => p.Stmt.Mut(rewrite(target).asInstanceOf[p.Term.Select], expr)
          case p.Stmt.Update(target, index, value) =>
            p.Stmt.Update(rewrite(target).asInstanceOf[p.Term.Select], index, value)
          case p.Stmt.Cond(c, t, e)              => p.Stmt.Cond(c, drop(t), drop(e))
          case p.Stmt.While(c, b)                => p.Stmt.While(c, drop(b))
          case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, drop(b))
          case p.Stmt.Try(b, handlers, fin)      => p.Stmt.Try(drop(b), handlers, drop(fin))
          case stmt                              => stmt
        }
        .modifyAll[p.Term](rewrite)
      entry.copy(body = body)
    }
  }

  private def typeAt(tpe: p.Type, steps: List[p.PathStep], members: Map[p.Sym, List[p.Named]]): Option[p.Type] =
    steps.foldLeft(Option(tpe)) {
      case (Some(current), p.PathStep.Field(field)) =>
        pointee(current) match {
          case p.Type.Struct(symbol, _) => members.get(symbol).flatMap(_.find(_.symbol == field).map(_.tpe))
          case _                        => None
        }
      case (Some(current), p.PathStep.Deref)                             => Some(pointee(current))
      case (Some(current), _: p.PathStep.Index | _: p.PathStep.IndexDyn) => Some(elem(current))
      case (None, _)                                                     => None
    }

  // immutable pointer values nothing reads any more, typically addresses whose every holder was forwarded
  private def dropDeadValues(entry: p.Function): p.Function = doUntilNotEq(entry) { (_, current) =>
    val read = current.collectAll[p.Term].collect { case p.Term.Select(root, _, _) => root }.toSet
    def dead(stmt: p.Stmt) = core(stmt) match {
      case p.Stmt.Var(
            name @ p.Named(_, _: p.Type.Ptr, _),
            Some(_: p.Expr.Alias | _: p.Expr.RefTo | _: p.Expr.Cast),
            false
          ) =>
        !read(name)
      case _ => false
    }
    dropStatements(current, dead)
  }._2

  // locals only ever written into their own storage, such as an inlined RAII guard whose destructor became trivial
  private def writeOnlyLocals(entry: p.Function, members: Map[p.Sym, List[p.Named]]): Set[p.Named] = {
    def pure(expr: p.Expr) = expr match {
      case _: p.Expr.Alias | _: p.Expr.RefTo | _: p.Expr.Cast | _: p.Expr.IntrOp | _: p.Expr.Index => true
      case _                                                                                       => false
    }
    val selects = entry
      .collectAll[p.Term]
      .collect { case p.Term.Select(root, _, _) => root }
      .groupMapReduce(identity)(_ => 1)(_ + _)
    val stmts     = entry.collectAll[p.Stmt].map(core)
    val arguments = entry.args.map(_.named).toSet
    def ownStorage(root: p.Named, path: List[p.PathStep]) =
      path.indices.forall(i => typeAt(root.tpe, path.take(i), members).exists(!_.isInstanceOf[p.Type.Ptr]))
    val writes =
      stmts.collect { case p.Stmt.Mut(p.Term.Select(root, path, _), expr) => root -> (path, expr) }.groupMap(_._1)(_._2)
    val inits = stmts.collect { case p.Stmt.Var(name, init, _) => name -> init }.toMap
    writes.collect {
      case (root, stores)
          if !arguments(root) && inits.get(root).exists(_.forall(pure)) && selects.getOrElse(root, 0) == stores.size &&
            stores.forall((path, expr) => ownStorage(root, path) && pure(expr)) =>
        root
    }.toSet
  }

  private def dropStatements(entry: p.Function, dead: p.Stmt => Boolean): p.Function = {
    def drop(stmts: List[p.Stmt]): List[p.Stmt] = stmts.filterNot(dead)
    entry.copy(body = drop(entry.body).modifyAll[p.Stmt] {
      case p.Stmt.Cond(c, t, e)              => p.Stmt.Cond(c, drop(t), drop(e))
      case p.Stmt.While(c, b)                => p.Stmt.While(c, drop(b))
      case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, drop(b))
      case p.Stmt.Try(b, handlers, fin) =>
        p.Stmt.Try(drop(b), handlers.map(h => h.copy(body = drop(h.body))), drop(fin))
      case stmt => stmt
    })
  }

  private def core(stmt: p.Stmt): p.Stmt = stmt match {
    case p.Stmt.Annotated(inner, _, _) => core(inner)
    case other                         => other
  }

  private def globalPointerLeaves(
      tpe: p.Type,
      members: Map[p.Sym, List[p.Named]],
      seen: Set[p.Sym] = Set.empty
  ): List[List[p.PathStep]] = tpe match {
    case p.Type.Ptr(_, p.Type.Space.Global) => List(Nil)
    case p.Type.Struct(symbol, _) if !seen(symbol) =>
      members
        .getOrElse(symbol, Nil)
        .flatMap(m => globalPointerLeaves(m.tpe, members, seen + symbol).map(p.PathStep.Field(m.symbol) :: _))
    case _ => Nil
  }

  // pointer leaves of a local whose initial value the stores directly after its definition replace before any read
  private def overwrittenOnDefinition(
      entry: p.Function,
      members: Map[p.Sym, List[p.Named]]
  ): Set[(p.Named, List[p.PathStep])] = {
    val blocks = entry.body :: entry.collectAll[p.Stmt].map(core).flatMap {
      case p.Stmt.Cond(_, t, e)           => List(t, e)
      case p.Stmt.While(_, b)             => List(b)
      case p.Stmt.ForRange(_, _, _, _, b) => List(b)
      case p.Stmt.Try(b, handlers, fin)   => b :: fin :: handlers.map(_.body)
      case p.Stmt.Raise(_, _, cleanup)    => List(cleanup)
      case _                              => Nil
    }
    blocks.flatMap { block =>
      block.map(core).tails.flatMap {
        case p.Stmt.Var(root, Some(_), _) :: rest =>
          val prefixes = rest
            .takeWhile {
              case p.Stmt.Mut(p.Term.Select(`root`, _, _), expr) =>
                !expr.collectAll[p.Term].exists {
                  case p.Term.Select(`root`, _, _) => true
                  case _                           => false
                }
              case _ => false
            }
            .collect { case p.Stmt.Mut(p.Term.Select(_, prefix, _), _) => prefix }
          globalPointerLeaves(root.tpe, members).filter(leaf => prefixes.exists(leaf.startsWith(_))).map(root -> _)
        case _ => Nil
      }
    }.toSet
  }

  // native global pointer leaves of locals whose aggregates are only ever copied or written through exact paths
  private def localPointerSlots(
      entry: p.Function,
      members: Map[p.Sym, List[p.Named]]
  ): Set[(p.Named, List[p.PathStep])] = {
    val arguments = entry.args.map(_.named).toSet
    val capture   = captureRoot(entry).map(_._1)
    val stmts     = entry.collectAll[p.Stmt].map(core)
    val locals    = stmts.collect { case p.Stmt.Var(name, _, _) => name }.toSet
    val addressed = entry
      .collectAll[p.Expr]
      .collect {
        case p.Expr.RefTo(p.Term.Select(_, _, p.Type.Ptr(component, _)), _, comp, _, _) if component == comp => None
        case p.Expr.RefTo(p.Term.Select(root, _, _), _, _, _, _) => Some(root)
      }
      .flatten
      .toSet
    val copies = stmts.collect {
      case p.Stmt.Var(_, Some(p.Expr.Alias(source: p.Term.Select)), _) => source
      case p.Stmt.Mut(_, p.Expr.Alias(source: p.Term.Select))          => source
    }
    val targets = stmts.collect {
      case p.Stmt.Mut(target, _)       => target
      case p.Stmt.Update(target, _, _) => target
    }
    val updated = stmts.collect {
      case p.Stmt.Update(p.Term.Select(root, path, _), _, _)
          if !globalPointerLeaves(root.tpe, members).contains(path) =>
        root
    }.toSet
    def aggregateOf(root: p.Named, steps: List[p.PathStep]) = globalPointerLeaves(root.tpe, members).exists { leaf =>
      leaf.startsWith(steps) && leaf.length > steps.length
    }
    val escaped = entry
      .collectAll[p.Term]
      .collect { case select @ p.Term.Select(root, steps, _) if aggregateOf(root, steps) => select }
      .groupMapReduce(identity)(_ => 1)(_ + _)
      .collect {
        case (select, uses) if uses > copies.count(_ == select) + targets.count(_ == select) => select.root
      }
      .toSet
    locals
      .filterNot(name => arguments(name) || addressed(name) || escaped(name) || updated(name) || capture.contains(name))
      .flatMap(name => globalPointerLeaves(name.tpe, members).map(name -> _))
  }

  // A pointer local chosen inside one conditional and then only loaded (std::min/max by reference) has no single
  // encoding when its candidates live in different storage, so each branch loads its own candidate instead.
  private def sinkSelectedLoads(entry: p.Function): p.Function = {
    def zero(term: p.Term) = term match {
      case p.Term.IntS64Const(0) | p.Term.IntS32Const(0) | p.Term.IntU64Const(0) | p.Term.IntU32Const(0) => true
      case _                                                                                             => false
    }
    def pure(stmt: p.Stmt) = core(stmt) match {
      case p.Stmt.Var(_, None, _)                                                                        => true
      case p.Stmt.Var(_, Some(_: p.Expr.Alias | _: p.Expr.IntrOp | _: p.Expr.Cast | _: p.Expr.RefTo), _) => true
      case p.Stmt.Var(_, Some(_: p.Expr.Index), _)                                                       => true
      case _                                                                                             => false
    }
    def mentions(stmt: p.Stmt, name: p.Named) =
      stmt.collectAll[p.Term].exists {
        case p.Term.Select(root, _, _) => root == name
        case _                         => false
      }
    val selects = entry
      .collectAll[p.Term]
      .collect { case p.Term.Select(root, _, _) => root }
      .groupMapReduce(identity)(_ => 1)(_ + _)
    val fresh      = AtomicLong()
    val reassigned = AddressRefinement.reassignedIn(entry)

    // assignments end their branch apart from pure statements, so loading there reads what the original load would
    def branchAssigns(branch: List[p.Stmt], pointer: p.Named): Option[List[p.Expr]] = {
      val assigned = branch.zipWithIndex.collect {
        case (stmt, i) if mentions(stmt, pointer) =>
          core(stmt) match {
            case _ if !branch.drop(i + 1).forall(pure) => None
            case p.Stmt.Mut(
                  p.Term.Select(`pointer`, Nil, _),
                  source @ (_: p.Expr.Alias | _: p.Expr.RefTo | _: p.Expr.Cast)
                ) =>
              Some(List(source))
            case p.Stmt.Cond(_, t, f) => branchAssigns(t, pointer).zip(branchAssigns(f, pointer)).map(_ ::: _)
            case _                    => None
          }
      }
      if (assigned.forall(_.nonEmpty)) Some(assigned.flatten.flatten) else None
    }
    def rewriteBranch(branch: List[p.Stmt], pointer: p.Named, value: p.Named, tpe: p.Type): List[p.Stmt] =
      branch.flatMap { stmt =>
        core(stmt) match {
          case p.Stmt.Mut(p.Term.Select(`pointer`, Nil, _), expr) =>
            val loaded = p.Named(s"#selected_load_${fresh.getAndIncrement()}", tpe)
            val (address, source) = expr match {
              case p.Expr.Alias(source) => Nil -> source
              case other =>
                val address = p.Named(s"#selected_address_${fresh.getAndIncrement()}", pointer.tpe)
                List(p.Stmt.Var(address, Some(other), false)) -> p.Term.Select(address, Nil, pointer.tpe)
            }
            address ::: List(
              p.Stmt.Var(loaded, Some(p.Expr.Index(source, p.Term.IntS64Const(0), tpe)), false),
              p.Stmt.Mut(
                p.Term.Select(value, Nil, tpe).asInstanceOf[p.Term.Select],
                p.Expr.Alias(p.Term.Select(loaded, Nil, tpe))
              )
            )
          case p.Stmt.Cond(c, t, f) if mentions(stmt, pointer) =>
            List(p.Stmt.Cond(c, rewriteBranch(t, pointer, value, tpe), rewriteBranch(f, pointer, value, tpe)))
          case _ => List(stmt)
        }
      }
    def block(stmts: List[p.Stmt]): List[p.Stmt] = {
      val sunk = stmts.zipWithIndex.iterator
        .collect { case (p.Stmt.Var(pointer @ p.Named(_, p.Type.Ptr(tpe, _), _), None, true), i) =>
          val after           = stmts.drop(i + 1)
          val choiceAt        = after.indexWhere(mentions(_, pointer))
          val (between, rest) = after.splitAt(math.max(choiceAt, 0))
          (pointer, tpe, i, choiceAt, between, rest)
        }
        .flatMap { (pointer, tpe, i, choiceAt, between, rest) =>
          rest match {
            case (choice @ p.Stmt.Cond(_, t, f)) :: tail if choiceAt >= 0 && between.forall(pure) =>
              val aliases = tail.collect {
                case p.Stmt.Var(alias, Some(p.Expr.Alias(p.Term.Select(`pointer`, Nil, _))), _)
                    if !reassigned(alias.symbol) =>
                  alias
              }.toSet
              val roots = aliases + pointer
              def load(expr: p.Expr) = expr match {
                case p.Expr.Index(p.Term.Select(root, Nil, _), index, _) => roots(root) && zero(index)
                case _                                                   => false
              }
              val lastLoad = tail.lastIndexWhere(stmt => roots.exists(mentions(stmt, _)))
              val uses     = tail.take(lastLoad + 1)
              val shaped = uses.zipWithIndex.forall {
                case (p.Stmt.Var(alias, Some(p.Expr.Alias(p.Term.Select(`pointer`, Nil, _))), _), _) =>
                  aliases(alias)
                case (p.Stmt.Var(_, Some(expr), _), _) if load(expr) => true
                case (p.Stmt.Mut(p.Term.Select(target, Nil, _), expr), at) if load(expr) =>
                  at == lastLoad && !roots(target)
                case (stmt, _) => pure(stmt) && !roots.exists(mentions(stmt, _))
              }
              val counted = roots.toList.map(selects.getOrElse(_, 0)).sum ==
                (choice :: uses)
                  .map(stmt =>
                    stmt.collectAll[p.Term].count {
                      case p.Term.Select(root, _, _) => roots(root)
                      case _                         => false
                    }
                  )
                  .sum
              Option.when(
                lastLoad >= 0 && shaped && counted && branchAssigns(t, pointer).nonEmpty && branchAssigns(
                  f,
                  pointer
                ).nonEmpty
              ) {
                val value = p.Named(s"${pointer.symbol}#selected", tpe)
                val rewritten = uses.flatMap {
                  case p.Stmt.Var(alias, _, _) if aliases(alias) => Nil
                  case p.Stmt.Var(name, Some(expr), mutable) if load(expr) =>
                    List(p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.Select(value, Nil, tpe))), mutable))
                  case p.Stmt.Mut(target, expr) if load(expr) =>
                    List(p.Stmt.Mut(target, p.Expr.Alias(p.Term.Select(value, Nil, tpe))))
                  case stmt => List(stmt)
                }
                stmts.take(i) ::: p.Stmt.Var(value, None, isMutable = true) :: between :::
                  rewriteBranch(List(choice), pointer, value, tpe) ::: rewritten ::: tail.drop(lastLoad + 1)
              }
            case _ => None
          }
        }
      sunk.nextOption().getOrElse(stmts)
    }
    entry.copy(body =
      doUntilNotEq(entry.body)((_, body) =>
        block(body).modifyAll[p.Stmt] {
          case p.Stmt.Cond(c, t, e)              => p.Stmt.Cond(c, block(t), block(e))
          case p.Stmt.While(c, b)                => p.Stmt.While(c, block(b))
          case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, block(b))
          case stmt                              => stmt
        }
      )._2
    )
  }

  // A native global pointer stored into a local aggregate is unrepresentable in logical SPIR-V, but a pointer leaf whose
  // every definition is the same never-reassigned argument (directly, or through struct copies of such leaves) always
  // holds that argument: its reads become the argument and its stores are dead. KernelCaptureFlatten's patched capture
  // copies are the common source; copies taken from the capture itself are always patched, so they define nothing.
  private def forwardArgumentSlots(entry: p.Function, members: Map[p.Sym, List[p.Named]]): p.Function = {
    type Key = (p.Named, List[p.PathStep])
    enum Value {
      case Unknown
      case Argument(argument: p.Named)
      case Reference(root: p.Named, path: List[p.PathStep])
      case Conflict
    }
    def meet(lhs: Value, rhs: Value): Value = (lhs, rhs) match {
      case (Value.Unknown, other)   => other
      case (other, Value.Unknown)   => other
      case (lhs, rhs) if lhs == rhs => lhs
      case _                        => Value.Conflict
    }
    val arguments  = entry.args.map(_.named).toSet
    val capture    = captureRoot(entry).map(_._1)
    val reassigned = AddressRefinement.reassignedIn(entry)
    val stmts      = entry.collectAll[p.Stmt].map(core)
    val inits = stmts.collect { case p.Stmt.Var(name, init, _) => name -> init }.groupMapReduce(_._1)(_._2)((a, _) => a)
    val candidates  = localPointerSlots(entry, members)
    val overwritten = overwrittenOnDefinition(entry, members)
    def slot(root: p.Named, path: List[p.PathStep]): Option[Either[Value, Key]] =
      if (path.isEmpty && arguments(root) && !reassigned(root.symbol)) Some(Left(Value.Argument(root)))
      else Option.when(candidates(root -> path))(Right(root -> path))
    // each definition is an argument, the address of a local subobject, another candidate leaf, or a conflict (None)
    val definitions: Map[Key, List[Option[Either[Value, Key]]]] = candidates.iterator.map { case key @ (root, leaf) =>
      val fromInit = inits.get(root).flatten.toList.flatMap {
        case _ if overwritten(key)                                                 => Nil
        case p.Expr.Alias(p.Term.Select(source, _, _)) if capture.contains(source) => Nil
        case p.Expr.Alias(p.Term.Select(source, steps, _))                         => List(slot(source, steps ++ leaf))
        case p.Expr.RefTo(p.Term.Select(target, path, _), None, comp, _, _)
            if leaf.isEmpty && !reassigned(root.symbol) && !arguments(target) &&
              typeAt(target.tpe, path, members).contains(comp) =>
          List(Some(Left(Value.Reference(target, path))))
        case _ => List(None)
      }
      val fromStores = stmts.collect {
        case p.Stmt.Mut(p.Term.Select(`root`, prefix, _), expr) if leaf.startsWith(prefix) =>
          expr match {
            case p.Expr.Alias(p.Term.Select(source, _, _)) if capture.contains(source) => None
            case p.Expr.Alias(p.Term.Select(source, steps, _)) => slot(source, steps ++ leaf.drop(prefix.length))
            case _                                             => None
          }
      }
      key -> (fromInit ::: fromStores)
    }.toMap
    // a slot holds a reference only as a pointer to exactly the referenced subobject's type
    def admits(key: Key, value: Value) = value match {
      case Value.Reference(target, path) =>
        typeAt(key._1.tpe, key._2, members).map(pointee).exists(typeAt(target.tpe, path, members).contains)
      case _ => true
    }
    val solved = doUntilNotEq(candidates.iterator.map(_ -> (Value.Unknown: Value)).toMap) { (_, known) =>
      known.map { case (key, _) =>
        key -> definitions(key).foldLeft(Value.Unknown: Value) {
          case (acc, Some(Left(value))) => meet(acc, if (admits(key, value)) value else Value.Conflict)
          case (acc, Some(Right(other))) =>
            val value = known(other)
            meet(acc, if (admits(key, value)) value else Value.Conflict)
          case (_, None) => Value.Conflict
        }
      }
    }._2
    // a leaf whose value also reaches an aggregate slot that is not forwarded must keep its store; a plain pointer
    // local legitimately receives the argument once its source read is rewritten
    val flows: List[(Key, Key)] = stmts.flatMap {
      case p.Stmt.Var(target, Some(p.Expr.Alias(p.Term.Select(source, steps, _))), _) =>
        globalPointerLeaves(target.tpe, members).map(leaf => (target -> leaf) -> (source -> (steps ++ leaf)))
      case p.Stmt.Mut(p.Term.Select(target, prefix, tpe), p.Expr.Alias(p.Term.Select(source, steps, _))) =>
        globalPointerLeaves(tpe, members).map(leaf => (target -> (prefix ++ leaf)) -> (source -> (steps ++ leaf)))
      case _ => Nil
    }
    // a reference is forwarded only where every read goes through it to a field; reading the address itself needs it
    val transfers = stmts.flatMap {
      case p.Stmt.Mut(target, p.Expr.Alias(source: p.Term.Select))     => List(target, source)
      case p.Stmt.Mut(target, _)                                       => List(target)
      case p.Stmt.Var(_, Some(p.Expr.Alias(source: p.Term.Select)), _) => List(source)
      case _                                                           => Nil
    }.toSet
    val leavesByRoot = candidates.toList.groupMap(_._1)(_._2)
    val addressReads = entry
      .collectAll[p.Term]
      .collect { case select @ p.Term.Select(root, path, _) =>
        leavesByRoot
          .getOrElse(root, Nil)
          .collect {
            case leaf if path.startsWith(leaf) =>
              val key = root -> leaf
              path.drop(leaf.length) match {
                case Nil if !transfers(select)                           => Some(key)
                case (_: p.PathStep.Index | _: p.PathStep.IndexDyn) :: _ => Some(key)
                case _                                                   => None
              }
          }
          .flatten
      }
      .flatten
      .toSet
    val forwarded = doUntilNotEq(solved.filter {
      case (_, _: Value.Argument)    => true
      case (key, _: Value.Reference) => !addressReads(key)
      case _                         => false
    }) { (_, known) =>
      val demoted = flows.collect {
        case (target @ (_, path), source) if path.nonEmpty && known.contains(source) && !known.contains(target) =>
          source
      }.toSet
      // a copied reference whose source keeps its address must keep its own
      val stranded = flows.collect {
        case (target, source)
            if known.get(target).exists(_.isInstanceOf[Value.Reference]) && !known.contains(source) &&
              candidates(source) =>
          target
      }.toSet
      known -- demoted -- stranded
    }._2
    if (forwarded.isEmpty) entry
    else {
      val byRoot = forwarded.groupMap(_._1._1) { case ((_, leaf), value) => leaf -> value }
      def rewrite(term: p.Term): p.Term = term match {
        case p.Term.Select(root, steps, tpe) =>
          byRoot
            .get(root)
            .flatMap(_.collectFirst {
              case (leaf, Value.Argument(argument)) if steps.startsWith(leaf) =>
                p.Term.Select(argument, steps.drop(leaf.length), tpe)
              case (leaf, Value.Reference(target, path)) if steps.startsWith(leaf) && steps.length > leaf.length =>
                val through = steps.drop(leaf.length) match {
                  case p.PathStep.Deref :: rest => rest
                  case rest                     => rest
                }
                p.Term.Select(target, path ++ through, tpe)
            })
            .getOrElse(term)
        case other => other
      }
      def isStore(stmt: p.Stmt): Boolean = core(stmt) match {
        case p.Stmt.Mut(p.Term.Select(root, steps, _), _) => forwarded.contains(root -> steps)
        case _                                            => false
      }
      def strip(stmts: List[p.Stmt]): List[p.Stmt] = stmts.filterNot(isStore)
      val body = strip(entry.body)
        .modifyAll[p.Stmt] {
          case p.Stmt.Cond(c, t, e)              => p.Stmt.Cond(c, strip(t), strip(e))
          case p.Stmt.While(c, b)                => p.Stmt.While(c, strip(b))
          case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, strip(b))
          case p.Stmt.Try(b, handlers, fin)      => p.Stmt.Try(strip(b), handlers, strip(fin))
          case stmt                              => stmt
        }
        .modifyAll[p.Term](rewrite)
        .modifyAll[p.Stmt] {
          case p.Stmt.Mut(target, expr) => p.Stmt.Mut(rewrite(target).asInstanceOf[p.Term.Select], expr)
          case p.Stmt.Update(target, index, value) =>
            p.Stmt.Update(rewrite(target).asInstanceOf[p.Term.Select], index, value)
          case stmt => stmt
        }
      // the logical backend resolves a buffer binding only from the argument itself, so the forwarded reads' aliases
      // fold into it
      val sources = forwarded.values.collect { case Value.Argument(argument) => argument }.toSet
      val aliases = doUntilNotEq(Map.empty[p.Named, p.Named]) { (_, known) =>
        known ++ body
          .flatMap(_.collectAll[p.Stmt])
          .collect {
            case p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), false)
                if !reassigned(name.symbol) && (sources(root) || known.contains(root)) =>
              name -> known.getOrElse(root, root)
          }
          .toMap
      }._2
      def fold(term: p.Term): p.Term = term match {
        case p.Term.Select(root, steps, tpe) if aliases.contains(root) => p.Term.Select(aliases(root), steps, tpe)
        case other                                                     => other
      }
      def drop(stmts: List[p.Stmt]): List[p.Stmt] = stmts.filter(stmt =>
        core(stmt) match {
          case p.Stmt.Var(name, _, _) => !aliases.contains(name)
          case _                      => true
        }
      )
      val folded = drop(body)
        .modifyAll[p.Stmt] {
          case p.Stmt.Mut(target, expr) => p.Stmt.Mut(fold(target).asInstanceOf[p.Term.Select], expr)
          case p.Stmt.Update(target, index, value) =>
            p.Stmt.Update(fold(target).asInstanceOf[p.Term.Select], index, value)
          case p.Stmt.Cond(c, t, e)              => p.Stmt.Cond(c, drop(t), drop(e))
          case p.Stmt.While(c, b)                => p.Stmt.While(c, drop(b))
          case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, drop(b))
          case p.Stmt.Try(b, handlers, fin)      => p.Stmt.Try(drop(b), handlers, drop(fin))
          case stmt                              => stmt
        }
        .modifyAll[p.Term](fold)
      entry.copy(body = folded)
    }
  }

  // A native global pointer slot whose every definition derives from the same argument holds an element offset from it.
  // The offset lives in a companion local, and reads rebuild the pointer from the argument, the only form a logical
  // binding is addressed through. Pointer values on the way to such a slot carry their offset in a companion as well.
  private def bindSlotOffsets(entry: p.Function, members: Map[p.Sym, List[p.Named]]): p.Function = {
    type Key = (p.Named, List[p.PathStep])
    enum Value {
      case Unknown
      case Argument(argument: p.Named)
      case Conflict
    }
    def meet(lhs: Value, rhs: Value): Value = (lhs, rhs) match {
      case (Value.Unknown, other)                           => other
      case (other, Value.Unknown)                           => other
      case (Value.Argument(a), Value.Argument(b)) if a == b => lhs
      case _                                                => Value.Conflict
    }
    def globalPointee(tpe: p.Type): Option[p.Type] = tpe match {
      case p.Type.Ptr(component, p.Type.Space.Global) => Some(component)
      case _                                          => None
    }
    val capture   = captureRoot(entry).map(_._1)
    val arguments = entry.args.map(_.named).filterNot(capture.contains).filter(a => globalPointee(a.tpe).nonEmpty).toSet
    val reassigned = AddressRefinement.reassignedIn(entry)
    val stmts      = entry.collectAll[p.Stmt].map(core)
    val inits      = stmts.collect { case p.Stmt.Var(name, init, _) => name -> init }.toMap
    val whileReads = stmts.collect { case p.Stmt.While(p.Term.Select(root, _, _), _) => root }.toSet
    val slots = localPointerSlots(entry, members).filter { (root, leaf) =>
      !whileReads(root) && (leaf.nonEmpty || reassigned(root.symbol))
    }
    val overwritten = overwrittenOnDefinition(entry, members)
    def leafType(root: p.Named, leaf: List[p.PathStep]) =
      if (leaf.isEmpty) root.tpe else fieldsAt(root.tpe, leaf, members).last._2
    val values = stmts.collect {
      case p.Stmt.Var(name, Some(expr), _) if !reassigned(name.symbol) && globalPointee(name.tpe).nonEmpty =>
        name -> expr
    }.toMap

    // the pointer a term or expression is derived from without changing its element type
    def sourceOf(expr: p.Expr): Option[Either[p.Named, Key]] = expr match {
      case p.Expr.Alias(p.Term.Select(root, Nil, _)) if arguments(root) && !reassigned(root.symbol) => Some(Left(root))
      case p.Expr.Alias(p.Term.Select(root, path, _)) if slots(root -> path)  => Some(Right(root -> path))
      case p.Expr.Alias(p.Term.Select(root, Nil, _)) if values.contains(root) => sourceOf(values(root))
      case p.Expr.Cast(from @ p.Term.Select(_, Nil, p.Type.Ptr(c, _)), p.Type.Ptr(to, _)) if c == to =>
        sourceOf(p.Expr.Alias(from))
      case p.Expr.RefTo(from @ p.Term.Select(_, Nil, p.Type.Ptr(c, p.Type.Space.Global)), _, comp, _, _) if c == comp =>
        sourceOf(p.Expr.Alias(from))
      case _ => None
    }
    def chain(expr: p.Expr): List[p.Named] = expr match {
      case p.Expr.Alias(p.Term.Select(root, Nil, _)) if values.contains(root) && !arguments(root) =>
        root :: chain(values(root))
      case p.Expr.Cast(from: p.Term.Select, _)           => chain(p.Expr.Alias(from))
      case p.Expr.RefTo(from: p.Term.Select, _, _, _, _) => chain(p.Expr.Alias(from))
      case _                                             => Nil
    }
    // definitions of a slot: exact stores, and the matching leaf of whole-aggregate copies
    val definitions: Map[Key, List[Option[p.Expr]]] = slots.iterator.map { case key @ (root, leaf) =>
      val leafTpe = leafType(root, leaf)
      def copied(source: p.Named, steps: List[p.PathStep], rest: List[p.PathStep]) =
        if (capture.contains(source)) Nil else List(Some(p.Expr.Alias(p.Term.Select(source, steps ++ rest, leafTpe))))
      val fromInit = inits.get(root).flatten.toList.flatMap {
        case _ if overwritten(key)                         => Nil
        case init if leaf.isEmpty                          => List(Some(init))
        case p.Expr.Alias(p.Term.Select(source, steps, _)) => copied(source, steps, leaf)
        case _                                             => List(None)
      }
      val fromStores = stmts.flatMap {
        case p.Stmt.Mut(p.Term.Select(`root`, `leaf`, _), expr) => List(Some(expr))
        case p.Stmt.Mut(p.Term.Select(`root`, prefix, _), expr) if leaf.startsWith(prefix) =>
          expr match {
            case p.Expr.Alias(p.Term.Select(source, steps, _)) => copied(source, steps, leaf.drop(prefix.length))
            case _                                             => List(None)
          }
        case _ => Nil
      }
      key -> (fromInit ::: fromStores)
    }.toMap
    val solved = doUntilNotEq(slots.iterator.map(_ -> (Value.Unknown: Value)).toMap) { (_, known) =>
      known.map { case (key @ (root, leaf), _) =>
        val leafPointee = globalPointee(leafType(root, leaf))
        key -> definitions(key).foldLeft(Value.Unknown: Value) {
          case (acc, Some(expr)) =>
            sourceOf(expr) match {
              case Some(Left(argument)) if globalPointee(argument.tpe) == leafPointee =>
                meet(acc, Value.Argument(argument))
              case Some(Right(other)) => meet(acc, known(other))
              case _                  => Value.Conflict
            }
          case (_, None) => Value.Conflict
        }
      }
    }._2
    // an aggregate copy carries the stored pointer itself, so its source stays a pointer unless the target is bound too
    val flows: List[(Key, Key)] = stmts.flatMap {
      case p.Stmt.Var(target, Some(p.Expr.Alias(p.Term.Select(source, steps, _))), _) =>
        globalPointerLeaves(target.tpe, members)
          .filter(_.nonEmpty)
          .map(leaf => (target -> leaf) -> (source -> (steps ++ leaf)))
      case p.Stmt.Mut(p.Term.Select(target, prefix, tpe), p.Expr.Alias(p.Term.Select(source, steps, _))) =>
        globalPointerLeaves(tpe, members)
          .filter(_.nonEmpty)
          .map(leaf => (target -> (prefix ++ leaf)) -> (source -> (steps ++ leaf)))
      case _ => Nil
    }
    val dependencies = definitions.view.mapValues(_.flatten.flatMap(sourceOf(_).flatMap(_.toOption))).toMap
    val bound = doUntilNotEq(solved.collect { case (key, Value.Argument(argument)) => key -> argument }) { (_, known) =>
      known --
        flows.collect { case (target, source) if known.contains(source) && !known.contains(target) => source } --
        known.keys.filter(key => dependencies(key).exists(!known.contains(_)))
    }._2
    if (bound.isEmpty) entry
    else {
      val fresh             = AtomicLong()
      def temp(tpe: p.Type) = p.Named(s"#slot_offset_${fresh.getAndIncrement()}", tpe)
      val companions        = bound.keys.map((root, leaf) => (root, leaf) -> temp(p.Type.IntS64)).toMap
      val byRoot            = bound.keys.toList.groupMap(_._1)(_._2)
      val reads = stmts.collect {
        case p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.Select(root, path, _))), _)
            if !reassigned(name.symbol) && bound.contains(root -> path) =>
          name -> (root -> path)
      }.toMap
      val derived = bound.keys.iterator
        .flatMap(definitions(_).flatten)
        .flatMap(chain)
        .filterNot(reads.contains)
        .toSet
      val offsets = (derived ++ reads.keySet).iterator.map(name => name -> temp(p.Type.IntS64)).toMap

      def offsetOf(expr: p.Expr): (List[p.Stmt], p.Term) = expr match {
        case p.Expr.Alias(p.Term.Select(root, Nil, _)) if arguments(root) => Nil -> p.Term.IntS64Const(0)
        case p.Expr.Alias(p.Term.Select(root, Nil, _)) if offsets.contains(root) =>
          Nil -> p.Term.Select(offsets(root), Nil, p.Type.IntS64)
        case p.Expr.Alias(p.Term.Select(root, path, _)) if companions.contains(root -> path) =>
          Nil -> p.Term.Select(companions(root -> path), Nil, p.Type.IntS64)
        case p.Expr.Cast(from: p.Term.Select, _)              => offsetOf(p.Expr.Alias(from))
        case p.Expr.RefTo(from: p.Term.Select, None, _, _, _) => offsetOf(p.Expr.Alias(from))
        case p.Expr.RefTo(from: p.Term.Select, Some(index), _, _, _) =>
          val (prefix, base) = offsetOf(p.Expr.Alias(from))
          val (widen, step) =
            if (index.tpe == p.Type.IntS64) Nil -> index
            else {
              val widened = temp(p.Type.IntS64)
              List(p.Stmt.Var(widened, Some(p.Expr.Cast(index, p.Type.IntS64)))) -> p.Term.Select(
                widened,
                Nil,
                p.Type.IntS64
              )
            }
          val sum = temp(p.Type.IntS64)
          (prefix ::: widen ::: List(p.Stmt.Var(sum, Some(p.Expr.IntrOp(p.Intr.Add(base, step, p.Type.IntS64)))))) ->
            p.Term.Select(sum, Nil, p.Type.IntS64)
        case other => throw IllegalStateException(s"bound slot source ${other.repr} has no offset")
      }
      def rebuild(argument: p.Named, offset: p.Term) =
        p.Expr.RefTo(
          p.Term.Select(argument, Nil, argument.tpe),
          Some(offset),
          globalPointee(argument.tpe).get,
          p.Type.Space.Global,
          p.Region.Opaque
        )
      def boundLeaf(root: p.Named, path: List[p.PathStep]) =
        byRoot.get(root).flatMap(_.find(leaf => path.startsWith(leaf)))
      // reads of a bound slot inside a statement's own terms become a rebuilt pointer defined just before it
      def hoist(stmt: p.Stmt): (List[p.Stmt], p.Stmt) = {
        val hoisted = ListBuffer.empty[p.Stmt]
        def term(t: p.Term): p.Term = t match {
          case p.Term.Select(root, path, tpe) if boundLeaf(root, path).nonEmpty =>
            val leaf    = boundLeaf(root, path).get
            val pointer = temp(leafType(root, leaf))
            hoisted += p.Stmt.Var(
              pointer,
              Some(rebuild(bound(root -> leaf), p.Term.Select(companions(root -> leaf), Nil, p.Type.IntS64)))
            )
            p.Term.Select(pointer, path.drop(leaf.length), tpe)
          case other => other
        }
        def select(t: p.Term.Select) = term(t).asInstanceOf[p.Term.Select]
        def expr(e: p.Expr)          = e.modifyAll[p.Term](term)
        val rewritten = stmt match {
          case p.Stmt.Var(name, init, mutable)   => p.Stmt.Var(name, init.map(expr), mutable)
          case p.Stmt.Mut(target, e)             => p.Stmt.Mut(select(target), expr(e))
          case p.Stmt.Update(target, index, v)   => p.Stmt.Update(select(target), term(index), term(v))
          case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, term(lb), term(ub), term(st), b)
          case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(term(c), t, f)
          case p.Stmt.Return(e)                  => p.Stmt.Return(expr(e))
          case p.Stmt.Raise(v, kind, cleanup)    => p.Stmt.Raise(term(v), kind, cleanup)
          case other                             => other
        }
        hoisted.toList -> rewritten
      }
      def copyOffsets(root: p.Named, prefix: List[p.PathStep], source: p.Named, steps: List[p.PathStep]) =
        byRoot.getOrElse(root, Nil).filter(_.startsWith(prefix)).flatMap { leaf =>
          companions.get(source -> (steps ++ leaf.drop(prefix.length))).map { from =>
            p.Stmt.Mut(
              p.Term.Select(companions(root -> leaf), Nil, p.Type.IntS64).asInstanceOf[p.Term.Select],
              p.Expr.Alias(p.Term.Select(from, Nil, p.Type.IntS64))
            )
          }
        }
      def statement(stmt: p.Stmt): List[p.Stmt] = stmt match {
        case p.Stmt.Annotated(inner, pos, comment) =>
          statement(inner) match {
            case List(single) => List(p.Stmt.Annotated(single, pos, comment))
            case many         => many
          }
        case p.Stmt.Var(name, _, mutable) if reads.contains(name) =>
          val key = reads(name)
          List(
            p.Stmt.Var(offsets(name), Some(p.Expr.Alias(p.Term.Select(companions(key), Nil, p.Type.IntS64)))),
            p.Stmt.Var(name, Some(rebuild(bound(key), p.Term.Select(offsets(name), Nil, p.Type.IntS64))), mutable)
          )
        case p.Stmt.Mut(p.Term.Select(root, path, _), value) if companions.contains(root -> path) =>
          val (prefix, offset) = offsetOf(value)
          prefix :+ p.Stmt.Mut(
            p.Term.Select(companions(root -> path), Nil, p.Type.IntS64).asInstanceOf[p.Term.Select],
            p.Expr.Alias(offset)
          )
        case _ =>
          val nested = stmt match {
            case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, block(t), block(f))
            case p.Stmt.While(c, b)                => p.Stmt.While(c, block(b))
            case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, block(b))
            case p.Stmt.Try(b, handlers, fin) =>
              p.Stmt.Try(block(b), handlers.map(h => h.copy(body = block(h.body))), block(fin))
            case p.Stmt.Raise(v, kind, cleanup) => p.Stmt.Raise(v, kind, block(cleanup))
            case other                          => other
          }
          val (before, rewritten) = hoist(nested)
          val after = nested match {
            case p.Stmt.Var(name, init, _) if companions.contains(name -> Nil) =>
              init.fold(List(p.Stmt.Var(companions(name -> Nil), None, isMutable = true))) { value =>
                val (prefix, offset) = offsetOf(value)
                prefix :+ p.Stmt.Var(companions(name -> Nil), Some(p.Expr.Alias(offset)), isMutable = true)
              }
            case p.Stmt.Var(name, init, _) =>
              val declared =
                byRoot.getOrElse(name, Nil).map(leaf => p.Stmt.Var(companions(name -> leaf), None, isMutable = true))
              val copied = init.toList.flatMap {
                case p.Expr.Alias(p.Term.Select(source, steps, _)) => copyOffsets(name, Nil, source, steps)
                case _                                             => Nil
              }
              val offset = offsets.get(name).filter(_ => derived(name)).toList.flatMap { companion =>
                val (prefix, value) = offsetOf(init.get)
                prefix :+ p.Stmt.Var(companion, Some(p.Expr.Alias(value)))
              }
              declared ::: copied ::: offset
            case p.Stmt.Mut(p.Term.Select(root, prefix, _), p.Expr.Alias(p.Term.Select(source, steps, _))) =>
              copyOffsets(root, prefix, source, steps)
            case _ => Nil
          }
          before ::: rewritten :: after
      }
      def block(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap(statement)
      entry.copy(body = block(entry.body))
    }
  }

  override def apply(input: p.Program, log: Log): p.Program = {
    // ORIGINAL member types drive the offset walk (each pointer field's pointee struct); retyping preserves
    // the layout, so emitted OffsetOf resolves the same against the retyped def
    val members = input.defs.iterator.map(d => d.name -> d.members).toMap
    val prepared = bindSlotOffsets(
      doUntilNotEq(input.entry.getOrElse(throw IllegalArgumentException("ArenaView requires a program entry"))) {
        (_, current) =>
          dropDeadValues(forwardArgumentSlots(resolveSelfPointers(sinkSelectedLoads(current), members), members))
      }._2,
      members
    )
    def plan(entry: p.Function) = {
      val program  = input.copy(entry = Some(entry))
      val analysis = AddressRefinement.solve(program, entry, AddressRefinement.AddressModel.Logical).requireSolved
      val plan     = AddressRefinement.logicalAddressPlan(program, analysis)
      (entry, program, analysis, plan, unsupportedNativeSlots(entry, members, plan.identityFields, analysis))
    }
    val (entry, program, analysis, logicalAddressPlan, unsupported) = plan(prepared) match {
      case planned @ (_, _, _, _, Nil) => planned
      case planned @ (entry, _, _, _, slots) =>
        val unused = writeOnlyLocals(entry, members)
        if (!slots.forall((root, _) => unused(root))) planned
        else
          plan(
            dropStatements(
              entry,
              stmt =>
                core(stmt) match {
                  case p.Stmt.Var(name, _, _)                   => unused(name)
                  case p.Stmt.Mut(p.Term.Select(root, _, _), _) => unused(root)
                  case _                                        => false
                }
            )
          )
    }
    if (unsupported.nonEmpty)
      throw IllegalArgumentException(
        s"logical SPIR-V cannot store native global pointers in local aggregate slots: ${unsupported.map(_._2).mkString(", ")}"
      )
    val arenaDefs      = logicalAddressPlan.arenaStructs
    val identityFields = logicalAddressPlan.identityFields
    val fieldAliases   = logicalAddressPlan.fieldAliases
    val locals         = entry.collectAll[p.Stmt].collect { case p.Stmt.Var(n, _, _) => n.symbol -> n }.toMap
    val offsetFields = analysis.slots.iterator.flatMap { case (AddressRefinement.Query.Slot(root, path), fact) =>
      val referencesOffset = fact.references.exists(token =>
        analysis.facts
          .get(token)
          .flatMap(_.encoding)
          .contains(AddressRefinement.Encoding.ArenaRelative)
      )
      Option
        .when(fact.encoding.contains(AddressRefinement.Encoding.ArenaRelative) || referencesOffset) {
          locals.get(root).flatMap(n => fieldAt(n.tpe, path, members))
        }
        .flatten
    }.toSet
    val encodedFields = identityFields ++ offsetFields ++ arenaDefs.flatMap { owner =>
      members.getOrElse(owner, Nil).collect {
        case member if member.tpe match {
              case p.Type.Ptr(_, p.Type.Space.Global) => true
              case _                                  => false
            } =>
          owner -> member.symbol
      }
    }
    // union: copy only the canonical (largest, head) member
    val unions = program.defs.iterator.filter(_.isUnion).map(_.name).toSet
    val retyped = program.defs.map(d =>
      d.copy(members = d.members.map(m => if (encodedFields(d.name -> m.symbol)) m.copy(tpe = i64ify(m.tpe)) else m))
    )
    program.copy(
      defs = retyped,
      entry = Some(
        run(
          members,
          unions,
          encodedFields,
          identityFields,
          fieldAliases,
          logicalAddressPlan.localPointerKeys,
          logicalAddressPlan.directLocalPointerKeys,
          analysis,
          entry
        )
      )
    )
  }

  // lift a stepped Select (the only term shape that can carry an arena access) out of a ForRange bound or
  // Cond condition into a preceding Var, so the leaf rewriter handles it; bare vars and constants stay
  private def hoistInlineTerms(stmts: List[p.Stmt]): List[p.Stmt] = {
    def lift(hint: String, t: p.Term): (List[p.Stmt], p.Term) = t match {
      case p.Term.Select(_, steps, _) if steps.nonEmpty =>
        val n = p.Named(s"#$hint${ctr.incrementAndGet()}", t.tpe);
        (List(p.Stmt.Var(n, Some(p.Expr.Alias(t)), isMutable = false)), sel(n))
      case _ => (Nil, t)
    }
    stmts.flatMap {
      case p.Stmt.ForRange(i, lb, ub, st, body) =>
        val (lbS, lbT) = lift("flb", lb); val (ubS, ubT) = lift("fub", ub); val (stS, stT) = lift("fst", st)
        lbS ::: ubS ::: stS ::: List(p.Stmt.ForRange(i, lbT, ubT, stT, hoistInlineTerms(body)))
      case p.Stmt.Cond(c, t, e) =>
        val (cS, cT) = lift("cnd", c); cS ::: List(p.Stmt.Cond(cT, hoistInlineTerms(t), hoistInlineTerms(e)))
      case p.Stmt.While(c, body)           => List(p.Stmt.While(c, hoistInlineTerms(body)))
      case p.Stmt.Annotated(inner, pos, k) => hoistInlineTerms(List(inner)).map(p.Stmt.Annotated(_, pos, k))
      case s                               => List(s)
    }
  }

  private def run(
      members: Map[p.Sym, List[p.Named]],
      unions: Set[p.Sym],
      encodedFields: Set[Field],
      identityFields: Set[Field],
      fieldAliases: Map[p.Named, (Field, p.Term)],
      localPointerKeys: Map[p.Named, String],
      directLocalPointerKeys: Set[String],
      analysis: AddressRefinement.Solution,
      f: p.Function
  ): p.Function = captureRoot(
    f
  ) match {
    case None => f
    case Some((capN, capTpe)) =>
      import AddressRefinement.{Provenance, Encoding}

      // The dispatch binds one arena buffer through this canonical typed-view ABI. Float16 needs its own view:
      // PolyAST Cast is numeric, so reading f16 bits through the i16 view would convert rather than reinterpret.
      // Unused views are pruned by the backend.
      val views = LogicalArenaViewAbi.bindings

      def rootedLocally(t: p.Term): Boolean = analysis.provenancesOf(t).exists {
        case _: Provenance.Local => true
        case _                   => false
      }
      val reassignedPointers = f
        .collectAll[p.Stmt]
        .collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _: p.Type.Ptr), _) =>
          n
        }
        .toSet
      val tokenByKey = (localPointerKeys.values.toSet ++ directLocalPointerKeys).toList.sorted.zipWithIndex.map {
        case (key, i) =>
          key -> (i.toLong + 1L)
      }.toMap
      val localPointerTokens = localPointerKeys.view.mapValues(tokenByKey).toMap

      // Inlined nullable base-pointer adjustments retain their source-level null guard after their actual argument
      // becomes either an immutable RefTo of stack storage or an immutable null binding. Keeping those guards
      // creates Function-pointer phis which logical SPIR-V cannot represent and some physical SPIR-V runtimes
      // miscompile for non-zero multiple-inheritance base offsets. Fold only guards with stable local proofs.
      def definitelyNonNullLocal(t: p.Term): Boolean = t match {
        case p.Term.Select(root, Nil, _: p.Type.Ptr) => !reassignedPointers(root) && rootedLocally(t)
        case _                                       => false
      }
      val stableNullPointers = f.collectAll[p.Stmt].foldLeft(Set.empty[p.Named]) {
        case (known, p.Stmt.Var(n, Some(p.Expr.Alias(_: p.Term.NullPtrConst)), _))
            if isPtr(n.tpe) && !reassignedPointers(n) =>
          known + n
        case (known, p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, Nil, _))), _))
            if isPtr(n.tpe) && !reassignedPointers(n) && known(root) =>
          known + n
        case (known, _) => known
      }
      def definitelyNull(t: p.Term): Boolean = t match {
        case _: p.Term.NullPtrConst      => true
        case p.Term.Select(root, Nil, _) => stableNullPointers(root)
        case _                           => false
      }
      val constantConditions = f
        .collectAll[p.Stmt]
        .collect {
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(x, _: p.Term.NullPtrConst))), false)
              if definitelyNonNullLocal(x) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(_: p.Term.NullPtrConst, y))), false)
              if definitelyNonNullLocal(y) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicEq(x, y))), false)
              if definitelyNull(x) && definitelyNull(y) =>
            n -> true
          case p.Stmt.Var(n, Some(p.Expr.IntrOp(p.Intr.LogicNeq(x, y))), false)
              if definitelyNull(x) && definitelyNull(y) =>
            n -> false
        }
        .toMap
      def simplifyStablePointerGuards(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
        case p.Stmt.Cond(p.Term.Select(root, Nil, _), whenTrue, whenFalse) if constantConditions.contains(root) =>
          simplifyStablePointerGuards(if (constantConditions(root)) whenTrue else whenFalse)
        case p.Stmt.Cond(c, whenTrue, whenFalse) =>
          List(p.Stmt.Cond(c, simplifyStablePointerGuards(whenTrue), simplifyStablePointerGuards(whenFalse)))
        case p.Stmt.While(c, body) => List(p.Stmt.While(c, simplifyStablePointerGuards(body)))
        case p.Stmt.ForRange(i, lb, ub, step, body) =>
          List(p.Stmt.ForRange(i, lb, ub, step, simplifyStablePointerGuards(body)))
        case t: p.Stmt.Try => List(t.mapBlocks(simplifyStablePointerGuards))
        case p.Stmt.Raise(value, exceptionKind, cleanup) =>
          List(p.Stmt.Raise(value, exceptionKind, simplifyStablePointerGuards(cleanup)))
        case p.Stmt.Annotated(inner, pos, comment) =>
          simplifyStablePointerGuards(List(inner)).map(p.Stmt.Annotated(_, pos, comment))
        case stmt => List(stmt)
      }

      def arenaFact(fact: AddressRefinement.AddressValue): Boolean =
        fact.hasArenaRoot || fact.encoding.contains(Encoding.ArenaRelative)

      def isArena(n: p.Named): Boolean =
        n == capN || analysis.bindings.get(n.symbol).exists(arenaFact)

      // ForRange bounds / Cond conditions hold terms inline (not in a visited leaf); hoist any stepped Select
      // into a preceding Var. bounds are loop-invariant so hoisting once is sound; While conds are plain vars
      val body =
        mapStmtsRec(hoistInlineTerms(simplifyStablePointerGuards(f.body)))(
          rewriteLeaf(
            members,
            unions,
            encodedFields,
            identityFields,
            fieldAliases,
            localPointerTokens,
            tokenByKey,
            capN,
            capTpe,
            views,
            analysis,
            isArena,
            f.collectAll[p.Stmt]
              .collect {
                case p.Stmt.Mut(p.Term.Select(root, _, _), _)       => root
                case p.Stmt.Update(p.Term.Select(root, _, _), _, _) => root
              }
              .toSet
          )
        )
      // neutralise view binding slots to an i8 view so the slot stays aligned, so we can avoid dragging unused types in
      val usedViews = body.flatMap(_.collectWhere[p.Term] { case p.Term.Select(r, _, _) => r.symbol }).toSet
      val pinnedViews =
        views.map(v => if (usedViews(v.symbol)) v else p.Named(v.symbol, p.Type.Ptr(p.Type.IntS8, Global)))
      // the views replace ONLY the capture, at its position (a receiver capture precedes the arguments), so a
      // dispatch binds the arena where it would have passed the capture
      // a kernel that never reads its capture keeps the (unread) capture argument, so the views are declared exactly
      // when a dispatch must bind them
      val viewArgs =
        if (usedViews.exists(symbol => views.exists(_.symbol == symbol))) pinnedViews.map(p.Arg(_)) else Nil
      val receiverCap = f.receiver.exists(_.named == capN) && viewArgs.nonEmpty
      val newArgs =
        if (viewArgs.isEmpty) f.args
        else if (receiverCap) viewArgs ++ f.args
        else f.args.flatMap(a => if (a.named == capN) viewArgs else List(a))
      val newReceiver = if (receiverCap) None else f.receiver
      f.copy(
        decl = f.decl
          .remapArgs(newArgs)
          .copy(
            receiver = newReceiver,
            moduleCaptures = Nil,
            termCaptures = Nil
          ),
        body = body
      )
  }

  private def rewriteLeaf(
      members: Map[p.Sym, List[p.Named]],
      unions: Set[p.Sym],
      encodedFields: Set[Field],
      identityFields: Set[Field],
      fieldAliases: Map[p.Named, (Field, p.Term)],
      localPointerTokens: Map[p.Named, Long],
      localReferenceTokens: Map[String, Long],
      capN: p.Named,
      capTpe: p.Type.Struct,
      views: List[p.Named],
      analysis: AddressRefinement.Solution,
      isArena: p.Named => Boolean,
      mutatedRoots: Set[p.Named]
  )(leaf: p.Stmt): List[p.Stmt] = {
    val pre = ListBuffer.empty[p.Stmt]

    def fresh(hint: String, t: p.Type): p.Named = p.Named(s"#$hint${ctr.incrementAndGet()}", t)
    def bind(hint: String, e: p.Expr): p.Term = e match {
      case p.Expr.Alias(t) => t
      case other => val n = fresh(hint, other.tpe); pre += p.Stmt.Var(n, Some(other), isMutable = false); sel(n)
    }
    def i64(v: Long): p.Term     = p.Term.IntS64Const(v)
    def asI64(t: p.Term): p.Term = if (t.tpe == I64) t else bind("ai", p.Expr.Cast(t, I64))
    def add(a: p.Term, b: p.Term): p.Term =
      if (b == i64(0)) a else bind("ao", p.Expr.IntrOp(p.Intr.Add(a, asI64(b), I64)))

    def memberTpe(sym: p.Sym, field: String): p.Type =
      members.get(sym).flatMap(_.find(_.symbol == field).map(_.tpe)).getOrElse(I64)
    def isIdentityField(root: p.Named, steps: List[p.PathStep]): Boolean =
      fieldAt(root.tpe, steps, members).exists(identityFields)
    def isEncodedField(root: p.Named, steps: List[p.PathStep]): Boolean =
      fieldAt(root.tpe, steps, members).exists(encodedFields)
    // union: copy/read just the canonical (largest, head) member
    def canonicalMembers(sym: p.Sym): List[p.Named] = {
      val ms = members.getOrElse(sym, Nil); if (unions.contains(sym)) ms.take(1) else ms
    }
    def structSym(t: p.Type): Option[p.Sym] = t match { case p.Type.Struct(s, _) => Some(s); case _ => None }
    def arenaTerm(t: p.Term): Boolean = {
      val represented = analysis
        .value(t)
      val representedAsArena =
        represented.hasArenaRoot || represented.encoding.contains(AddressRefinement.Encoding.ArenaRelative)
      representedAsArena || analysis.provenancesOf(t).exists {
        case AddressRefinement.Provenance.ArenaRoot(_) => true
        case _                                         => false
      }
    }

    def viewFor(t: p.Type): (p.Named, p.Type, Int) = t match {
      case _: p.Type.Ptr                              => (views(3), p.Type.IntS64, 3)
      case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8 => (views(0), p.Type.IntS8, 0)
      case p.Type.IntU16 | p.Type.IntS16              => (views(1), p.Type.IntS16, 1)
      case p.Type.IntU32 | p.Type.IntS32              => (views(2), p.Type.IntS32, 2)
      case p.Type.Float32                             => (views(4), p.Type.Float32, 2)
      case p.Type.IntU64 | p.Type.IntS64              => (views(3), p.Type.IntS64, 3)
      case p.Type.Float64                             => (views(5), p.Type.Float64, 3)
      case p.Type.Float16                             => (views(6), p.Type.Float16, 1)
      case _                                          => (views(3), p.Type.IntS64, 3)
    }
    def indexOf(off: p.Term, sh: Int): p.Term =
      if (sh == 0) off else bind("ix", p.Expr.IntrOp(p.Intr.BSR(off, i64(sh.toLong), I64)))
    def isAgg(t: p.Type): Boolean = t match {
      case _: p.Type.Struct => true; case _: p.Type.Arr => true; case _ => false
    }
    def loadAt(off: p.Term, t: p.Type): p.Term =
      if (isAgg(t)) loadAgg(off, t)
      else {
        val (v, comp, sh) = viewFor(t)
        val raw           = bind("ld", p.Expr.Index(sel(v), indexOf(off, sh), comp))
        if (t == comp || isPtr(t)) raw else bind("lc", p.Expr.Cast(raw, t))
      }
    // a struct/array read by value cannot go through a scalar view; materialise a local copy, filling each
    // scalar leaf from the arena (pointer fields are i64 offsets in the retyped def, so they copy as i64)
    def loadAgg(off: p.Term, t: p.Type): p.Term = {
      val sv = fresh("sv", t); pre += p.Stmt.Var(sv, None, isMutable = true)
      def fill(prefix: List[p.PathStep], o: p.Term, ft: p.Type): Unit = ft match {
        case s: p.Type.Struct =>
          canonicalMembers(s.name).foreach { m =>
            fill(
              prefix :+ p.PathStep.Field(m.symbol),
              add(o, asI64(bind("of", p.Expr.OffsetOf(ft, m.symbol)))),
              i64ify(m.tpe)
            )
          }
        case p.Type.Arr(elem, n, _) =>
          (0 until n).foreach(e =>
            fill(prefix :+ p.PathStep.Index(e), add(o, mulBytes(i64(e.toLong), elem)), i64ify(elem))
          )
        case scalar => pre += p.Stmt.Mut(p.Term.Select(sv, prefix, scalar), p.Expr.Alias(loadAt(o, scalar)))
      }
      fill(Nil, off, t)
      sel(sv)
    }
    def storeAt(off: p.Term, t: p.Type, value: p.Term): p.Stmt = {
      val (v, comp, sh) = viewFor(t)
      val sv            = if (value.tpe == comp || isPtr(value.tpe)) value else bind("sc", p.Expr.Cast(value, comp))
      p.Stmt.Update(sel(v), indexOf(off, sh), sv)
    }
    // store a struct/array value into the arena scalar-leaf by scalar-leaf (the dual of loadAgg); the source
    // is read field-wise through the normal term rewrite
    def storeAgg(off: p.Term, t: p.Type, src: p.Term): List[p.Stmt] = {
      val srcSel          = src match { case s: p.Term.Select => s; case _ => bindTerm("sv", src) }
      val (sRoot, sSteps) = (srcSel.root, srcSel.steps)
      val out             = ListBuffer.empty[p.Stmt]
      def copy(prefix: List[p.PathStep], o: p.Term, ft: p.Type): Unit = ft match {
        case s: p.Type.Struct =>
          canonicalMembers(s.name)
            .foreach(m =>
              copy(
                prefix :+ p.PathStep.Field(m.symbol),
                add(o, asI64(bind("of", p.Expr.OffsetOf(ft, m.symbol)))),
                i64ify(m.tpe)
              )
            )
        case p.Type.Arr(elem, n, _) =>
          (0 until n).foreach(e =>
            copy(prefix :+ p.PathStep.Index(e), add(o, mulBytes(i64(e.toLong), elem)), i64ify(elem))
          )
        case scalar => out += storeAt(o, scalar, rwTerm(p.Term.Select(sRoot, sSteps ::: prefix, scalar)))
      }
      copy(Nil, off, t)
      out.toList
    }
    def storeVal(off: p.Term, t: p.Type, value: p.Term): List[p.Stmt] =
      if (isAgg(t)) storeAgg(off, t, value) else List(storeAt(off, t, value))
    def byteSize(t: p.Type): p.Term = scalarBytes(t) match {
      case Some(n) => i64(n)
      case None    => asI64(bind("sz", p.Expr.SizeOf(t)))
    }
    def mulBytes(idx: p.Term, comp: p.Type): p.Term =
      if (idx == i64(0)) i64(0) else bind("mo", p.Expr.IntrOp(p.Intr.Mul(asI64(idx), byteSize(comp), I64)))

    def i64Var(n: p.Named): p.Named = p.Named(n.symbol, I64)
    def base(root: p.Named): (p.Term, p.Type) =
      if (root == capN) (i64(0), capTpe) else (sel(i64Var(root)), pointee(root.tpe))

    def rwStep(s: p.PathStep): p.PathStep = s match {
      case p.PathStep.IndexDyn(i) => p.PathStep.IndexDyn(rwTerm(i)); case x: p.PathStep => x
    }
    def bindTerm(hint: String, t: p.Term): p.Term.Select = {
      val n = fresh(hint, t.tpe); pre += p.Stmt.Var(n, Some(p.Expr.Alias(t)), isMutable = false); sel(n)
    }

    // Physical SPIR-V still uses ArenaView's typed scalar descriptors, but an immutable array
    // binding does not need a private aggregate copy.  Keep a pointer to the selected first
    // element and expose it through a dereferenced pointer-to-array binding.  The LLVM backend
    // can then bind the array name directly to that storage.  Restrict this to scalar arrays and
    // untouched bindings: aggregate elements need the ordinary field-wise materialisation, and a
    // write through a by-value binding must not unexpectedly alias the arena object.
    def arrayAlias(n: p.Named, e: p.Expr, isMutable: Boolean): Option[p.Stmt] =
      if (isMutable) None
      else
        (n.tpe, e) match {
          case (arr @ p.Type.Arr(component, length, _), p.Expr.Alias(source: p.Term.Select))
              if length > 0 && !isAgg(component) && source.steps.nonEmpty && lvalueOffset(
                source.root,
                source.steps
              ).nonEmpty =>
            if (mutatedRoots(n)) None
            else {
              val off              = lvalueOffset(source.root, source.steps).get
              val (view, _, shift) = viewFor(component)
              val elemPtrTpe       = p.Type.Ptr(component, Global)
              val elemPtr          = fresh("av", elemPtrTpe)
              val ptrTpe           = p.Type.Ptr(arr, Global)
              val ptr              = fresh("av", ptrTpe)
              val index            = indexOf(off, shift)
              val ref              = p.Expr.RefTo(sel(view), Some(index), component, Global, p.Region.Rooted(view))
              pre += p.Stmt.Var(elemPtr, Some(ref), isMutable = false)
              pre += p.Stmt.Var(ptr, Some(p.Expr.Cast(sel(elemPtr), ptrTpe)), isMutable = false)
              Some(
                p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(ptr, List(p.PathStep.Deref), arr))), isMutable = false)
              )
            }
          case _ => None
        }

    // arena byte-offset walk from a base offset + pointee type; a Field/Index on a loaded pointer field
    // auto-derefs it (the `ptr->field` idiom carries no explicit Deref), an explicit Deref does its own load
    def offsetFrom(off0: p.Term, cur0: p.Type, steps: List[p.PathStep]): p.Term = {
      def deref(off: p.Term, cur: p.Type): (p.Term, p.Type) = (loadAt(off, I64), pointee(cur))
      steps
        .foldLeft((off0, cur0)) {
          case ((off, cur), p.PathStep.Field(field)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (
              add(o, asI64(bind("of", p.Expr.OffsetOf(c, field)))),
              structSym(c).fold(c)(s => memberTpe(s, field))
            )
          case ((off, cur), p.PathStep.Deref) => deref(off, cur)
          case ((off, cur), p.PathStep.Index(k)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (add(o, mulBytes(i64(k.toLong), elem(c))), elem(c))
          case ((off, cur), p.PathStep.IndexDyn(idx)) =>
            val (o, c) = if (isPtr(cur)) deref(off, cur) else (off, cur)
            (add(o, mulBytes(rwTerm(idx), elem(c))), elem(c))
        }
        ._1
    }
    def offsetTo(root: p.Named, steps: List[p.PathStep]): p.Term = { val (o, c) = base(root); offsetFrom(o, c, steps) }

    // first pointer field a later step dereferences - the local->arena crossing in a Select rooted at a
    // local (an iterator's `_M_node` read off the stack, then chased in). ORIGINAL member types drive this
    def findCrossing(rootTpe: p.Type, steps: List[p.PathStep]): Option[(List[p.PathStep], p.Type, List[p.PathStep])] = {
      val n = steps.length
      def go(cur: p.Type, i: Int): Option[(List[p.PathStep], p.Type, List[p.PathStep])] =
        if (i >= n) None
        else
          steps(i) match {
            case p.PathStep.Field(f) =>
              val c  = if (isPtr(cur)) pointee(cur) else cur
              val ft = structSym(c).fold(c)(s => memberTpe(s, f))
              if (isPtr(ft) && i < n - 1) Some((steps.take(i + 1), pointee(ft), steps.drop(i + 1)))
              else go(ft, i + 1)
            case p.PathStep.Deref                             => go(pointee(cur), i + 1)
            case p.PathStep.Index(_) | p.PathStep.IndexDyn(_) => go(elem(cur), i + 1)
          }
      go(rootTpe, 0)
    }

    // arena byte offset of the lvalue a Select denotes; None if the whole access stays in local memory
    def lvalueOffset(root: p.Named, steps: List[p.PathStep]): Option[p.Term] =
      if (isArena(root)) Some(offsetTo(root, steps))
      else
        findCrossing(root.tpe, steps).map { case (prefix, pointeeT, suffix) =>
          offsetFrom(bindTerm("lo", p.Term.Select(root, prefix.map(rwStep), I64)), pointeeT, suffix)
        }

    // the i64 offset value a pointer-typed term denotes
    def ptrValue(t: p.Term): p.Term = t match {
      case p.Term.Select(root, Nil, _) => if (root == capN) i64(0) else sel(i64Var(root))
      case p.Term.Select(root, steps, _) =>
        lvalueOffset(root, steps) match {
          case Some(off) => loadAt(off, I64)
          case None      => p.Term.Select(root, steps.map(rwStep), I64) // pure-local pointer field, read directly
        }
      case _ => asI64(t)
    }

    def rwTerm(t: p.Term): p.Term = t match {
      case p.Term.Select(root, Nil, _) if root == capN => i64(0) // the capture itself is arena offset 0
      case p.Term.Select(root, Nil, _) if isArena(root) && isPtr(root.tpe) => sel(i64Var(root))
      case p.Term.Select(root, Nil, _) if fieldAliases.get(root).exists(x => identityFields(x._1)) =>
        rwTerm(fieldAliases(root)._2)
      case p.Term.Select(root, steps, resultT) if steps.nonEmpty =>
        lvalueOffset(root, steps) match {
          case Some(off) => loadAt(off, if (isPtr(resultT)) I64 else resultT)
          case None =>
            val result =
              if (isPtr(resultT) && (arenaTerm(t) || isEncodedField(root, steps))) i64ify(resultT) else resultT
            p.Term.Select(root, steps.map(rwStep), result)
        }
      case x => x
    }

    // i64 base offset for an indexed arena access (Some), else None to keep a real local pointer: a pointer
    // base is loaded (its value is the offset), an array base IS the offset (its lvalue location)
    def derefOffset(base: p.Term): Option[p.Term] =
      if (!arenaTerm(base)) None
      else if (isPtr(base.tpe)) Some(ptrValue(base))
      else Some(addrOffset(base))
    def scalarRefAt(off: p.Term, tpe: p.Type): p.Term = {
      val (view, _, sh) = viewFor(tpe)
      bind(
        "vr",
        p.Expr.RefTo(
          sel(view),
          Some(indexOf(off, sh)),
          tpe,
          Global,
          p.Region.Rooted(view)
        )
      )
    }
    def arenaScalarRef(ptr: p.Term, tpe: p.Type): p.Term = {
      if (isAgg(tpe))
        throw IllegalArgumentException(s"arena atomic access requires a scalar type; got ${tpe.repr}")
      scalarRefAt(
        derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
        tpe
      )
    }
    def volatileLoadAt(off: p.Term, tpe: p.Type): p.Term =
      if (!isAgg(tpe)) bind("vl", p.Expr.SpecOp(p.Spec.GpuVolatileLoad(scalarRefAt(off, tpe), tpe)))
      else {
        val value = fresh("vv", tpe)
        pre += p.Stmt.Var(value, None, isMutable = true)
        def load(prefix: List[p.PathStep], at: p.Term, fieldTpe: p.Type): Unit = fieldTpe match {
          case struct: p.Type.Struct =>
            canonicalMembers(struct.name).foreach { member =>
              load(
                prefix :+ p.PathStep.Field(member.symbol),
                add(at, asI64(bind("of", p.Expr.OffsetOf(fieldTpe, member.symbol)))),
                i64ify(member.tpe)
              )
            }
          case p.Type.Arr(component, size, _) =>
            (0 until size).foreach(index =>
              load(
                prefix :+ p.PathStep.Index(index),
                add(at, mulBytes(i64(index.toLong), component)),
                i64ify(component)
              )
            )
          case scalar =>
            pre += p.Stmt.Mut(
              p.Term.Select(value, prefix, scalar),
              p.Expr.Alias(bind("vl", p.Expr.SpecOp(p.Spec.GpuVolatileLoad(scalarRefAt(at, scalar), scalar))))
            )
        }
        load(Nil, off, tpe)
        sel(value)
      }
    def volatileStoreAt(off: p.Term, tpe: p.Type, value: p.Term): Unit = {
      val source          = value match { case s: p.Term.Select => s; case _ => bindTerm("vs", value) }
      val (root, initial) = (source.root, source.steps)
      def store(prefix: List[p.PathStep], at: p.Term, fieldTpe: p.Type): Unit = fieldTpe match {
        case struct: p.Type.Struct =>
          canonicalMembers(struct.name).foreach { member =>
            store(
              prefix :+ p.PathStep.Field(member.symbol),
              add(at, asI64(bind("of", p.Expr.OffsetOf(fieldTpe, member.symbol)))),
              i64ify(member.tpe)
            )
          }
        case p.Type.Arr(component, size, _) =>
          (0 until size).foreach(index =>
            store(
              prefix :+ p.PathStep.Index(index),
              add(at, mulBytes(i64(index.toLong), component)),
              i64ify(component)
            )
          )
        case scalar =>
          val done = fresh("vs", p.Type.Unit0)
          pre += p.Stmt.Var(
            done,
            Some(
              p.Expr.SpecOp(
                p.Spec.GpuVolatileStore(
                  scalarRefAt(at, scalar),
                  rwTerm(p.Term.Select(root, initial ::: prefix, scalar))
                )
              )
            ),
            isMutable = false
          )
      }
      store(Nil, off, tpe)
    }
    // offset of an arena data lvalue whose address is taken (`&obj.field`, field non-pointer)
    def addrOffset(base: p.Term): p.Term = base match {
      case p.Term.Select(root, steps, _) => lvalueOffset(root, steps).getOrElse(asI64(rwTerm(base)))
      case _                             => asI64(rwTerm(base))
    }
    def selectedIdentityField(t: p.Term): Boolean = t match {
      case p.Term.Select(root, steps, _: p.Type.Ptr) if steps.nonEmpty => isIdentityField(root, steps)
      case p.Term.Select(root, Nil, _: p.Type.Ptr) => fieldAliases.get(root).exists(x => identityFields(x._1))
      case _                                       => false
    }
    def identityComparable(t: p.Term): Option[p.Term] = t match {
      case _: p.Term.NullPtrConst                                           => Some(i64(0))
      case p.Term.Select(root, Nil, _) if localPointerTokens.contains(root) => Some(i64(localPointerTokens(root)))
      case p.Term.Select(root, Nil, _) if fieldAliases.get(root).exists(x => identityFields(x._1)) =>
        Some(rwTerm(fieldAliases(root)._2))
      case selected if selectedIdentityField(selected) => Some(rwTerm(selected))
      case _                                           => None
    }
    def equality(x: p.Term, y: p.Term, eq: (p.Term, p.Term) => p.Intr): p.Expr =
      if (selectedIdentityField(x) || selectedIdentityField(y)) {
        (identityComparable(x), identityComparable(y)) match {
          case (Some(a), Some(b)) => p.Expr.IntrOp(eq(a, b))
          case _                  => rewrittenEquality(x, y, eq)
        }
      } else rewrittenEquality(x, y, eq)

    def rewrittenEquality(x: p.Term, y: p.Term, eq: (p.Term, p.Term) => p.Intr): p.Expr = {
      val a = rwTerm(x)
      val b = rwTerm(y)
      val aa = x match {
        case _: p.Term.NullPtrConst if b.tpe == I64 => i64(0)
        case _                                      => a
      }
      val bb = y match {
        case _: p.Term.NullPtrConst if a.tpe == I64 => i64(0)
        case _                                      => b
      }
      p.Expr.IntrOp(eq(aa, bb))
    }

    def rwExpr(e: p.Expr): p.Expr = e match {
      case p.Expr.Alias(t) => p.Expr.Alias(rwTerm(t))
      case p.Expr.Cast(from, as) if isPtr(from.tpe) && arenaTerm(from) =>
        val v = ptrValue(from)
        if (isPtr(as) || as == I64) p.Expr.Alias(v) else p.Expr.Cast(v, as)
      case p.Expr.Cast(from, as) => p.Expr.Cast(rwTerm(from), as)
      case p.Expr.RefTo(base, idx, comp, p.Type.Space.Private, r) =>
        p.Expr.RefTo(rwTerm(base), idx.map(rwTerm), i64ify(comp), p.Type.Space.Private, r)
      case p.Expr.RefTo(base, idx, comp, sp, r) if arenaTerm(base) =>
        val off0 = if (isPtr(base.tpe)) ptrValue(base) else addrOffset(base)
        p.Expr.Alias(add(off0, idx.fold(i64(0))(i => mulBytes(rwTerm(i), comp))))
      case p.Expr.RefTo(base, idx, comp, sp, r) => p.Expr.RefTo(rwTerm(base), idx.map(rwTerm), comp, sp, r)
      case p.Expr.Index(base, idx, comp) =>
        derefOffset(base) match {
          case Some(off0) => p.Expr.Alias(loadAt(add(off0, mulBytes(rwTerm(idx), comp)), comp))
          case None       => p.Expr.Index(rwTerm(base), rwTerm(idx), i64ify(comp))
        }
      case p.Expr.Alloc(c, sz, sp, r)            => p.Expr.Alloc(c, rwTerm(sz), sp, r)
      case p.Expr.ForeignCall(n, args, rtn)      => p.Expr.ForeignCall(n, args.map(rwTerm), rtn)
      case p.Expr.Invoke(n, ts, recv, args, rtn) => p.Expr.Invoke(n, ts, recv.map(rwTerm), args.map(rwTerm), rtn)
      case p.Expr.IntrOp(p.Intr.LogicEq(x, y))   => equality(x, y, p.Intr.LogicEq.apply)
      case p.Expr.IntrOp(p.Intr.LogicNeq(x, y))  => equality(x, y, p.Intr.LogicNeq.apply)
      case op: p.Expr.IntrOp                     => op.modifyAll[p.Term](rwTerm)
      case op: p.Expr.MathOp                     => op.modifyAll[p.Term](rwTerm)
      case p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op, ptr, value, scope, order, rtn)) if arenaTerm(ptr) =>
        p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op, arenaScalarRef(ptr, rtn), rwTerm(value), scope, order, rtn))
      case p.Expr.SpecOp(p.Spec.GpuAtomicCAS(ptr, expected, desired, scope, order, rtn)) if arenaTerm(ptr) =>
        p.Expr.SpecOp(
          p.Spec.GpuAtomicCAS(arenaScalarRef(ptr, rtn), rwTerm(expected), rwTerm(desired), scope, order, rtn)
        )
      case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(ptr, rtn)) if arenaTerm(ptr) =>
        if (isAgg(rtn))
          p.Expr.Alias(
            volatileLoadAt(
              derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
              rtn
            )
          )
        else p.Expr.SpecOp(p.Spec.GpuVolatileLoad(arenaScalarRef(ptr, rtn), rtn))
      case p.Expr.SpecOp(p.Spec.GpuVolatileStore(ptr, value)) if arenaTerm(ptr) =>
        if (isAgg(value.tpe)) {
          volatileStoreAt(
            derefOffset(ptr).getOrElse(throw IllegalArgumentException(s"expected arena pointer: ${ptr.repr}")),
            value.tpe,
            value
          )
          p.Expr.Alias(p.Term.Unit0Const)
        } else p.Expr.SpecOp(p.Spec.GpuVolatileStore(arenaScalarRef(ptr, value.tpe), rwTerm(value)))
      case op: p.Expr.SpecOp => op.modifyAll[p.Term](rwTerm)
      case x                 => x
    }

    def rwArenaPointer(e: p.Expr): p.Expr = e match {
      case p.Expr.Alias(_: p.Term.NullPtrConst) => p.Expr.Alias(i64(0))
      case _                                    => rwExpr(e)
    }

    def rwInit(n: p.Named, e: p.Expr): (p.Named, p.Expr) =
      if (isArena(n) && isPtr(n.tpe)) i64Var(n) -> rwArenaPointer(e)
      else {
        val rewritten = rwExpr(e)
        (if (isPtr(n.tpe) && rewritten.tpe == I64) i64Var(n) else n) -> rewritten
      }

    val out = leaf match {
      case p.Stmt.Var(n, Some(e), m) =>
        arrayAlias(n, e, m).toList match {
          case Nil     => val (nn, ne) = rwInit(n, e); List(p.Stmt.Var(nn, Some(ne), m))
          case aliases => aliases
        }
      case p.Stmt.Var(n, None, m) => List(p.Stmt.Var(if (isArena(n) && isPtr(n.tpe)) i64Var(n) else n, None, m))
      case p.Stmt.Mut(p.Term.Select(n, Nil, t), e) =>
        if (isArena(n) && isPtr(n.tpe)) List(p.Stmt.Mut(p.Term.Select(i64Var(n), Nil, I64), rwArenaPointer(e)))
        else List(p.Stmt.Mut(p.Term.Select(n, Nil, t), rwExpr(e)))
      case p.Stmt.Mut(p.Term.Select(n, steps, scalarT), e) =>
        // mirrors rwTerm: a select crossing into arena partway (a stack-local iterator's node pointer
        // chased into the heap) still needs the byte-offset store, not a plain local field write
        lvalueOffset(n, steps) match {
          case Some(off) => storeVal(off, scalarT, bind("st", rwExpr(e)))
          case None      =>
            // local struct field write; a pointer field is now i64
            val identityField = isPtr(scalarT) && isIdentityField(n, steps)
            val lhsT          = if (isPtr(scalarT) && isEncodedField(n, steps)) i64ify(scalarT) else scalarT
            val rhs = e match {
              case p.Expr.Alias(_: p.Term.NullPtrConst) if identityField => p.Expr.Alias(i64(0))
              case p.Expr.Alias(p.Term.Select(root, Nil, _)) if identityField && localPointerTokens.contains(root) =>
                p.Expr.Alias(i64(localPointerTokens(root)))
              case p.Expr.Cast(source, _: p.Type.Ptr) if identityField =>
                identityComparable(source).map(p.Expr.Alias.apply).getOrElse(rwExpr(e))
              case p.Expr.RefTo(base: p.Term.Select, index, _, _, _) if identityField =>
                localReferenceKey(base, index)
                  .flatMap(localReferenceTokens.get)
                  .map(token => p.Expr.Alias(i64(token)))
                  .getOrElse(rwExpr(e))
              case _ => rwExpr(e)
            }
            List(p.Stmt.Mut(p.Term.Select(n, steps.map(rwStep), lhsT), rhs))
        }
      case p.Stmt.Update(base @ p.Term.Select(_, _, ptrT), idx, v) =>
        derefOffset(base) match {
          case Some(off0) => storeVal(add(off0, mulBytes(rwTerm(idx), elem(ptrT))), elem(ptrT), rwTerm(v))
          case None       => List(p.Stmt.Update(rwTerm(base).asInstanceOf[p.Term.Select], rwTerm(idx), rwTerm(v)))
        }
      case p.Stmt.Return(e) => List(p.Stmt.Return(rwExpr(e)))
      case s                => List(s)
    }
    (pre ++= out).toList
  }
}
