package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// picks the host launch ABI of an offload entry and lowers work-group kernels to one call per group.
// the choice is recorded as `#host_abi:<entry>` program metadata, which the backend exports next to the entry:
//   flat   (1)  the kernel cannot observe the group (no barrier, local array, or local/group id or size), so the
//               launcher runs one task per work-item and the body is untouched
//   group  (2)  the kernel observes the group: local arrays are hoisted once above lane loops that replace the
//               work-item, with local and global ids rewritten to the lane index. barrier-free kernels get one lane
//               loop; barrier kernels whose barriers all lanes reach together get one lane loop per barrier-free stretch
//               (GroupUniform), others are split into regions run by a per-lane state machine (GroupRegions)
//   lane   (0)  one thread per work-item (the default for an absent symbol): kernels whose lanes wait on each other
//               and kernels the lane loops cannot express
// examples:
//   out[global_id] = f(x)                       ->  unchanged, flat
//   var s: Local[F32 x 64]; s[local_id] = v     ->  var s; for lid in 0 until local_size { s[lid] = v }, group
//   s[lid] = v; barrier(); out[lid] = s[lid ^ 1] ->  two lane loops, group
//   while (volatile_load(flag) == 0) {}         ->  unchanged, lane, plus a `#host_abi_note:` entry
// edge cases:
//   early return in a lane       ->  an exit flag fences the rest of the lane body (shared with FnInline)
//   parameter written by a lane  ->  copied into a lane-private variable first
//   value-returning entry        ->  lane
//   re-run on a marked entry     ->  unchanged
// an entry reaching this pass must be fully lowered: no calls, exceptions, assertions, remote operations or group
// collectives, and barriers only in the `var _ = barrier()` form
object GroupFission extends ProgramPass {
  import GroupVocabulary.{
    Base,
    Global,
    GroupId,
    Groups,
    Lanes,
    Lid,
    Prepared,
    U32,
    addressOfOwn,
    isBarrier,
    ownStorage,
    params,
    renameRoots,
    sel,
    spec,
    u32
  }

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private enum Abi(val value: Int) {
    case Lane  extends Abi(0)
    case Flat  extends Abi(1)
    case Group extends Abi(2)
  }

  private val Exit     = p.Named("#gf_exit", p.Type.Bool1)
  private val Reserved = List(Lanes, GroupId, Base, Groups, Global, Lid, Exit)

  private def constructName(x: Any): String = x match {
    case product: Product => product.productPrefix
    case other            => other.toString
  }

  private def checkInvariants(entry: p.Function): Unit = {
    def fail(construct: String): Nothing =
      throw IllegalStateException(s"GroupFission: ${entry.name.repr}: ${construct} must be lowered before this pass")
    entry.collectAll[p.Stmt].foreach {
      case s @ (_: p.Stmt.Try | _: p.Stmt.Raise | p.Stmt.Rethrow) => fail(s"Stmt.${constructName(s)}")
      case _                                                      => ()
    }
    val exprs = entry.collectAll[p.Expr]
    exprs.foreach {
      case p.Expr.Invoke(_, _, _, _, _) => fail("Expr.Invoke")
      case p.Expr.SpecOp(op) =>
        op match {
          case p.Spec.Assert(_, _) | p.Spec.GpuGroupReduce(_, _, _) | p.Spec.GpuGroupInclusiveScan(_, _, _) |
              p.Spec.GpuGroupExclusiveScan(_, _, _) | p.Spec.RemoteLaunch(_, _, _, _, _, _, _, _, _, _, _) |
              p.Spec.RemoteAlloc(_, _) | p.Spec.RemoteTempAlloc(_, _) | p.Spec.RemoteFree(_, _) |
              p.Spec.RemoteMemcpy(_, _, _, _, _) | p.Spec.RemoteSync(_) =>
            fail(s"Spec.${constructName(op)}")
          case _ => ()
        }
      case _ => ()
    }
    val total = exprs.count { case p.Expr.SpecOp(op) => isBarrier(op); case _ => false }
    val declared = entry.collectAll[p.Stmt].count {
      case p.Stmt.Var(_, Some(p.Expr.SpecOp(op)), _) => isBarrier(op)
      case _                                         => false
    }
    if (total != declared) fail("a barrier outside a variable initialiser")
  }

  private def hasBarrier(entry: p.Function): Boolean =
    entry.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(op) => isBarrier(op)
      case _                 => false
    }

  private[pass] def isLocalArray(tpe: p.Type): Boolean = tpe match {
    case p.Type.Arr(_, _, p.Type.Space.Local) => true
    case _                                    => false
  }

  private def observesGroup(entry: p.Function): Boolean =
    entry.collectAll[p.Expr].exists {
      case p.Expr
            .SpecOp(p.Spec.GpuLocalIdx(_) | p.Spec.GpuLocalSize(_) | p.Spec.GpuGroupIdx(_) | p.Spec.GpuGroupSize(_)) =>
        true
      case p.Expr.Alloc(space = p.Type.Space.Local) => true
      case p.Expr.RefTo(space = p.Type.Space.Local) => true
      case _                                        => false
    } || entry.collectAll[p.Type].exists {
      case p.Type.Ptr(_, p.Type.Space.Local) => true
      case tpe                               => isLocalArray(tpe)
    }

  // a local or global id whose dimension is not a literal cannot be rewritten to the lane index
  private def rewritable(entry: p.Function): Boolean =
    entry.collectAll[p.Expr].forall {
      case p.Expr.SpecOp(p.Spec.GpuLocalIdx(dim))  => IntLiteral.isZero(dim).isDefined
      case p.Expr.SpecOp(p.Spec.GpuGlobalIdx(dim)) => IntLiteral.isZero(dim).isDefined
      case _                                       => true
    }

  private def isDimZero(dim: p.Term): Boolean = IntLiteral.isZero(dim).contains(true)

  // makes every declared symbol unique so hoisted declarations cannot be captured or merged by shadowing
  private def uniquify(entry: p.Function): List[p.Stmt] = {
    val taken = mutable.Set.from(entry.args.map(_.named.symbol) ++ entry.moduleCaptures.map(_.named.symbol))
    taken ++= entry.termCaptures.map(_.named.symbol) ++ entry.receiver.map(_.named.symbol)
    taken ++= Reserved.map(_.symbol)
    var counter = 0L

    type Env = Map[p.Named, p.Named]

    def declare(n: p.Named, env: Env): (p.Named, Env) =
      if (taken.add(n.symbol)) (n, env - n)
      else {
        var fresh = n.copy(symbol = s"${n.symbol}#gf_$counter")
        counter += 1
        while (!taken.add(fresh.symbol)) {
          fresh = n.copy(symbol = s"${n.symbol}#gf_$counter")
          counter += 1
        }
        (fresh, env + (n -> fresh))
      }

    def renamed(env: Env)(n: p.Named): p.Named = env.getOrElse(n, n)
    def rooted(env: Env)(region: p.Region): p.Region = region match {
      case p.Region.Rooted(root) => p.Region.Rooted(renamed(env)(root))
      case other                 => other
    }

    def sub[A](a: A, env: Env)(using Traversal[A, p.Term], Traversal[A, p.Expr]): A =
      if (env.isEmpty) a
      else
        a.modifyAll[p.Term] {
          case p.Term.Select(root, steps, tpe)          => p.Term.Select(renamed(env)(root), steps, tpe)
          case p.Term.NullPtrConst(comp, space, region) => p.Term.NullPtrConst(comp, space, rooted(env)(region))
          case other                                    => other
        }.modifyAll[p.Expr] {
          case p.Expr.RefTo(lhs, idx, comp, space, region) => p.Expr.RefTo(lhs, idx, comp, space, rooted(env)(region))
          case p.Expr.Alloc(comp, size, space, region)     => p.Expr.Alloc(comp, size, space, rooted(env)(region))
          case other                                       => other
        }

    def block(stmts: List[p.Stmt], env: Env): List[p.Stmt] = {
      var scope   = env
      val renamed = List.newBuilder[p.Stmt]
      stmts.foreach { s =>
        val (out, next) = stmt(s, scope)
        scope = next
        renamed += out
      }
      renamed.result()
    }

    def stmt(s: p.Stmt, env: Env): (p.Stmt, Env) = s match {
      case p.Stmt.Var(n, init, mut) =>
        val (renamed, next) = declare(n, env)
        (p.Stmt.Var(renamed, init.map(sub[p.Expr](_, env)), mut), next)
      case p.Stmt.Mut(lhs, rhs) => (p.Stmt.Mut(sub[p.Term.Select](lhs, env), sub[p.Expr](rhs, env)), env)
      case p.Stmt.Update(lhs, idx, v) =>
        (p.Stmt.Update(sub[p.Term.Select](lhs, env), sub[p.Term](idx, env), sub[p.Term](v, env)), env)
      case p.Stmt.While(c, b)         => (p.Stmt.While(sub[p.Term](c, env), block(b, env)), env)
      case p.Stmt.Cond(c, t, f)       => (p.Stmt.Cond(sub[p.Term](c, env), block(t, env), block(f, env)), env)
      case p.Stmt.Return(e)           => (p.Stmt.Return(sub[p.Expr](e, env)), env)
      case p.Stmt.Break | p.Stmt.Cont => (s, env)
      case p.Stmt.ForRange(i, lb, ub, st, b) =>
        val (renamed, next) = declare(i, env)
        (
          p.Stmt.ForRange(renamed, sub[p.Term](lb, env), sub[p.Term](ub, env), sub[p.Term](st, env), block(b, next)),
          env
        )
      case p.Stmt.Annotated(inner, pos, comment) =>
        val (out, next) = stmt(inner, env)
        (p.Stmt.Annotated(out, pos, comment), next)
      case other => throw IllegalStateException(s"GroupFission: unexpected ${constructName(other)} while renaming")
    }

    block(entry.body, Map.empty)
  }

  private def hoistLocals(body: List[p.Stmt]): (List[p.Stmt], List[p.Stmt]) = {
    val hoisted = List.newBuilder[p.Stmt]
    def go(stmts: List[p.Stmt]): List[p.Stmt] = {
      val out = List.newBuilder[p.Stmt]
      stmts.foreach {
        case p.Stmt.Var(n, init, mut) if isLocalArray(n.tpe) =>
          hoisted += p.Stmt.Var(n, None, mut)
          out ++= init.map(p.Stmt.Mut(sel(n), _))
        case p.Stmt.Cond(c, t, f)                  => out += p.Stmt.Cond(c, go(t), go(f))
        case p.Stmt.While(c, b)                    => out += p.Stmt.While(c, go(b))
        case p.Stmt.ForRange(i, lb, ub, st, b)     => out += p.Stmt.ForRange(i, lb, ub, st, go(b))
        case p.Stmt.Annotated(inner, pos, comment) => out ++= go(List(inner)).map(p.Stmt.Annotated(_, pos, comment))
        case other                                 => out += other
      }
      out.result()
    }
    val rest = go(body)
    (hoisted.result(), rest)
  }

  private def rewriteIds(body: List[p.Stmt]): (List[p.Stmt], Set[p.Named]) = {
    val used = mutable.Set.empty[p.Named]
    def read(n: p.Named): p.Expr = {
      used += n
      p.Expr.Alias(sel(n))
    }
    val out = body.modifyAll[p.Expr] {
      case p.Expr.SpecOp(p.Spec.GpuLocalIdx(dim)) if isDimZero(dim) => p.Expr.Alias(sel(Lid))
      case p.Expr.SpecOp(p.Spec.GpuGlobalIdx(dim)) if isDimZero(dim) =>
        used += Base
        p.Expr.IntrOp(p.Intr.Add(sel(Base), sel(Lid), U32))
      case p.Expr.SpecOp(p.Spec.GpuGroupIdx(dim)) if isDimZero(dim)   => read(GroupId)
      case p.Expr.SpecOp(p.Spec.GpuGroupSize(dim)) if isDimZero(dim)  => read(Groups)
      case p.Expr.SpecOp(p.Spec.GpuGlobalSize(dim)) if isDimZero(dim) => read(Global)
      case p.Expr.SpecOp(p.Spec.GpuLocalSize(dim)) if isDimZero(dim)  => read(Lanes)
      case other                                                      => other
    }
    (out, used.toSet)
  }

  // a field of a captured struct is the same in every lane, so it is read once instead of once per lane iteration
  private def hoistCaptures(entry: p.Function, body: List[p.Stmt], fresh: Fresh): (List[p.Stmt], List[p.Stmt]) = {
    val roots = params(entry).map(_.symbol).toSet
    val reads = mutable.LinkedHashMap.empty[(p.Named, List[p.PathStep]), p.Named]
    val out = body.modifyAll[p.Term] {
      case p.Term.Select(root, steps, tpe)
          if roots(root.symbol) && steps.nonEmpty && steps.forall(_.isInstanceOf[p.PathStep.Field]) &&
            GroupState.scalar(tpe) =>
        p.Term.Select(reads.getOrElseUpdate((root, steps), fresh("#gf_field", tpe)), Nil, tpe)
      case other => other
    }
    val decls = reads.toList.map { case ((root, steps), n) =>
      p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(root, steps, n.tpe))))
    }
    (decls, out)
  }

  private def laneBody(body: List[p.Stmt]): List[p.Stmt] = {
    var discards = 0L
    def onReturn(e: p.Expr): List[p.Stmt] = e match {
      case p.Expr.Alias(_) => Nil
      case other =>
        discards += 1
        List(p.Stmt.Var(p.Named(s"#gf_ret_$discards", p.Type.Unit0), Some(other)))
    }
    val (stmts, last) = body.lastOption match {
      case Some(p.Stmt.Return(e)) => (body.init, onReturn(e))
      case _                      => (body, Nil)
    }
    if (!stmts.exists(ReturnUnwind.containsReturn)) stmts ::: last
    else {
      val exit = sel(Exit)
      val tail = if (last.isEmpty) Nil else List(p.Stmt.Cond(exit, Nil, last))
      p.Stmt.Var(Exit, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true) ::
        ReturnUnwind(stmts, exit, onReturn, () => p.Named(s"#gf_pending_${discards}", p.Type.Bool1)) ::: tail
    }
  }

  // a work-group call shares the entry parameters between lanes, so a lane that writes one gets its own copy
  private def privatise(entry: p.Function, body: List[p.Stmt], fresh: Fresh): Either[String, List[p.Stmt]] = {
    val written = body.flatMap(_.collectWhere[p.Stmt] {
      case p.Stmt.Mut(p.Term.Select(root, steps, _), _) if ownStorage(root, steps) => root.symbol
      case p.Stmt.Update(p.Term.Select(root, steps, tpe), _, _)
          if !tpe.isInstanceOf[p.Type.Ptr] && !steps.contains(p.PathStep.Deref) =>
        root.symbol
    }) ::: body.flatMap(_.collectWhere[p.Expr] { case r: p.Expr.RefTo => addressOfOwn(r).map(_.symbol) }).flatten
    val writtenSet = written.toSet
    val copies     = params(entry).filter(n => writtenSet(n.symbol))
    if (copies.exists(_.tpe.isInstanceOf[p.Type.Arr])) Left("a lane writes a by-value array parameter")
    else if (copies.isEmpty) Right(body)
    else {
      val env  = copies.map(n => n.symbol -> fresh(s"${n.symbol}#gf_lane", n.tpe)).toMap
      val init = copies.map(n => p.Stmt.Var(env(n.symbol), Some(p.Expr.Alias(sel(n))), isMutable = true))
      Right(init ::: renameRoots(body, env))
    }
  }

  private def prepare(entry: p.Function): Either[String, Prepared] = {
    val renamed = uniquify(entry)
    val taken   = mutable.Set.from(params(entry).map(_.symbol) ::: Reserved.map(_.symbol))
    taken ++= renamed.flatMap(_.collectWhere[p.Stmt] {
      case p.Stmt.Var(n, _, _)            => n.symbol
      case p.Stmt.ForRange(induction = i) => i.symbol
    })
    val fresh = Fresh(taken)
    privatise(entry, renamed, fresh).map { body =>
      val (locals, rest)      = hoistLocals(body)
      val (identified, ids)   = rewriteIds(rest)
      val (fields, rewritten) = hoistCaptures(entry, identified, fresh)
      val groupId             = ids(GroupId) || ids(Base)
      val reads = List(
        Option.when(groupId)(p.Stmt.Var(GroupId, Some(spec(p.Spec.GpuGroupIdx(u32(0)))))),
        Option.when(ids(Base))(p.Stmt.Var(Base, Some(p.Expr.IntrOp(p.Intr.Mul(sel(GroupId), sel(Lanes), U32))))),
        Option.when(ids(Groups))(p.Stmt.Var(Groups, Some(spec(p.Spec.GpuGroupSize(u32(0)))))),
        Option.when(ids(Global))(p.Stmt.Var(Global, Some(spec(p.Spec.GpuGlobalSize(u32(0))))))
      ).flatten
      Prepared(
        entry,
        locals ::: p.Stmt.Var(Lanes, Some(spec(p.Spec.GpuLocalSize(u32(0))))) :: reads ::: fields,
        rewritten,
        fresh
      )
    }
  }

  private[pass] def checkHoisted(entry: p.Function, driver: List[p.Stmt]): Unit =
    driver
      .flatMap(_.collectAll[p.Stmt])
      .collectFirst { case v @ p.Stmt.Var(n, _, _) if isLocalArray(n.tpe) => v }
      .foreach(v =>
        throw IllegalStateException(s"GroupFission: ${entry.name.repr}: local array ${v.name.repr} was not hoisted")
      )

  private def toGroup(entry: p.Function, prepared: Prepared): p.Function = {
    val loop = p.Stmt.ForRange(Lid, u32(0), sel(Lanes), u32(1), laneBody(prepared.body))
    checkHoisted(entry, List(loop))
    entry.copy(body = prepared.prologue ::: List(loop, p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
  }

  private def position(pos: Option[p.SourcePosition]): String =
    pos.fold("")(x => s" at ${x.file}:${x.line}${x.col.fold("")(c => s":$c")}")

  override def apply(program: p.Program, log: Log): p.Program = lower(program, log, uniform = true)

  private[pass] def lower(program: p.Program, log: Log, uniform: Boolean): p.Program = program.entry match {
    case Some(entry)
        if entry.convention == p.CallConvention.OffloadEntry &&
          !program.metadata.exists(_.key == p.Conventions.HostAbiMeta + entry.name.fqcn) =>
      checkInvariants(entry)
      val name = entry.name.fqcn
      def marked(f: p.Function, abi: Abi, extra: List[p.MetaEntry] = Nil, defs: List[p.StructDef] = Nil) =
        program.copy(
          entry = Some(f),
          defs = program.defs ::: defs,
          metadata = program.metadata ::: p.MetaEntry(p.Conventions.HostAbiMeta + name, abi.value.toString) :: extra
        )
      def declined(reason: String) = {
        log.info(s"$name: $reason, lane abi")
        marked(entry, Abi.Lane)
      }
      WaitLoops.find(entry) match {
        case Some(pos) =>
          val note =
            s"kernel $name: loop${position(pos)} waits on a volatile or atomic value; launching with one thread per lane"
          log.info(note)
          marked(entry, Abi.Lane, List(p.MetaEntry(p.Conventions.HostAbiNoteMeta + name, note)))
        case None if !hasBarrier(entry) && !observesGroup(entry) =>
          log.info(s"$name: no group structure observed, flat abi")
          marked(entry, Abi.Flat)
        case None if entry.rtn != p.Type.Unit0 || !rewritable(entry) => declined("lane loops cannot express the entry")
        case None if hasBarrier(entry) =>
          prepare(entry).flatMap { prepared =>
            val fissioned = if (uniform) GroupUniform(prepared, program.defs) else Left("disabled")
            fissioned match {
              case Right(lowered) =>
                log.info(s"$name: barriers agree across lanes, ${lowered.regions} lane loops")
                Right(lowered)
              case Left(reason) =>
                if (uniform) log.info(s"$name: $reason, region machine")
                GroupRegions(prepared, program.defs).map { lowered =>
                  log.info(s"$name: split into ${lowered.regions} regions")
                  lowered
                }
            }
          } match {
            case Left(reason) => declined(reason)
            case Right(lowered) =>
              checkHoisted(entry, lowered.driver)
              marked(entry.copy(body = lowered.prologue ::: lowered.driver), Abi.Group, defs = lowered.defs)
          }
        case None =>
          prepare(entry) match {
            case Left(reason) => declined(reason)
            case Right(prepared) =>
              log.info(s"$name: lowered to one call per work-group")
              marked(toGroup(entry, prepared), Abi.Group)
          }
      }
    case _ => program
  }
}

private[pass] object IntLiteral {
  def isZero(t: p.Term): Option[Boolean] = t match {
    case p.Term.IntU8Const(v)  => Some(v == 0)
    case p.Term.IntU16Const(v) => Some(v == 0)
    case p.Term.IntU32Const(v) => Some(v == 0)
    case p.Term.IntU64Const(v) => Some(v == 0)
    case p.Term.IntS8Const(v)  => Some(v == 0)
    case p.Term.IntS16Const(v) => Some(v == 0)
    case p.Term.IntS32Const(v) => Some(v == 0)
    case p.Term.IntS64Const(v) => Some(v == 0)
    case _                     => None
  }
}

// structural detector for work-items that wait on each other without a barrier. values read through volatile loads
// and atomics are tainted, transitively through assignments, memory stores and the conditions they control. a loop
// waits when its condition or one of its exits (a conditional break or return) depends on a tainted value
// exempt, since sequential lanes always make progress through them:
//   counters      atomic add or sub of a non-zero literal
//   retry loops   a compare-and-swap whose expected value is re-read from memory on every iteration
// examples:
//   while (volatile_load(flag) == 0) {}         ->  wait
//   while (cas(lock, 0, 1) != 0) {}             ->  wait
//   loop { old = *p; if cas(p, old, f(old)) == old { break } }  ->  no wait
//   i = atomic_add(next, 1); while i < n { ... }                ->  no wait
private[pass] object WaitLoops {

  private def counter(op: p.AtomicOp, value: p.Term): Boolean = op match {
    case p.AtomicOp.Add | p.AtomicOp.Sub => IntLiteral.isZero(value).contains(false)
    case _                               => false
  }

  private def definitions(root: p.Named, body: List[p.Stmt]): List[p.Expr] =
    body.flatMap(_.collectWhere[p.Stmt] {
      case p.Stmt.Var(n, Some(e), _) if n == root => e
      case p.Stmt.Mut(lhs, e) if lhs.root == root => e
    })

  private def freshIn(root: p.Named, body: List[p.Stmt], seen: Set[p.Named]): Boolean = {
    val defs = definitions(root, body)
    defs.nonEmpty && defs.forall(readsMemory(_, body, seen + root))
  }

  private def readsMemory(e: p.Expr, body: List[p.Stmt], seen: Set[p.Named]): Boolean = e match {
    case p.Expr.Index(_, _, _) => true
    case p.Expr.SpecOp(
          p.Spec.GpuVolatileLoad(_, _) | p.Spec.GpuAtomicRMW(_, _, _, _, _, _) | p.Spec.GpuAtomicCAS(_, _, _, _, _, _)
        ) =>
      true
    case _ =>
      e.collectWhere[p.Term] { case p.Term.Select(root, steps, _) => (root, steps) }
        .exists((root, steps) => steps.contains(p.PathStep.Deref) || (!seen(root) && freshIn(root, body, seen)))
  }

  private def retry(expected: p.Term, loop: Option[List[p.Stmt]]): Boolean = (expected, loop) match {
    case (p.Term.Select(root, _, _), Some(body)) => freshIn(root, body, Set.empty)
    case _                                       => false
  }

  private def source(e: p.Expr, loop: Option[List[p.Stmt]]): Boolean = e match {
    case p.Expr.SpecOp(p.Spec.GpuVolatileLoad(_, _))                => true
    case p.Expr.SpecOp(p.Spec.GpuAtomicRMW(op = op, value = value)) => !counter(op, value)
    case p.Expr.SpecOp(p.Spec.GpuAtomicCAS(expected = expected))    => !retry(expected, loop)
    case _                                                          => false
  }

  private def returns(body: List[p.Stmt]): Boolean = body.exists(ReturnUnwind.containsReturn)

  private def leaves(body: List[p.Stmt]): Boolean = body.exists {
    case p.Stmt.Break | p.Stmt.Return(_) => true
    case p.Stmt.Cond(_, t, f)            => leaves(t) || leaves(f)
    case p.Stmt.While(_, b)              => returns(b)
    case p.Stmt.ForRange(body = b)       => returns(b)
    case p.Stmt.Annotated(inner, _, _)   => leaves(List(inner))
    case _                               => false
  }

  def find(entry: p.Function): Option[Option[p.SourcePosition]] = {
    val tainted = mutable.Set.empty[p.Named]
    var changed = false

    def taint(n: p.Named): Unit = if (tainted.add(n)) changed = true

    val tainting: PartialFunction[p.Term, Unit] = { case p.Term.Select(root, _, _) if tainted(root) => () }
    def reads(t: p.Term): Boolean               = t.collectFirst_[p.Term](tainting).isDefined
    def readsExpr(e: p.Expr): Boolean           = e.collectFirst_[p.Term](tainting).isDefined

    def walk(stmts: List[p.Stmt], ctrl: Boolean, loop: Option[List[p.Stmt]]): Unit = stmts.foreach {
      case p.Stmt.Var(n, Some(e), _) => if (ctrl || source(e, loop) || readsExpr(e)) taint(n)
      case p.Stmt.Mut(lhs, e)        => if (ctrl || source(e, loop) || readsExpr(e)) taint(lhs.root)
      case p.Stmt.Update(lhs, idx, value) =>
        if (ctrl || reads(value)) taint(lhs.root)
      case p.Stmt.Cond(c, t, f) =>
        val next = ctrl || reads(c)
        walk(t, next, loop)
        walk(f, next, loop)
      case p.Stmt.While(c, b) => walk(b, ctrl || reads(c), Some(b))
      case p.Stmt.ForRange(i, lb, ub, st, b) =>
        val next = ctrl || reads(lb) || reads(ub) || reads(st)
        if (next) taint(i)
        walk(b, next, Some(b))
      case p.Stmt.Annotated(inner, _, _) => walk(List(inner), ctrl, loop)
      case _                             => ()
    }

    while ({
      changed = false
      walk(entry.body, ctrl = false, loop = None)
      changed
    }) ()

    def exits(body: List[p.Stmt]): Boolean = body.exists {
      case p.Stmt.Cond(c, t, f)          => (reads(c) && (leaves(t) || leaves(f))) || exits(t) || exits(f)
      case p.Stmt.Annotated(inner, _, _) => exits(List(inner))
      case _                             => false
    }

    def search(stmts: List[p.Stmt], pos: Option[p.SourcePosition]): Option[Option[p.SourcePosition]] =
      stmts.iterator.flatMap(visit(_, pos)).nextOption()

    def visit(s: p.Stmt, pos: Option[p.SourcePosition]): Option[Option[p.SourcePosition]] = s match {
      case p.Stmt.Annotated(inner, at, _) => visit(inner, at.orElse(pos))
      case p.Stmt.While(c, b)             => if (reads(c) || exits(b)) Some(pos) else search(b, pos)
      case p.Stmt.ForRange(body = b)      => if (exits(b)) Some(pos) else search(b, pos)
      case p.Stmt.Cond(_, t, f)           => search(t, pos).orElse(search(f, pos))
      case _                              => None
    }

    search(entry.body, None)
  }
}
