package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// emulates fixed-width subgroup operations with workgroup-local scratch and barriers for targets without
// native subgroup support. subgroup size/lane become constants and local-id arithmetic; shuffles, votes and
// ballots exchange scalar or aggregate leaves through scratch sized to the configured workgroup ceiling
// examples:
//   subgroupSize()             ->  width
//   laneIdx()                  ->  linearLocalIdx & (width - 1)
//   shuffleDown(x, delta)      ->  scratch[localIdx] = x; barrier; scratch[subgroupBase + lane + delta]
//   voteAny(p) / voteAll(p)    ->  one predicate slot per subgroup, reduced by its lanes
//   ballot(p)                  ->  one predicate slot per workgroup lane, packed into an i32 mask
// edge cases:
//   source outside subgroup/workgroup  ->  shuffle retains the calling lane's value
//   aggregate shuffle                  ->  one scratch buffer per scalar leaf
//   control flow                       ->  native collectives require subgroup-uniform participation; emulation requires
//                                         whole-workgroup-uniform participation because it synchronises local scratch
//   width / maxGroupSize               ->  require a power-of-two width <= 32 and a divisible ceiling <= 1024
//   group lowering disabled            ->  leave GpuGroup* operations for a native backend
case class SubgroupLower(
    width: Int = 32,
    maxGroupSize: Int = 1024,
    lowerSubgroups: Boolean = true,
    lowerGroups: Boolean = false
) extends ProgramPass
    derives PassArgCodec {
  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  override def apply(program: p.Program, log: Log): p.Program = {
    require(
      width > 0 && width <= 32 && Integer.bitCount(width) == 1,
      s"width must be a power of two in [1, 32]: $width"
    )
    require(
      maxGroupSize >= width && maxGroupSize <= 1024 && maxGroupSize % width == 0,
      s"maxGroupSize must be a multiple of width in [$width, 1024]: $maxGroupSize"
    )
    Lowering(program, width, maxGroupSize, lowerSubgroups, lowerGroups).run()
  }
}

private final class Lowering(
    program: p.Program,
    width: Int,
    maxGroupSize: Int,
    lowerSubgroups: Boolean,
    lowerGroups: Boolean
) {
  private type Leaves = p.Type => List[(List[p.PathStep], p.Type)]

  private var counter = 0

  private def fresh(name: String, tpe: p.Type): p.Named = {
    counter += 1
    p.Named(s"#sg_${name}_$counter", tpe)
  }

  private def u32(value: Int): p.Term = p.Term.IntU32Const(value)

  private final class ScratchPool {
    private val pool = mutable.LinkedHashMap.empty[p.Type, mutable.ArrayBuffer[p.Named]]

    def allocFor(types: List[p.Type]): List[p.Named] = {
      val drawn = mutable.HashMap.empty[p.Type, Int]
      types.map { tpe =>
        val buffers = pool.getOrElseUpdate(tpe, mutable.ArrayBuffer.empty)
        val index   = drawn.getOrElse(tpe, 0)
        drawn(tpe) = index + 1
        if (index >= buffers.size) buffers += fresh("scratch", tpe)
        buffers(index)
      }
    }

    def declarations: List[p.Stmt] =
      pool.valuesIterator.flatMap(_.iterator).map(p.Stmt.Var(_, None, isMutable = false)).toList
  }

  def run(): p.Program = {
    validateUniformSubgroupParticipation()
    val definitions = program.defs.map(definition => definition.name -> definition).toMap
    val leaves: Leaves = {
      def loop(tpe: p.Type): List[(List[p.PathStep], p.Type)] = tpe match {
        case p.Type.Struct(name, _) =>
          definitions.get(name) match {
            case Some(definition) if definition.members.nonEmpty =>
              definition.members.flatMap(member =>
                loop(member.tpe).map((path, leafType) => (p.PathStep.Field(member.symbol) :: path, leafType))
              )
            case _ => List(Nil -> tpe)
          }
        case _ => List(Nil -> tpe)
      }
      loop
    }
    def lower(function: p.Function): p.Function = {
      val pool = ScratchPool()
      val body = mapStmtsRec(function.body) { leaf =>
        val (statement, prepended) = leaf.modifyCollect[p.Expr, List[p.Stmt]] {
          case p.Expr.SpecOp(op) => expand(op, leaves, pool).getOrElse((p.Expr.SpecOp(op), Nil))
          case expr              => (expr, Nil)
        }
        prepended.flatten ::: statement :: Nil
      }
      function.copy(body = pool.declarations ::: body)
    }
    program.copy(entry = program.entry.map(lower), functions = program.functions.map(lower))
  }

  private def validateUniformSubgroupParticipation(): Unit = {
    if (!lowerSubgroups && !lowerGroups) return

    def requiresBarrier(op: p.Spec): Boolean = op match {
      case _: p.Spec.GpuShuffleDown | _: p.Spec.GpuShuffleUp | _: p.Spec.GpuShuffleIdx | _: p.Spec.GpuShuffleXor |
          _: p.Spec.GpuVoteAny | _: p.Spec.GpuVoteAll | _: p.Spec.GpuBallot | _: p.Spec.GpuSubgroupBarrier
          if lowerSubgroups =>
        true
      case _: p.Spec.GpuGroupReduce | _: p.Spec.GpuGroupInclusiveScan | _: p.Spec.GpuGroupExclusiveScan
          if lowerGroups =>
        true
      case _ => false
    }
    def direct(function: p.Function): Boolean = function.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(op) => requiresBarrier(op)
      case _                 => false
    }
    def calls(function: p.Function): Set[p.Sym] = function
      .collectAll[p.Expr]
      .collect { case invoke: p.Expr.Invoke =>
        invoke.calleeSym
      }
      .flatten
      .toSet

    val functions = program.entry.toList ::: program.functions
    var requiring = functions.filter(direct).map(_.decl.name).toSet
    var changed   = true
    while (changed) {
      changed = false
      functions.foreach { function =>
        if (!requiring(function.decl.name) && calls(function).exists(requiring)) {
          requiring += function.decl.name
          changed = true
        }
      }
    }

    def synchronises(stmts: List[p.Stmt]): Boolean = stmts.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(op)     => requiresBarrier(op)
      case invoke: p.Expr.Invoke => invoke.calleeSym.exists(requiring)
      case _                     => false
    }
    def reject(function: p.Function, kind: String, body: List[p.Stmt]): Unit =
      if (synchronises(body))
        throw IllegalArgumentException(
          s"Subgroup emulation requires whole-workgroup-uniform participation; ${function.decl.name.fqcn} contains a synchronising subgroup operation under $kind"
        )
    def exits(body: List[p.Stmt]): Boolean = body.collectAll[p.Stmt].exists {
      case _: p.Stmt.Return | _: p.Stmt.Raise | p.Stmt.Rethrow | p.Stmt.Break | p.Stmt.Cont => true
      case _                                                                                => false
    }
    def rejectEarlyExit(function: p.Function, kind: String, body: List[p.Stmt]): Unit =
      if (requiring(function.decl.name) && exits(body))
        throw IllegalArgumentException(
          s"Subgroup emulation requires whole-workgroup-uniform participation; ${function.decl.name.fqcn} contains an early exit under $kind"
        )
    def validate(function: p.Function, stmts: List[p.Stmt]): Unit = stmts.foreach {
      case p.Stmt.Cond(_, trueBr, falseBr) =>
        reject(function, "conditional control flow", trueBr)
        reject(function, "conditional control flow", falseBr)
        rejectEarlyExit(function, "conditional control flow", trueBr)
        rejectEarlyExit(function, "conditional control flow", falseBr)
        validate(function, trueBr)
        validate(function, falseBr)
      case p.Stmt.While(_, body) =>
        reject(function, "loop control flow", body)
        rejectEarlyExit(function, "loop control flow", body)
        validate(function, body)
      case p.Stmt.ForRange(_, _, _, _, body) =>
        reject(function, "loop control flow", body)
        rejectEarlyExit(function, "loop control flow", body)
        validate(function, body)
      case p.Stmt.Try(body, handlers, fin) =>
        reject(function, "exceptional control flow", body)
        rejectEarlyExit(function, "exceptional control flow", body)
        handlers.foreach { handler =>
          reject(function, "exceptional control flow", handler.body)
          rejectEarlyExit(function, "exceptional control flow", handler.body)
        }
        reject(function, "exceptional control flow", fin)
        rejectEarlyExit(function, "exceptional control flow", fin)
      case p.Stmt.Raise(_, _, cleanup) =>
        reject(function, "exceptional cleanup", cleanup)
        rejectEarlyExit(function, "exceptional cleanup", cleanup)
        validate(function, cleanup)
      case p.Stmt.Annotated(inner, _, _) => validate(function, List(inner))
      case _                             => ()
    }
    functions.foreach(function => validate(function, function.body))
  }

  private def barrier: p.Stmt =
    p.Stmt.Var(
      fresh("barrier", p.Type.Unit0),
      Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)),
      isMutable = false
    )

  private def lanePrelude(
      localId: p.Named,
      localSize: p.Named,
      lane: p.Named,
      base: Option[p.Named] = None
  ): List[p.Stmt] =
    subgroupPrelude(localId, localSize) ::: List(
      p.Stmt.Var(
        lane,
        Some(p.Expr.IntrOp(p.Intr.BAnd(sel(localId), u32(width - 1), p.Type.IntU32))),
        isMutable = false
      )
    ) ::: base.toList.map(name =>
      p.Stmt.Var(
        name,
        Some(p.Expr.IntrOp(p.Intr.Sub(sel(localId), sel(lane), p.Type.IntU32))),
        isMutable = false
      )
    )

  private def membership(name: String, lane: p.Term, mask: p.Term): (p.Named, List[p.Stmt]) = {
    val bit     = fresh(s"${name}_bit", p.Type.IntU32)
    val shifted = fresh(s"${name}_shifted", p.Type.IntU32)
    val set     = fresh(s"${name}_set", p.Type.IntU32)
    val bounded = fresh(s"${name}_bounded", p.Type.Bool1)
    val present = fresh(s"${name}_present", p.Type.Bool1)
    val member  = fresh(s"${name}_member", p.Type.Bool1)
    member -> List(
      p.Stmt.Var(bit, Some(p.Expr.IntrOp(p.Intr.BAnd(lane, u32(31), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(shifted, Some(p.Expr.IntrOp(p.Intr.BSR(mask, sel(bit), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(set, Some(p.Expr.IntrOp(p.Intr.BAnd(sel(shifted), u32(1), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(bounded, Some(p.Expr.IntrOp(p.Intr.LogicLt(lane, u32(32)))), isMutable = false),
      p.Stmt.Var(present, Some(p.Expr.IntrOp(p.Intr.LogicNeq(sel(set), u32(0)))), isMutable = false),
      p.Stmt.Var(member, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(bounded), sel(present)))), isMutable = false)
    )
  }

  private def expand(op: p.Spec, leaves: Leaves, pool: ScratchPool): Option[(p.Expr, List[p.Stmt])] = op match {
    case p.Spec.GpuSubgroupSize if lowerSubgroups => Some((p.Expr.Alias(u32(width)), Nil))
    case p.Spec.GpuLaneIdx if lowerSubgroups =>
      val localId   = fresh("local_id", p.Type.IntU32)
      val localSize = fresh("local_size", p.Type.IntU32)
      Some(
        p.Expr.IntrOp(p.Intr.BAnd(sel(localId), u32(width - 1), p.Type.IntU32)) ->
          subgroupPrelude(localId, localSize)
      )
    case p.Spec.GpuShuffleDown(value, delta, clamp, mask, rtn) if lowerSubgroups =>
      Some(
        shuffle(
          value,
          clamp,
          mask,
          rtn,
          leaves,
          pool,
          (lane, _) => (p.Expr.IntrOp(p.Intr.Add(lane, delta, p.Type.IntU32)), Nil)
        )
      )
    case p.Spec.GpuShuffleUp(value, delta, clamp, mask, rtn) if lowerSubgroups =>
      Some(
        shuffle(
          value,
          clamp,
          mask,
          rtn,
          leaves,
          pool,
          (lane, _) => (p.Expr.IntrOp(p.Intr.Sub(lane, delta, p.Type.IntU32)), Nil)
        )
      )
    case p.Spec.GpuShuffleIdx(value, sourceLane, clamp, mask, rtn) if lowerSubgroups =>
      Some(
        shuffle(
          value,
          clamp,
          mask,
          rtn,
          leaves,
          pool,
          (_, segmentBase) => {
            val relative = fresh("relative", p.Type.IntU32)
            p.Expr.IntrOp(p.Intr.BOr(segmentBase, sel(relative), p.Type.IntU32)) -> List(
              p.Stmt
                .Var(relative, Some(p.Expr.IntrOp(p.Intr.BAnd(sourceLane, clamp, p.Type.IntU32))), isMutable = false)
            )
          }
        )
      )
    case p.Spec.GpuShuffleXor(value, laneMask, clamp, mask, rtn) if lowerSubgroups =>
      Some(
        shuffle(
          value,
          clamp,
          mask,
          rtn,
          leaves,
          pool,
          (lane, _) => (p.Expr.IntrOp(p.Intr.BXor(lane, laneMask, p.Type.IntU32)), Nil)
        )
      )
    case p.Spec.GpuVoteAny(mask, predicate) if lowerSubgroups => Some(vote(mask, predicate, pool, all = false))
    case p.Spec.GpuVoteAll(mask, predicate) if lowerSubgroups => Some(vote(mask, predicate, pool, all = true))
    case p.Spec.GpuBallot(mask, predicate) if lowerSubgroups  => Some(ballot(mask, predicate, pool))
    case p.Spec.GpuGroupReduce(op, value, rtn) if lowerGroups => Some(groupReduce(op, value, rtn, pool))
    case p.Spec.GpuGroupInclusiveScan(op, value, rtn) if lowerGroups =>
      Some(groupScan(op, value, rtn, pool, inclusive = true))
    case p.Spec.GpuGroupExclusiveScan(op, value, rtn) if lowerGroups =>
      Some(groupScan(op, value, rtn, pool, inclusive = false))
    case p.Spec.GpuSubgroupBarrier(p.Term.IntU32Const(-1)) if lowerSubgroups =>
      Some((p.Expr.SpecOp(p.Spec.GpuBarrierLocal), Nil))
    case p.Spec.GpuSubgroupBarrier(_) if lowerSubgroups =>
      throw IllegalArgumentException("Masked subgroup barriers cannot be emulated with a work-group barrier")
    case _ => None
  }

  private def shuffle(
      value: p.Term,
      clamp: p.Term,
      mask: p.Term,
      rtn: p.Type,
      leaves: Leaves,
      pool: ScratchPool,
      sourceOf: (p.Term, p.Term) => (p.Expr, List[p.Stmt])
  ): (p.Expr, List[p.Stmt]) = {
    val (valueBinding, valueSelect) = value match {
      case pointer if pointer.tpe match {
            case p.Type.Ptr(comp, _) => comp == rtn
            case _                   => false
          } =>
        val name = fresh("value", rtn)
        (List(p.Stmt.Var(name, Some(p.Expr.Index(pointer, u32(0), rtn)), isMutable = false)), sel(name))
      case select: p.Term.Select => (Nil, select)
      case other =>
        val name = fresh("value", rtn)
        (List(p.Stmt.Var(name, Some(p.Expr.Alias(other)), isMutable = false)), sel(name))
    }
    val leafList     = leaves(rtn)
    val buffers      = pool.allocFor(leafList.map((_, tpe) => p.Type.Arr(tpe, maxGroupSize, p.Type.Space.Local)))
    val fields       = leafList.zip(buffers).map { case ((path, tpe), buffer) => (path, buffer, tpe) }
    val localId      = fresh("local_id", p.Type.IntU32)
    val localSize    = fresh("local_size", p.Type.IntU32)
    val lane         = fresh("lane", p.Type.IntU32)
    val base         = fresh("base", p.Type.IntU32)
    val inverseClamp = fresh("inverse_clamp", p.Type.IntU32)
    val segmentBase  = fresh("segment_base", p.Type.IntU32)
    val segmentLast  = fresh("segment_last", p.Type.IntU32)
    val target       = fresh("target", p.Type.IntU32)
    val source       = fresh("source", p.Type.IntU32)
    val aboveBase    = fresh("above_base", p.Type.Bool1)
    val belowLast    = fresh("below_last", p.Type.Bool1)
    val inSegment    = fresh("in_segment", p.Type.Bool1)
    val inSubgroup   = fresh("in_subgroup", p.Type.Bool1)
    val inGroup      = fresh("in_group", p.Type.Bool1)
    val segmentLane  = fresh("segment_lane", p.Type.Bool1)
    val spatial      = fresh("spatial", p.Type.Bool1)
    val members      = fresh("members", p.Type.Bool1)
    val inRange      = fresh("in_range", p.Type.Bool1)
    val result       = fresh("result", rtn)
    val (callerMember, callerMembership) = membership("caller", sel(lane), mask)
    val (sourceMember, sourceMembership) = membership("source", sel(target), mask)
    val (targetExpr, targetPrelude)      = sourceOf(sel(lane), sel(segmentBase))

    def valueField(path: List[p.PathStep], tpe: p.Type): p.Term.Select =
      p.Term.Select(valueSelect.root, valueSelect.steps ::: path, tpe)
    def resultField(path: List[p.PathStep], tpe: p.Type): p.Term.Select = p.Term.Select(result, path, tpe)

    val statements = valueBinding ::: lanePrelude(localId, localSize, lane, Some(base)) :::
      fields.map((path, buffer, tpe) => p.Stmt.Update(sel(buffer), sel(localId), valueField(path, tpe))) :::
      List(
        barrier,
        p.Stmt.Var(inverseClamp, Some(p.Expr.IntrOp(p.Intr.BNot(clamp, p.Type.IntU32))), isMutable = false),
        p.Stmt.Var(
          segmentBase,
          Some(p.Expr.IntrOp(p.Intr.BAnd(sel(lane), sel(inverseClamp), p.Type.IntU32))),
          isMutable = false
        ),
        p.Stmt
          .Var(segmentLast, Some(p.Expr.IntrOp(p.Intr.BOr(sel(segmentBase), clamp, p.Type.IntU32))), isMutable = false)
      ) ::: targetPrelude ::: List(
        p.Stmt.Var(target, Some(targetExpr), isMutable = false)
      ) ::: callerMembership ::: sourceMembership ::: List(
        p.Stmt.Var(source, Some(p.Expr.IntrOp(p.Intr.Add(sel(base), sel(target), p.Type.IntU32))), isMutable = false),
        p.Stmt.Var(aboveBase, Some(p.Expr.IntrOp(p.Intr.LogicGte(sel(target), sel(segmentBase)))), isMutable = false),
        p.Stmt.Var(belowLast, Some(p.Expr.IntrOp(p.Intr.LogicLte(sel(target), sel(segmentLast)))), isMutable = false),
        p.Stmt.Var(inSegment, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(aboveBase), sel(belowLast)))), isMutable = false),
        p.Stmt.Var(
          inSubgroup,
          Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(target), u32(width)))),
          isMutable = false
        ),
        p.Stmt.Var(
          inGroup,
          Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(localSize)))),
          isMutable = false
        ),
        p.Stmt
          .Var(segmentLane, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(inSegment), sel(inSubgroup)))), isMutable = false),
        p.Stmt.Var(spatial, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(segmentLane), sel(inGroup)))), isMutable = false),
        p.Stmt
          .Var(members, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(callerMember), sel(sourceMember)))), isMutable = false),
        p.Stmt.Var(
          inRange,
          Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(spatial), sel(members)))),
          isMutable = false
        ),
        p.Stmt.Var(result, None, isMutable = true),
        p.Stmt.Cond(
          sel(inRange),
          fields.map((path, buffer, tpe) =>
            p.Stmt.Mut(resultField(path, tpe), p.Expr.Index(sel(buffer), sel(source), tpe))
          ),
          fields.map((path, _, tpe) => p.Stmt.Mut(resultField(path, tpe), p.Expr.Alias(valueField(path, tpe))))
        ),
        barrier
      )
    (p.Expr.Alias(sel(result)), statements)
  }

  private def groupPrelude(localId: p.Named, localSize: p.Named): List[p.Stmt] = {
    val x    = fresh("group_x", p.Type.IntU32)
    val y    = fresh("group_y", p.Type.IntU32)
    val z    = fresh("group_z", p.Type.IntU32)
    val sx   = fresh("group_sx", p.Type.IntU32)
    val sy   = fresh("group_sy", p.Type.IntU32)
    val sz   = fresh("group_sz", p.Type.IntU32)
    val syx  = fresh("group_syx", p.Type.IntU32)
    val yx   = fresh("group_yx", p.Type.IntU32)
    val szyx = fresh("group_szyx", p.Type.IntU32)
    val sxsy = fresh("group_sxsy", p.Type.IntU32)
    List(
      p.Stmt.Var(x, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(0)))), isMutable = false),
      p.Stmt.Var(y, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(1)))), isMutable = false),
      p.Stmt.Var(z, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(2)))), isMutable = false),
      p.Stmt.Var(sx, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(0)))), isMutable = false),
      p.Stmt.Var(sy, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(1)))), isMutable = false),
      p.Stmt.Var(sz, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(2)))), isMutable = false),
      p.Stmt.Var(syx, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sy), sel(x), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(yx, Some(p.Expr.IntrOp(p.Intr.Add(sel(y), sel(syx), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(szyx, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sz), sel(yx), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(
        localId,
        Some(p.Expr.IntrOp(p.Intr.Add(sel(z), sel(szyx), p.Type.IntU32))),
        isMutable = false
      ),
      p.Stmt.Var(sxsy, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sx), sel(sy), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(
        localSize,
        Some(p.Expr.IntrOp(p.Intr.Mul(sel(sxsy), sel(sz), p.Type.IntU32))),
        isMutable = false
      )
    )
  }

  private def subgroupPrelude(localId: p.Named, localSize: p.Named): List[p.Stmt] = {
    val x    = fresh("subgroup_x", p.Type.IntU32)
    val y    = fresh("subgroup_y", p.Type.IntU32)
    val z    = fresh("subgroup_z", p.Type.IntU32)
    val sx   = fresh("subgroup_sx", p.Type.IntU32)
    val sy   = fresh("subgroup_sy", p.Type.IntU32)
    val sz   = fresh("subgroup_sz", p.Type.IntU32)
    val syz  = fresh("subgroup_syz", p.Type.IntU32)
    val yz   = fresh("subgroup_yz", p.Type.IntU32)
    val sxyz = fresh("subgroup_sxyz", p.Type.IntU32)
    val sxsy = fresh("subgroup_sxsy", p.Type.IntU32)
    List(
      p.Stmt.Var(x, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(0)))), isMutable = false),
      p.Stmt.Var(y, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(1)))), isMutable = false),
      p.Stmt.Var(z, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(2)))), isMutable = false),
      p.Stmt.Var(sx, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(0)))), isMutable = false),
      p.Stmt.Var(sy, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(1)))), isMutable = false),
      p.Stmt.Var(sz, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(2)))), isMutable = false),
      p.Stmt.Var(syz, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sy), sel(z), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(yz, Some(p.Expr.IntrOp(p.Intr.Add(sel(y), sel(syz), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(sxyz, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sx), sel(yz), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(
        localId,
        Some(p.Expr.IntrOp(p.Intr.Add(sel(x), sel(sxyz), p.Type.IntU32))),
        isMutable = false
      ),
      p.Stmt.Var(sxsy, Some(p.Expr.IntrOp(p.Intr.Mul(sel(sx), sel(sy), p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(
        localSize,
        Some(p.Expr.IntrOp(p.Intr.Mul(sel(sxsy), sel(sz), p.Type.IntU32))),
        isMutable = false
      )
    )
  }

  private def combine(lhs: p.Term, rhs: p.Term, op: p.AtomicOp, tpe: p.Type): p.Expr = op match {
    case p.AtomicOp.Add                        => p.Expr.IntrOp(p.Intr.Add(lhs, rhs, tpe))
    case p.AtomicOp.And if tpe == p.Type.Bool1 => p.Expr.IntrOp(p.Intr.LogicAnd(lhs, rhs))
    case p.AtomicOp.Or if tpe == p.Type.Bool1  => p.Expr.IntrOp(p.Intr.LogicOr(lhs, rhs))
    case p.AtomicOp.And                        => p.Expr.IntrOp(p.Intr.BAnd(lhs, rhs, tpe))
    case p.AtomicOp.Or                         => p.Expr.IntrOp(p.Intr.BOr(lhs, rhs, tpe))
    case p.AtomicOp.Xor                        => p.Expr.IntrOp(p.Intr.BXor(lhs, rhs, tpe))
    case p.AtomicOp.Min                        => p.Expr.IntrOp(p.Intr.Min(lhs, rhs, tpe))
    case p.AtomicOp.Max                        => p.Expr.IntrOp(p.Intr.Max(lhs, rhs, tpe))
    case p.AtomicOp.Xchg | p.AtomicOp.Sub =>
      throw IllegalArgumentException(s"work-group lowering requires an associative operation; got ${op.repr}")
  }

  private def groupReduce(op: p.AtomicOp, value: p.Term, rtn: p.Type, pool: ScratchPool): (p.Expr, List[p.Stmt]) = {
    val scratch   = pool.allocFor(List(p.Type.Arr(rtn, maxGroupSize, p.Type.Space.Local))).head
    val localId   = fresh("group_local_id", p.Type.IntU32)
    val localSize = fresh("group_local_size", p.Type.IntU32)
    val setup = groupPrelude(localId, localSize) ::: List(
      p.Stmt.Update(sel(scratch), sel(localId), value),
      barrier
    )
    val stages = Iterator
      .iterate(Integer.highestOneBit(maxGroupSize - 1))(_ / 2)
      .takeWhile(_ > 0)
      .flatMap { stride =>
        val peer     = fresh("group_peer", p.Type.IntU32)
        val below    = fresh("group_below", p.Type.Bool1)
        val inRange  = fresh("group_in_range", p.Type.Bool1)
        val active   = fresh("group_active", p.Type.Bool1)
        val current  = fresh("group_current", rtn)
        val incoming = fresh("group_incoming", rtn)
        val combined = fresh("group_combined", rtn)
        List(
          p.Stmt
            .Var(peer, Some(p.Expr.IntrOp(p.Intr.Add(sel(localId), u32(stride), p.Type.IntU32))), isMutable = false),
          p.Stmt.Var(below, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(localId), u32(stride)))), isMutable = false),
          p.Stmt.Var(inRange, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(peer), sel(localSize)))), isMutable = false),
          p.Stmt.Var(
            active,
            Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(below), sel(inRange)))),
            isMutable = false
          ),
          p.Stmt.Cond(
            sel(active),
            List(
              p.Stmt.Var(current, Some(p.Expr.Index(sel(scratch), sel(localId), rtn)), isMutable = false),
              p.Stmt.Var(incoming, Some(p.Expr.Index(sel(scratch), sel(peer), rtn)), isMutable = false),
              p.Stmt.Var(combined, Some(combine(sel(current), sel(incoming), op, rtn)), isMutable = false),
              p.Stmt.Update(sel(scratch), sel(localId), sel(combined))
            ),
            Nil
          ),
          barrier
        )
      }
      .toList
    val result = fresh("group_result", rtn)
    (
      p.Expr.Alias(sel(result)),
      setup ::: stages ::: List(
        p.Stmt.Var(result, Some(p.Expr.Index(sel(scratch), u32(0), rtn)), false),
        barrier
      )
    )
  }

  private def zero(tpe: p.Type): p.Term = tpe match {
    case p.Type.IntS8   => p.Term.IntS8Const(0)
    case p.Type.IntU8   => p.Term.IntU8Const(0)
    case p.Type.IntS16  => p.Term.IntS16Const(0)
    case p.Type.IntU16  => p.Term.IntU16Const(0)
    case p.Type.IntS32  => p.Term.IntS32Const(0)
    case p.Type.IntU32  => p.Term.IntU32Const(0)
    case p.Type.IntS64  => p.Term.IntS64Const(0)
    case p.Type.IntU64  => p.Term.IntU64Const(0)
    case p.Type.Float32 => p.Term.Float32Const(0f)
    case p.Type.Float64 => p.Term.Float64Const(0d)
    case other =>
      throw IllegalArgumentException(s"work-group exclusive scan has no additive identity for ${other.repr}")
  }

  private def groupScan(
      op: p.AtomicOp,
      value: p.Term,
      rtn: p.Type,
      pool: ScratchPool,
      inclusive: Boolean
  ): (p.Expr, List[p.Stmt]) = {
    if (!inclusive && op != p.AtomicOp.Add)
      throw IllegalArgumentException(s"work-group exclusive scan lowering supports only add; got ${op.repr}")
    val buffers   = pool.allocFor(List.fill(2)(p.Type.Arr(rtn, maxGroupSize, p.Type.Space.Local)))
    val localId   = fresh("group_local_id", p.Type.IntU32)
    val localSize = fresh("group_local_size", p.Type.IntU32)
    val setup = groupPrelude(localId, localSize) ::: List(
      p.Stmt.Update(sel(buffers.head), sel(localId), value),
      barrier
    )
    var source = buffers.head
    var target = buffers(1)
    val stages = Iterator
      .iterate(1)(_ * 2)
      .takeWhile(_ < maxGroupSize)
      .flatMap { stride =>
        val active   = fresh("group_active", p.Type.Bool1)
        val peer     = fresh("group_peer", p.Type.IntU32)
        val current  = fresh("group_current", rtn)
        val incoming = fresh("group_incoming", rtn)
        val next     = fresh("group_next", rtn)
        val statements = List(
          p.Stmt.Var(active, Some(p.Expr.IntrOp(p.Intr.LogicGte(sel(localId), u32(stride)))), isMutable = false),
          p.Stmt
            .Var(peer, Some(p.Expr.IntrOp(p.Intr.Sub(sel(localId), u32(stride), p.Type.IntU32))), isMutable = false),
          p.Stmt.Var(current, Some(p.Expr.Index(sel(source), sel(localId), rtn)), isMutable = false),
          p.Stmt.Var(next, Some(p.Expr.Alias(sel(current))), isMutable = true),
          p.Stmt.Cond(
            sel(active),
            List(
              p.Stmt.Var(
                incoming,
                Some(
                  p.Expr.Index(sel(source), sel(peer), rtn)
                ),
                isMutable = false
              ),
              p.Stmt.Mut(sel(next), combine(sel(incoming), sel(current), op, rtn))
            ),
            Nil
          ),
          p.Stmt.Update(sel(target), sel(localId), sel(next)),
          barrier
        )
        val previous = source
        source = target
        target = previous
        statements
      }
      .toList
    val result = fresh("group_result", rtn)
    val finish =
      if (inclusive) List(p.Stmt.Var(result, Some(p.Expr.Index(sel(source), sel(localId), rtn)), isMutable = false))
      else {
        val first    = fresh("group_first", p.Type.Bool1)
        val previous = fresh("group_previous", p.Type.IntU32)
        List(
          p.Stmt.Var(first, Some(p.Expr.IntrOp(p.Intr.LogicEq(sel(localId), u32(0)))), isMutable = false),
          p.Stmt.Var(previous, Some(p.Expr.IntrOp(p.Intr.Sub(sel(localId), u32(1), p.Type.IntU32))), isMutable = false),
          p.Stmt.Var(result, Some(p.Expr.Alias(zero(rtn))), isMutable = true),
          p.Stmt.Cond(
            sel(first),
            Nil,
            List(
              p.Stmt.Mut(
                sel(result),
                p.Expr.Index(sel(source), sel(previous), rtn)
              )
            )
          )
        )
      }
    (p.Expr.Alias(sel(result)), setup ::: stages ::: finish ::: List(barrier))
  }

  private def ballot(requestedMask: p.Term, predicate: p.Term, pool: ScratchPool): (p.Expr, List[p.Stmt]) = {
    val buffer    = pool.allocFor(List(p.Type.Arr(p.Type.IntU32, maxGroupSize, p.Type.Space.Local))).head
    val localId   = fresh("ballot_local_id", p.Type.IntU32)
    val localSize = fresh("ballot_local_size", p.Type.IntU32)
    val lane      = fresh("ballot_lane", p.Type.IntU32)
    val base      = fresh("ballot_base", p.Type.IntU32)
    val index     = fresh("ballot_index", p.Type.IntU32)
    val source    = fresh("ballot_source", p.Type.IntU32)
    val bit       = fresh("ballot_bit", p.Type.IntU32)
    val inGroup   = fresh("ballot_in_group", p.Type.Bool1)
    val write     = fresh("ballot_write", p.Type.Bool1)
    val writeBit  = fresh("ballot_write_bit", p.Type.IntU32)
    val bitSet    = fresh("ballot_bit_set", p.Type.Bool1)
    val one       = fresh("ballot_one", p.Type.IntU32)
    val mask      = fresh("ballot_mask", p.Type.IntU32)
    val (member, membershipStatements) = membership("ballot", sel(lane), requestedMask)
    val statements = lanePrelude(localId, localSize, lane, Some(base)) ::: membershipStatements ::: List(
      p.Stmt.Var(write, Some(p.Expr.IntrOp(p.Intr.LogicAnd(predicate, sel(member)))), isMutable = false),
      p.Stmt.Var(writeBit, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.Cond(sel(write), List(p.Stmt.Mut(sel(writeBit), p.Expr.Alias(u32(1)))), Nil),
      p.Stmt.Update(sel(buffer), sel(localId), sel(writeBit)),
      barrier,
      p.Stmt.Var(mask, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.Var(source, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.Var(bit, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.ForRange(
        index,
        u32(0),
        u32(width),
        u32(1),
        List(
          p.Stmt.Mut(sel(source), p.Expr.IntrOp(p.Intr.Add(sel(base), sel(index), p.Type.IntU32))),
          p.Stmt.Var(inGroup, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(localSize)))), isMutable = false),
          p.Stmt.Mut(sel(bit), p.Expr.Alias(u32(0))),
          p.Stmt
            .Cond(sel(inGroup), List(p.Stmt.Mut(sel(bit), p.Expr.Index(sel(buffer), sel(source), p.Type.IntU32))), Nil),
          p.Stmt.Var(bitSet, Some(p.Expr.IntrOp(p.Intr.LogicNeq(sel(bit), u32(0)))), isMutable = false),
          p.Stmt.Var(one, Some(p.Expr.IntrOp(p.Intr.BSL(u32(1), sel(index), p.Type.IntU32))), isMutable = false),
          p.Stmt.Cond(
            sel(bitSet),
            List(p.Stmt.Mut(sel(mask), p.Expr.IntrOp(p.Intr.BOr(sel(mask), sel(one), p.Type.IntU32)))),
            Nil
          )
        )
      ),
      barrier
    )
    (p.Expr.Alias(sel(mask)), statements)
  }

  private def vote(
      requestedMask: p.Term,
      predicate: p.Term,
      pool: ScratchPool,
      all: Boolean
  ): (p.Expr, List[p.Stmt]) = {
    val buffer    = pool.allocFor(List(p.Type.Arr(p.Type.IntU32, maxGroupSize, p.Type.Space.Local))).head
    val localId   = fresh("vote_local_id", p.Type.IntU32)
    val localSize = fresh("vote_local_size", p.Type.IntU32)
    val lane      = fresh("vote_lane", p.Type.IntU32)
    val base      = fresh("vote_base", p.Type.IntU32)
    val index     = fresh("vote_index", p.Type.IntU32)
    val source    = fresh("vote_source", p.Type.IntU32)
    val inGroup   = fresh("vote_in_group", p.Type.Bool1)
    val opposite  = fresh("vote_opposite", p.Type.Bool1)
    val fold      = fresh("vote_fold", p.Type.Bool1)
    val writeBit  = fresh("vote_write_bit", p.Type.IntU32)
    val bit       = fresh("vote_bit", p.Type.IntU32)
    val bitSet    = fresh("vote_bit_set", p.Type.Bool1)
    val result    = fresh("vote_result", p.Type.Bool1)
    val (member, membershipStatements) = membership("vote", sel(lane), requestedMask)
    val oppositeExpr                   = if (all) p.Expr.IntrOp(p.Intr.LogicNot(predicate)) else p.Expr.Alias(predicate)
    val statements = lanePrelude(localId, localSize, lane, Some(base)) ::: membershipStatements ::: List(
      p.Stmt.Var(opposite, Some(oppositeExpr), isMutable = false),
      p.Stmt.Var(fold, Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(member), sel(opposite)))), isMutable = false),
      p.Stmt.Var(writeBit, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.Cond(sel(fold), List(p.Stmt.Mut(sel(writeBit), p.Expr.Alias(u32(1)))), Nil),
      p.Stmt.Update(sel(buffer), sel(localId), sel(writeBit)),
      barrier,
      p.Stmt.Var(result, Some(p.Expr.Alias(p.Term.Bool1Const(all))), isMutable = true),
      p.Stmt.ForRange(
        index,
        u32(0),
        u32(width),
        u32(1),
        List(
          p.Stmt.Var(source, Some(p.Expr.IntrOp(p.Intr.Add(sel(base), sel(index), p.Type.IntU32))), isMutable = false),
          p.Stmt.Var(inGroup, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(localSize)))), isMutable = false),
          p.Stmt.Var(bit, Some(p.Expr.Alias(u32(0))), isMutable = true),
          p.Stmt
            .Cond(sel(inGroup), List(p.Stmt.Mut(sel(bit), p.Expr.Index(sel(buffer), sel(source), p.Type.IntU32))), Nil),
          p.Stmt.Var(bitSet, Some(p.Expr.IntrOp(p.Intr.LogicNeq(sel(bit), u32(0)))), isMutable = false),
          p.Stmt.Cond(sel(bitSet), List(p.Stmt.Mut(sel(result), p.Expr.Alias(p.Term.Bool1Const(!all)))), Nil)
        )
      ),
      barrier
    )
    (p.Expr.Alias(sel(result)), statements)
  }
}
