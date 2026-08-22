package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// emulates fixed-width subgroup operations with workgroup-local scratch and barriers for targets without
// native subgroup support. subgroup size/lane become constants and local-id arithmetic; shuffles, votes and
// ballots exchange scalar or aggregate leaves through scratch sized to the configured workgroup ceiling
// examples:
//   subgroupSize()             ->  width
//   laneIdx()                  ->  localIdx(0) & (width - 1)
//   shuffleDown(x, delta)      ->  scratch[localIdx] = x; barrier; scratch[subgroupBase + lane + delta]
//   voteAny(p) / voteAll(p)    ->  one predicate slot per subgroup, reduced by its lanes
//   ballot(p)                  ->  one predicate slot per workgroup lane, packed into an i32 mask
// edge cases:
//   source outside subgroup/workgroup  ->  shuffle retains the calling lane's value
//   aggregate shuffle                  ->  one scratch buffer per scalar leaf
//   control flow                       ->  native collectives require subgroup-uniform participation; emulation requires
//                                         whole-workgroup-uniform participation because it synchronises local scratch
//   width / maxGroupSize               ->  require a power-of-two width <= 32 and a divisible ceiling
case class SubgroupLower(width: Int = 32, maxGroupSize: Int = 1024) extends ProgramPass derives PassArgCodec {
  override def phase: p.PassPhase = p.PassPhase.PostMono

  override def apply(program: p.Program, log: Log): p.Program = {
    require(
      width > 0 && width <= 32 && Integer.bitCount(width) == 1,
      s"width must be a power of two in [1, 32]: $width"
    )
    require(
      maxGroupSize >= width && maxGroupSize % width == 0,
      s"maxGroupSize must be a positive multiple of width: $maxGroupSize"
    )
    Lowering(program, width, maxGroupSize).run()
  }
}

private final class Lowering(program: p.Program, width: Int, maxGroupSize: Int) {
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
    program.copy(entry = lower(program.entry), functions = program.functions.map(lower))
  }

  private def barrier: p.Stmt =
    p.Stmt.Var(
      fresh("barrier", p.Type.Unit0),
      Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)),
      isMutable = false
    )

  private def lanePrelude(localId: p.Named, lane: p.Named, base: Option[p.Named] = None): List[p.Stmt] =
    List(
      p.Stmt.Var(localId, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(0)))), isMutable = false),
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

  private def activeGroupSize(name: String, localId: p.Term): (p.Named, List[p.Stmt]) = {
    val localSize  = fresh(s"${name}_local_size", p.Type.IntU32)
    val globalId   = fresh(s"${name}_global_id", p.Type.IntU32)
    val globalSize = fresh(s"${name}_global_size", p.Type.IntU32)
    val groupBase  = fresh(s"${name}_group_base", p.Type.IntU32)
    val remaining  = fresh(s"${name}_remaining", p.Type.IntU32)
    val activeSize = fresh(s"${name}_active_size", p.Type.IntU32)
    activeSize -> List(
      p.Stmt.Var(localSize, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(0)))), isMutable = false),
      p.Stmt.Var(globalId, Some(p.Expr.SpecOp(p.Spec.GpuGlobalIdx(u32(0)))), isMutable = false),
      p.Stmt.Var(globalSize, Some(p.Expr.SpecOp(p.Spec.GpuGlobalSize(u32(0)))), isMutable = false),
      p.Stmt.Var(groupBase, Some(p.Expr.IntrOp(p.Intr.Sub(sel(globalId), localId, p.Type.IntU32))), isMutable = false),
      p.Stmt.Var(
        remaining,
        Some(p.Expr.IntrOp(p.Intr.Sub(sel(globalSize), sel(groupBase), p.Type.IntU32))),
        isMutable = false
      ),
      p.Stmt.Var(
        activeSize,
        Some(p.Expr.IntrOp(p.Intr.Min(sel(localSize), sel(remaining), p.Type.IntU32))),
        isMutable = false
      )
    )
  }

  private def expand(op: p.Spec, leaves: Leaves, pool: ScratchPool): Option[(p.Expr, List[p.Stmt])] = op match {
    case p.Spec.GpuSubgroupSize => Some((p.Expr.Alias(u32(width)), Nil))
    case p.Spec.GpuLaneIdx =>
      val localId = fresh("local_id", p.Type.IntU32)
      Some(
        p.Expr.IntrOp(p.Intr.BAnd(sel(localId), u32(width - 1), p.Type.IntU32)) ->
          List(p.Stmt.Var(localId, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(0)))), isMutable = false))
      )
    case p.Spec.GpuShuffleDown(value, delta, clamp, mask, rtn) =>
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
    case p.Spec.GpuShuffleUp(value, delta, clamp, mask, rtn) =>
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
    case p.Spec.GpuShuffleIdx(value, sourceLane, clamp, mask, rtn) =>
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
    case p.Spec.GpuShuffleXor(value, laneMask, clamp, mask, rtn) =>
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
    case p.Spec.GpuVoteAny(mask, predicate) => Some(vote(mask, predicate, pool, all = false))
    case p.Spec.GpuVoteAll(mask, predicate) => Some(vote(mask, predicate, pool, all = true))
    case p.Spec.GpuBallot(mask, predicate)  => Some(ballot(mask, predicate, pool))
    case p.Spec.GpuSubgroupBarrier(_)       => Some((p.Expr.SpecOp(p.Spec.GpuBarrierLocal), Nil))
    case _                                  => None
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
    val (groupSize, groupSizeStatements) = activeGroupSize("shuffle", sel(localId))

    def valueField(path: List[p.PathStep], tpe: p.Type): p.Term.Select =
      p.Term.Select(valueSelect.root, valueSelect.steps ::: path, tpe)
    def resultField(path: List[p.PathStep], tpe: p.Type): p.Term.Select = p.Term.Select(result, path, tpe)

    val statements = valueBinding ::: lanePrelude(localId, lane, Some(base)) :::
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
      ) ::: callerMembership ::: sourceMembership ::: groupSizeStatements ::: List(
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
          Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(groupSize)))),
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

  private def ballot(requestedMask: p.Term, predicate: p.Term, pool: ScratchPool): (p.Expr, List[p.Stmt]) = {
    val buffer   = pool.allocFor(List(p.Type.Arr(p.Type.IntU32, maxGroupSize, p.Type.Space.Local))).head
    val localId  = fresh("ballot_local_id", p.Type.IntU32)
    val lane     = fresh("ballot_lane", p.Type.IntU32)
    val base     = fresh("ballot_base", p.Type.IntU32)
    val index    = fresh("ballot_index", p.Type.IntU32)
    val source   = fresh("ballot_source", p.Type.IntU32)
    val bit      = fresh("ballot_bit", p.Type.IntU32)
    val inGroup  = fresh("ballot_in_group", p.Type.Bool1)
    val write    = fresh("ballot_write", p.Type.Bool1)
    val writeBit = fresh("ballot_write_bit", p.Type.IntU32)
    val bitSet   = fresh("ballot_bit_set", p.Type.Bool1)
    val one      = fresh("ballot_one", p.Type.IntU32)
    val mask     = fresh("ballot_mask", p.Type.IntU32)
    val (member, membershipStatements)   = membership("ballot", sel(lane), requestedMask)
    val (groupSize, groupSizeStatements) = activeGroupSize("ballot", sel(localId))
    val statements = lanePrelude(localId, lane, Some(base)) ::: membershipStatements ::: groupSizeStatements ::: List(
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
          p.Stmt.Var(inGroup, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(groupSize)))), isMutable = false),
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
    val buffer   = pool.allocFor(List(p.Type.Arr(p.Type.IntU32, maxGroupSize, p.Type.Space.Local))).head
    val localId  = fresh("vote_local_id", p.Type.IntU32)
    val lane     = fresh("vote_lane", p.Type.IntU32)
    val base     = fresh("vote_base", p.Type.IntU32)
    val index    = fresh("vote_index", p.Type.IntU32)
    val source   = fresh("vote_source", p.Type.IntU32)
    val inGroup  = fresh("vote_in_group", p.Type.Bool1)
    val opposite = fresh("vote_opposite", p.Type.Bool1)
    val fold     = fresh("vote_fold", p.Type.Bool1)
    val writeBit = fresh("vote_write_bit", p.Type.IntU32)
    val bit      = fresh("vote_bit", p.Type.IntU32)
    val bitSet   = fresh("vote_bit_set", p.Type.Bool1)
    val result   = fresh("vote_result", p.Type.Bool1)
    val (member, membershipStatements) = membership("vote", sel(lane), requestedMask)
    val oppositeExpr                   = if (all) p.Expr.IntrOp(p.Intr.LogicNot(predicate)) else p.Expr.Alias(predicate)
    val (groupSize, groupSizeStatements) = activeGroupSize("vote", sel(localId))
    val statements = lanePrelude(localId, lane, Some(base)) ::: membershipStatements ::: groupSizeStatements ::: List(
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
          p.Stmt.Var(inGroup, Some(p.Expr.IntrOp(p.Intr.LogicLt(sel(source), sel(groupSize)))), isMutable = false),
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
