package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

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
      Some(p.Expr.SpecOp(p.Spec.GpuSubgroupBarrier(u32(-1)))),
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

  private def expand(op: p.Spec, leaves: Leaves, pool: ScratchPool): Option[(p.Expr, List[p.Stmt])] = op match {
    case p.Spec.GpuSubgroupSize => Some((p.Expr.Alias(u32(width)), Nil))
    case p.Spec.GpuLaneIdx =>
      val localId = fresh("local_id", p.Type.IntU32)
      Some(
        p.Expr.IntrOp(p.Intr.BAnd(sel(localId), u32(width - 1), p.Type.IntU32)) ->
          List(p.Stmt.Var(localId, Some(p.Expr.SpecOp(p.Spec.GpuLocalIdx(u32(0)))), isMutable = false))
      )
    case p.Spec.GpuShuffleDown(value, delta, _, _, rtn) =>
      Some(shuffle(value, rtn, leaves, pool, lane => p.Expr.IntrOp(p.Intr.Add(lane, delta, p.Type.IntU32))))
    case p.Spec.GpuShuffleUp(value, delta, _, _, rtn) =>
      Some(shuffle(value, rtn, leaves, pool, lane => p.Expr.IntrOp(p.Intr.Sub(lane, delta, p.Type.IntU32))))
    case p.Spec.GpuShuffleIdx(value, sourceLane, _, _, rtn) =>
      Some(
        shuffle(
          value,
          rtn,
          leaves,
          pool,
          _ => p.Expr.IntrOp(p.Intr.BAnd(sourceLane, u32(width - 1), p.Type.IntU32))
        )
      )
    case p.Spec.GpuShuffleXor(value, laneMask, _, _, rtn) =>
      Some(shuffle(value, rtn, leaves, pool, lane => p.Expr.IntrOp(p.Intr.BXor(lane, laneMask, p.Type.IntU32))))
    case p.Spec.GpuVoteAny(_, predicate) => Some(vote(predicate, pool, all = false))
    case p.Spec.GpuVoteAll(_, predicate) => Some(vote(predicate, pool, all = true))
    case p.Spec.GpuBallot(_, predicate)  => Some(ballot(predicate, pool))
    case _                               => None
  }

  private def shuffle(
      value: p.Term,
      rtn: p.Type,
      leaves: Leaves,
      pool: ScratchPool,
      sourceOf: p.Term => p.Expr
  ): (p.Expr, List[p.Stmt]) = {
    val (valueBinding, valueSelect) = value match {
      case select: p.Term.Select => (Nil, select)
      case other =>
        val name = fresh("value", rtn)
        (List(p.Stmt.Var(name, Some(p.Expr.Alias(other)), isMutable = false)), sel(name))
    }
    val leafList   = leaves(rtn)
    val buffers    = pool.allocFor(leafList.map((_, tpe) => p.Type.Arr(tpe, maxGroupSize, p.Type.Space.Local)))
    val fields     = leafList.zip(buffers).map { case ((path, tpe), buffer) => (path, buffer, tpe) }
    val localId    = fresh("local_id", p.Type.IntU32)
    val lane       = fresh("lane", p.Type.IntU32)
    val base       = fresh("base", p.Type.IntU32)
    val target     = fresh("target", p.Type.IntU32)
    val source     = fresh("source", p.Type.IntU32)
    val groupSize  = fresh("group_size", p.Type.IntU32)
    val inSubgroup = fresh("in_subgroup", p.Type.Bool1)
    val inGroup    = fresh("in_group", p.Type.Bool1)
    val inRange    = fresh("in_range", p.Type.Bool1)
    val result     = fresh("result", rtn)

    def valueField(path: List[p.PathStep], tpe: p.Type): p.Term.Select =
      p.Term.Select(valueSelect.root, valueSelect.steps ::: path, tpe)
    def resultField(path: List[p.PathStep], tpe: p.Type): p.Term.Select = p.Term.Select(result, path, tpe)

    val statements = valueBinding ::: lanePrelude(localId, lane, Some(base)) :::
      fields.map((path, buffer, tpe) => p.Stmt.Update(sel(buffer), sel(localId), valueField(path, tpe))) :::
      List(
        barrier,
        p.Stmt.Var(target, Some(sourceOf(sel(lane))), isMutable = false),
        p.Stmt.Var(source, Some(p.Expr.IntrOp(p.Intr.Add(sel(base), sel(target), p.Type.IntU32))), isMutable = false),
        p.Stmt.Var(groupSize, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(u32(0)))), isMutable = false),
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
        p.Stmt.Var(
          inRange,
          Some(p.Expr.IntrOp(p.Intr.LogicAnd(sel(inSubgroup), sel(inGroup)))),
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

  private def ballot(predicate: p.Term, pool: ScratchPool): (p.Expr, List[p.Stmt]) = {
    val buffer  = pool.allocFor(List(p.Type.Arr(p.Type.Bool1, maxGroupSize, p.Type.Space.Local))).head
    val localId = fresh("ballot_local_id", p.Type.IntU32)
    val lane    = fresh("ballot_lane", p.Type.IntU32)
    val base    = fresh("ballot_base", p.Type.IntU32)
    val index   = fresh("ballot_index", p.Type.IntU32)
    val source  = fresh("ballot_source", p.Type.IntU32)
    val bit     = fresh("ballot_bit", p.Type.Bool1)
    val one     = fresh("ballot_one", p.Type.IntU32)
    val mask    = fresh("ballot_mask", p.Type.IntU32)
    val statements = lanePrelude(localId, lane, Some(base)) ::: List(
      p.Stmt.Update(sel(buffer), sel(localId), predicate),
      barrier,
      p.Stmt.Var(mask, Some(p.Expr.Alias(u32(0))), isMutable = true),
      p.Stmt.ForRange(
        index,
        u32(0),
        u32(width),
        u32(1),
        List(
          p.Stmt.Var(
            source,
            Some(p.Expr.IntrOp(p.Intr.Add(sel(base), sel(index), p.Type.IntU32))),
            isMutable = false
          ),
          p.Stmt.Var(bit, Some(p.Expr.Index(sel(buffer), sel(source), p.Type.Bool1)), isMutable = false),
          p.Stmt.Var(one, Some(p.Expr.IntrOp(p.Intr.BSL(u32(1), sel(index), p.Type.IntU32))), isMutable = false),
          p.Stmt.Cond(
            sel(bit),
            List(p.Stmt.Mut(sel(mask), p.Expr.IntrOp(p.Intr.BOr(sel(mask), sel(one), p.Type.IntU32)))),
            Nil
          )
        )
      ),
      barrier
    )
    (p.Expr.Alias(sel(mask)), statements)
  }

  private def vote(predicate: p.Term, pool: ScratchPool, all: Boolean): (p.Expr, List[p.Stmt]) = {
    val groups   = maxGroupSize / width
    val buffer   = pool.allocFor(List(p.Type.Arr(p.Type.Bool1, groups, p.Type.Space.Local))).head
    val localId  = fresh("vote_local_id", p.Type.IntU32)
    val lane     = fresh("vote_lane", p.Type.IntU32)
    val group    = fresh("vote_group", p.Type.IntU32)
    val leader   = fresh("vote_leader", p.Type.Bool1)
    val fold     = fresh("vote_fold", p.Type.Bool1)
    val result   = fresh("vote_result", p.Type.Bool1)
    val foldExpr = if (all) p.Expr.IntrOp(p.Intr.LogicNot(predicate)) else p.Expr.Alias(predicate)
    val statements = lanePrelude(localId, lane) ::: List(
      p.Stmt.Var(
        group,
        Some(p.Expr.IntrOp(p.Intr.BSR(sel(localId), u32(Integer.numberOfTrailingZeros(width)), p.Type.IntU32))),
        isMutable = false
      ),
      p.Stmt.Var(leader, Some(p.Expr.IntrOp(p.Intr.LogicEq(sel(lane), u32(0)))), isMutable = false),
      p.Stmt.Cond(sel(leader), List(p.Stmt.Update(sel(buffer), sel(group), p.Term.Bool1Const(all))), Nil),
      barrier,
      p.Stmt.Var(fold, Some(foldExpr), isMutable = false),
      p.Stmt.Cond(sel(fold), List(p.Stmt.Update(sel(buffer), sel(group), p.Term.Bool1Const(!all))), Nil),
      barrier,
      p.Stmt.Var(result, Some(p.Expr.Index(sel(buffer), sel(group), p.Type.Bool1)), isMutable = false),
      barrier
    )
    (p.Expr.Alias(sel(result)), statements)
  }
}
