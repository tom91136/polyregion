package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// runs a barrier kernel as a per-lane state machine over regions, one call per work-group (continuation-based
// synchronisation). a region starts at the entry, after each barrier, and at the head, latch, exit and join of a
// construct holding a barrier. `pc[lane]` is the region a lane runs next; bit 31 parks the lane at a barrier and
// 0x7fffffff marks a lane that returned. a round sweeps the regions in ascending id, running each occupied region
// over its lanes, until every live lane is parked or done, then releases the parked lanes into the next round. a
// region transition keeps the lane runnable within the round and only a park or a return ends its turn, which keeps
// the count-based barrier of one thread per lane: the n-th barrier of every live lane meets in round n, whatever the
// site. ids follow the reverse post-order of the region graph without loop back edges, so one sweep runs every
// forward chain and each back edge costs one more sweep
// variables:
//   used in one region                          ->  unchanged, private to an iteration of the lane loop
//   crossing regions, immutable, uniform value  ->  one shared scalar written alike by every lane
//   crossing regions otherwise                  ->  a per-lane slot in the work-group state, one array per variable
//   address taken with a barrier in scope       ->  a per-lane slot even when used in one region
// examples:
//   a = in[lid]; barrier(); out[lid] = a           ->  r0: a[lid] = in[lid]; park r1    r1: out[lid] = a[lid]; exit
//   for i <- 0 until n { s[lid] += i; barrier() }  ->  r0: i = 0 -> r1: i < n ? park r2 : exit    r2: i += 1 -> r1
// edge cases:
//   lanes leave a loop after different trip counts  ->  they take more transitions in the round and meet at the park
//   return before a barrier                         ->  pc = exit; later barriers no longer wait for the lane
//   return inside a barrier-free loop               ->  exit flag and break chain, then exit at region level
//   whole-array assignment to a crossing array      ->  declined, the entry keeps the lane abi
private[pass] object GroupRegions {
  import GroupVocabulary.{Lanes, Lid, isBarrier, sel, u32}
  import GroupState.*

  private val Parked = Int.MinValue
  private val Done   = Int.MaxValue

  case class Lowered(
      prologue: List[p.Stmt],
      driver: List[p.Stmt],
      defs: List[p.StructDef],
      regions: Int
  )

  private enum Target {
    case To(id: Int)
    case Exit
  }

  private enum Item {
    case S(stmt: p.Stmt)
    case Jump(target: Target, park: Boolean, back: Boolean)
    case Branch(cond: p.Term, t: List[Item], f: List[Item], pos: Option[p.SourcePosition], comment: Option[String])
  }

  private case class Loop(head: Int, exit: Target, cont: Target)

  private def inc(n: p.Named): p.Stmt = p.Stmt.Mut(sel(n), intr(p.Intr.Add(sel(n), u32(1), U32)))

  private def escapes(s: p.Stmt): Boolean = s match {
    case p.Stmt.Break | p.Stmt.Cont    => true
    case p.Stmt.Cond(_, t, f)          => t.exists(escapes) || f.exists(escapes)
    case p.Stmt.Annotated(inner, _, _) => escapes(inner)
    case _                             => false
  }

  private def transforms(s: p.Stmt): Boolean = hasBarrier(s) || ReturnUnwind.containsReturn(s) || escapes(s)

  private final class Formation(fresh: Fresh) {
    val bodies = mutable.ArrayBuffer.empty[List[Item]]

    private def alloc(): Int = {
      bodies += Nil
      bodies.size - 1
    }

    private def jump(target: Target, loops: List[Loop], park: Boolean = false): Item =
      Item.Jump(
        target,
        park,
        target match {
          case Target.To(id) => loops.exists(_.head == id)
          case Target.Exit   => false
        }
      )

    private def region(rest: List[p.Stmt], fall: Option[Target], loops: List[Loop]): Target =
      if (rest.isEmpty)
        fall.getOrElse(throw IllegalStateException("GroupFission: a split construct has no continuation"))
      else {
        val id = alloc()
        bodies(id) = gen(rest, fall, loops)
        Target.To(id)
      }

    private def resume(rest: List[p.Stmt], fall: Option[Target], loops: List[Loop]): Target = (rest, fall) match {
      case (Nil, Some(Target.Exit)) =>
        val id = alloc()
        bodies(id) = List(jump(Target.Exit, loops))
        Target.To(id)
      case _ => region(rest, fall, loops)
    }

    private def innermost(loops: List[Loop]): Loop =
      loops.headOption.getOrElse(throw IllegalStateException("GroupFission: break or continue outside a loop"))

    private def onReturn(e: p.Expr): List[p.Stmt] = e match {
      case p.Expr.Alias(_) => Nil
      case other           => List(p.Stmt.Var(fresh("#gf_discard", p.Type.Unit0), Some(other)))
    }

    def entry(body: List[p.Stmt]): Int = {
      val id = alloc()
      bodies(id) = gen(body, Some(Target.Exit), Nil)
      id
    }

    // region code for `stmts` that ends in a transition to `fall`, or falls through when `fall` is empty
    private def gen(stmts: List[p.Stmt], fall: Option[Target], loops: List[Loop]): List[Item] = {
      val out  = List.newBuilder[Item]
      var rest = stmts
      var open = true
      while (open) rest match {
        case Nil =>
          out ++= fall.map(jump(_, loops))
          open = false
        case s :: tail if !transforms(s) =>
          out += Item.S(s)
          rest = tail
        case s :: tail =>
          val (items, next) = split(s, tail, fall, loops)
          out ++= items
          next match {
            case Some(more) => rest = more
            case None       => open = false
          }
      }
      out.result()
    }

    // the second result is the code that continues in the current region, if any
    private def split(
        s: p.Stmt,
        rest: List[p.Stmt],
        fall: Option[Target],
        loops: List[Loop]
    ): (List[Item], Option[List[p.Stmt]]) = s match {
      case p.Stmt.Annotated(inner, pos, comment) =>
        split(inner, rest, fall, loops) match {
          case (Item.Branch(c, t, f, None, None) :: items, next) => (Item.Branch(c, t, f, pos, comment) :: items, next)
          case other                                             => other
        }
      case p.Stmt.Var(_, Some(p.Expr.SpecOp(op)), _) if isBarrier(op) =>
        (List(jump(resume(rest, fall, loops), loops, park = true)), None)
      case p.Stmt.Return(e) => (onReturn(e).map(Item.S(_)) :+ jump(Target.Exit, loops), None)
      case p.Stmt.Break     => (List(jump(innermost(loops).exit, loops)), None)
      case p.Stmt.Cont      => (List(jump(innermost(loops).cont, loops)), None)
      case p.Stmt.Cond(cond, t, f) if hasBarrier(s) =>
        val join = region(rest, fall, loops)
        (List(Item.Branch(cond, gen(t, Some(join), loops), gen(f, Some(join), loops), None, None)), None)
      case p.Stmt.Cond(cond, t, f) =>
        (List(Item.Branch(cond, gen(t, None, loops), gen(f, None, loops), None, None)), Some(rest))
      case p.Stmt.While(cond, body) if hasBarrier(s) =>
        val exit  = region(rest, fall, loops)
        val head  = alloc()
        val inner = Loop(head, exit, Target.To(head)) :: loops
        bodies(head) = List(
          Item.Branch(cond, gen(body, Some(Target.To(head)), inner), List(jump(exit, inner)), None, None)
        )
        (List(jump(Target.To(head), loops)), None)
      case p.Stmt.ForRange(i, lb, ub, step, body) if hasBarrier(s) =>
        val exit  = region(rest, fall, loops)
        val head  = alloc()
        val latch = alloc()
        val inner = Loop(head, exit, Target.To(latch)) :: loops
        val more  = fresh("#gf_more", Bool)
        bodies(head) = List(
          Item.S(p.Stmt.Var(more, Some(intr(p.Intr.LogicLt(sel(i), ub))))),
          Item.Branch(sel(more), gen(body, Some(Target.To(latch)), inner), List(jump(exit, inner)), None, None)
        )
        bodies(latch) = List(
          Item.S(p.Stmt.Mut(sel(i), intr(p.Intr.Add(sel(i), step, i.tpe)))),
          jump(Target.To(head), inner)
        )
        (List(Item.S(p.Stmt.Var(i, Some(alias(lb)), isMutable = true)), jump(Target.To(head), loops)), None)
      case loop @ (_: p.Stmt.While | _: p.Stmt.ForRange) =>
        val flag    = fresh("#gf_ret", Bool)
        val unwound = ReturnUnwind(List(loop), sel(flag), onReturn, () => fresh("#gf_pending", Bool))
        val items = Item.S(p.Stmt.Var(flag, Some(alias(p.Term.Bool1Const(false))), isMutable = true)) ::
          unwound.map(Item.S(_)) ::: List(Item.Branch(sel(flag), List(jump(Target.Exit, loops)), Nil, None, None))
        (items, Some(rest))
      case other => throw IllegalStateException(s"GroupFission: cannot split ${other.repr}")
    }
  }

  private def forward(items: List[Item]): List[Int] = items.flatMap {
    case Item.Jump(Target.To(id), _, false) => List(id)
    case Item.Branch(t = t, f = f)          => forward(t) ::: forward(f)
    case _                                  => Nil
  }

  private def renumber(items: List[Item], ids: Map[Int, Int]): List[Item] = items.map {
    case Item.Jump(Target.To(id), park, back) => Item.Jump(Target.To(ids(id)), park, back)
    case Item.Branch(c, t, f, pos, comment)   => Item.Branch(c, renumber(t, ids), renumber(f, ids), pos, comment)
    case other                                => other
  }

  // regions in reverse post-order of the graph without back edges; unreachable regions are dropped
  private def ordered(bodies: IndexedSeq[List[Item]], entry: Int): Vector[List[Item]] = {
    val seen = mutable.Set.empty[Int]
    val post = mutable.ArrayBuffer.empty[Int]
    def visit(id: Int): Unit = if (seen.add(id)) {
      forward(bodies(id)).foreach(visit)
      post += id
    }
    visit(entry)
    val order = post.reverse.toVector
    val ids   = order.zipWithIndex.toMap
    order.map(id => renumber(bodies(id), ids))
  }

  private def statements(items: List[Item]): List[p.Stmt] = items.map {
    case Item.S(s)                  => s
    case Item.Jump(_, _, _)         => p.Stmt.Cont
    case Item.Branch(c, t, f, _, _) => p.Stmt.Cond(c, statements(t), statements(f))
  }

  private def jumps(items: List[Item]): List[Item.Jump] = items.flatMap {
    case j: Item.Jump              => List(j)
    case Item.Branch(t = t, f = f) => jumps(t) ::: jumps(f)
    case _                         => Nil
  }

  def apply(prepared: GroupVocabulary.Prepared, defs: List[p.StructDef]): Either[String, Lowered] = {
    val formation = Formation(prepared.fresh)
    val entry     = formation.entry(prepared.body)
    lower(prepared, defs, ordered(formation.bodies.toVector, entry), pinned(prepared.body))
  }

  private def lower(
      prepared: GroupVocabulary.Prepared,
      defs: List[p.StructDef],
      regions: Vector[List[Item]],
      pinned: Set[String]
  ): Either[String, Lowered] =
    classify(prepared, defs, regions.map(statements), pinned).map(emit(prepared, regions, _))

  private def emit(prepared: GroupVocabulary.Prepared, regions: Vector[List[Item]], classes: Classes): Lowered = {
    val fresh  = prepared.fresh
    val access = Access(prepared, classes)

    val k     = regions.size
    val pc    = fresh("#gf_pc", p.Type.Ptr(U32, G))
    val cnt   = fresh("#gf_cnt", p.Type.Arr(U32, k, p.Type.Space.Private))
    val ready = fresh("#gf_runnable", U32)
    val live  = fresh("#gf_live", U32)

    def block(id: Int, items: List[Item]): List[p.Stmt] = {
      val leaves = jumps(items)
      val moves  = leaves.collect { case Item.Jump(Target.To(t), false, _) => t }.distinct.sorted
      val counts = moves.map(t => t -> fresh("#gf_moved", U32)).toMap
      val exited = Option.when(leaves.exists(_.target == Target.Exit))(fresh("#gf_exited", U32))
      val scope  = access.scope(statements(items))
      def code(xs: List[Item]): List[p.Stmt] = xs.flatMap {
        case Item.S(s) => access.statement(s, scope)
        case Item.Jump(Target.To(t), park, _) =>
          p.Stmt.Update(sel(pc), sel(Lid), u32(if (park) t | Parked else t)) ::
            (if (park) Nil else List(inc(counts(t)))) ::: List(p.Stmt.Cont)
        case Item.Jump(Target.Exit, _, _) =>
          List(p.Stmt.Update(sel(pc), sel(Lid), u32(Done)), inc(exited.get), p.Stmt.Cont)
        case Item.Branch(c, t, f, pos, comment) =>
          val cond = p.Stmt.Cond(access.rewrite(c, scope), code(t), code(f))
          List(if (pos.isEmpty && comment.isEmpty) cond else p.Stmt.Annotated(cond, pos, comment))
      }
      val n        = fresh("#gf_n", U32)
      val occupied = fresh("#gf_occupied", Bool)
      val current  = fresh("#gf_at_pc", U32)
      val here     = fresh("#gf_here", Bool)
      val lane = p.Stmt.ForRange(
        Lid,
        u32(0),
        sel(Lanes),
        u32(1),
        List(
          p.Stmt.Var(current, Some(p.Expr.Index(sel(pc), sel(Lid), U32))),
          p.Stmt.Var(here, Some(intr(p.Intr.LogicEq(sel(current), u32(id))))),
          p.Stmt.Cond(sel(here), scope.prelude ::: access.slots(scope) ::: code(items), Nil)
        )
      )
      val settle = moves.flatMap { t =>
        val before = fresh("#gf_was", U32)
        val after  = fresh("#gf_now", U32)
        List(
          p.Stmt.Var(before, Some(p.Expr.Index(sel(cnt), u32(t), U32))),
          p.Stmt.Var(after, Some(intr(p.Intr.Add(sel(before), sel(counts(t)), U32)))),
          p.Stmt.Update(sel(cnt), u32(t), sel(after)),
          p.Stmt.Mut(sel(ready), intr(p.Intr.Add(sel(ready), sel(counts(t)), U32)))
        )
      }
      val retire = exited.map(x => p.Stmt.Mut(sel(live), intr(p.Intr.Sub(sel(live), sel(x), U32)))).toList
      val zeroed = moves.map(t => p.Stmt.Var(counts(t), Some(alias(u32(0))), isMutable = true)) :::
        exited.map(x => p.Stmt.Var(x, Some(alias(u32(0))), isMutable = true)).toList
      val run = p.Stmt.Update(sel(cnt), u32(id), u32(0)) ::
        p.Stmt.Mut(sel(ready), intr(p.Intr.Sub(sel(ready), sel(n), U32))) :: zeroed ::: lane :: settle ::: retire
      List(
        p.Stmt.Var(n, Some(p.Expr.Index(sel(cnt), u32(id), U32))),
        p.Stmt.Var(occupied, Some(intr(p.Intr.LogicNeq(sel(n), u32(0))))),
        p.Stmt.Cond(sel(occupied), run, Nil)
      )
    }

    val (stateDecls, state) = access.state(List(U32))
    val setup = access.shared ::: stateDecls ::: List(p.Stmt.Var(pc, Some(p.Expr.Cast(sel(state), pc.tpe))))

    val counter = fresh("#gf_k", U32)
    val init = List(
      p.Stmt.Var(cnt, None, isMutable = true),
      p.Stmt.ForRange(counter, u32(0), u32(k), u32(1), List(p.Stmt.Update(sel(cnt), sel(counter), u32(0)))),
      p.Stmt.Update(sel(cnt), u32(0), sel(Lanes)),
      p.Stmt.ForRange(Lid, u32(0), sel(Lanes), u32(1), List(p.Stmt.Update(sel(pc), sel(Lid), u32(0)))),
      p.Stmt.Var(ready, Some(alias(sel(Lanes))), isMutable = true),
      p.Stmt.Var(live, Some(alias(sel(Lanes))), isMutable = true)
    )

    val idle = fresh("#gf_idle", Bool)
    val sweeps = p.Stmt.While(
      p.Term.Bool1Const(true),
      p.Stmt.Var(idle, Some(intr(p.Intr.LogicEq(sel(ready), u32(0))))) ::
        p.Stmt.Cond(sel(idle), List(p.Stmt.Break), Nil) :: regions.zipWithIndex.toList.flatMap((items, id) =>
          block(id, items)
        )
    )
    val value  = fresh("#gf_resume", U32)
    val parked = fresh("#gf_parked_bit", U32)
    val waits  = fresh("#gf_waits", Bool)
    val next   = fresh("#gf_next", U32)
    val was    = fresh("#gf_was", U32)
    val now    = fresh("#gf_now", U32)
    val release = p.Stmt.ForRange(
      Lid,
      u32(0),
      sel(Lanes),
      u32(1),
      List(
        p.Stmt.Var(value, Some(p.Expr.Index(sel(pc), sel(Lid), U32))),
        p.Stmt.Var(parked, Some(intr(p.Intr.BAnd(sel(value), u32(Parked), U32)))),
        p.Stmt.Var(waits, Some(intr(p.Intr.LogicNeq(sel(parked), u32(0))))),
        p.Stmt.Cond(
          sel(waits),
          List(
            p.Stmt.Var(next, Some(intr(p.Intr.BXor(sel(value), u32(Parked), U32)))),
            p.Stmt.Update(sel(pc), sel(Lid), sel(next)),
            p.Stmt.Var(was, Some(p.Expr.Index(sel(cnt), sel(next), U32))),
            p.Stmt.Var(now, Some(intr(p.Intr.Add(sel(was), u32(1), U32)))),
            p.Stmt.Update(sel(cnt), sel(next), sel(now)),
            inc(ready)
          ),
          Nil
        )
      )
    )
    val finished = fresh("#gf_finished", Bool)
    val rounds = p.Stmt.While(
      p.Term.Bool1Const(true),
      List(
        p.Stmt.Var(finished, Some(intr(p.Intr.LogicEq(sel(live), u32(0))))),
        p.Stmt.Cond(sel(finished), List(p.Stmt.Break), Nil),
        sweeps,
        release
      )
    )
    Lowered(
      prepared.prologue,
      setup ::: init ::: List(rounds, p.Stmt.Return(alias(p.Term.Unit0Const))),
      classes.wrappers,
      k
    )
  }
}
