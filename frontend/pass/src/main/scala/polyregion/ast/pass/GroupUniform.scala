package polyregion.ast.pass

import scala.collection.mutable

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// lowers a barrier kernel whose barriers every lane reaches together to plain lane loops (loop fission): each barrier-free
// stretch runs as one lane loop and a construct that holds a barrier runs once per group, outside the lane loops.
// a construct is lowered this way only if its condition or bounds agree across lanes, so no lane can be anywhere else than
// its siblings; kernels with a lane-varying condition, return, break or continue around a barrier keep the region machine
// (GroupRegions), which carries a resume point per lane
// a value agrees across lanes if every definition of it reads only values that agree, group constants and literals,
// under conditions that agree
// examples:
//   t[lid] = a; barrier(); b = t[n - lid]       ->  for lid { t[lid] = a }  for lid { b = t[n - lid] }
//   while (s > 0) { f(lid, s); barrier(); s /= 2 }  ->  while (s[0] > 0) { for lid { f(lid, s[lid]) } for lid { s[lid] /= 2 } }
//   if (lid < 4) barrier()                       ->  declined, the condition varies
// edge cases:
//   break or continue of a construct holding a barrier  ->  declined
//   return before the end of the kernel                 ->  declined
//   a loop with lane-varying exits                      ->  nothing assigned in it agrees
private[pass] object GroupUniform {
  import GroupVocabulary.{Lanes, Lid, isBarrier, sel, u32}
  import GroupState.*

  private type Note = (Option[p.SourcePosition], Option[String])

  private enum Piece {
    case Run(stmts: List[p.Stmt])
    case If(cond: p.Term, t: List[Piece], f: List[Piece], notes: List[Note])
    case Loop(cond: p.Term, body: List[Piece], notes: List[Note])
    case Range(i: p.Named, lb: p.Term, ub: p.Term, step: p.Term, body: List[Piece], notes: List[Note])
  }

  private def escapes(s: p.Stmt): Boolean = s match {
    case p.Stmt.Break | p.Stmt.Cont    => true
    case p.Stmt.Cond(_, t, f)          => t.exists(escapes) || f.exists(escapes)
    case p.Stmt.Annotated(inner, _, _) => escapes(inner)
    case _                             => false
  }

  private def writes(i: p.Named, body: List[p.Stmt]): Boolean =
    body.exists(_.collectFirst_[p.Stmt] { case p.Stmt.Mut(p.Term.Select(root, _, _), _) if root == i => () }.isDefined)

  // symbols of the variables whose value is the same in every lane at every point
  private def agreeing(prepared: GroupVocabulary.Prepared, defs: List[p.StructDef]): p.Term => Boolean = {
    val defMap   = defs.map(d => d.name -> d).toMap
    val fix      = fixed(prepared)
    val body     = prepared.body
    val aliased  = addressed(body)
    val declared = mutable.LinkedHashMap.empty[String, p.Named]
    body.foreach(_.collectWhere[p.Stmt] {
      case p.Stmt.Var(n, _, _)            => declared.getOrElseUpdate(n.symbol, n)
      case p.Stmt.ForRange(induction = i) => declared.getOrElseUpdate(i.symbol, i)
    })
    val agree = mutable.Set.from(declared.collect { case (sym, n) if scalar(n.tpe) && !aliased(sym) => sym })

    def term(t: p.Term): Boolean = stableTerm(defMap, fix, agree)(t)
    def expr(e: p.Expr): Boolean = e match {
      case p.Expr.Alias(t)       => term(t)
      case p.Expr.IntrOp(op)     => op.terms.forall(term)
      case p.Expr.MathOp(op)     => op.terms.forall(term)
      case p.Expr.Cast(t, _)     => term(t)
      case p.Expr.BitCast(t, _)  => term(t)
      case p.Expr.SizeOf(_)      => true
      case p.Expr.OffsetOf(_, _) => true
      case p.Expr.SpecOp(
            op @ (_: p.Spec.GpuGroupIdx | _: p.Spec.GpuGroupSize | _: p.Spec.GpuLocalSize | _: p.Spec.GpuGlobalSize)
          ) =>
        op.terms.forall(term)
      case _ => false
    }
    def divergentExit(stmts: List[p.Stmt], varying: Boolean): Boolean = stmts.exists {
      case p.Stmt.Break | p.Stmt.Cont => varying
      case p.Stmt.Cond(c, t, f) =>
        val next = varying || !term(c)
        divergentExit(t, next) || divergentExit(f, next)
      case p.Stmt.Annotated(inner, _, _) => divergentExit(List(inner), varying)
      case _                             => false
    }
    var changed                 = false
    def drop(sym: String): Unit = if (agree.remove(sym)) changed = true
    def walk(stmts: List[p.Stmt], ctrl: Boolean): Unit = stmts.foreach {
      case p.Stmt.Var(n, Some(e), _) => if (!(ctrl && expr(e))) drop(n.symbol)
      case p.Stmt.Mut(p.Term.Select(root, steps, _), e) if GroupVocabulary.ownStorage(root, steps) =>
        if (!(ctrl && expr(e))) drop(root.symbol)
      case p.Stmt.Cond(c, t, f) =>
        val next = ctrl && term(c)
        walk(t, next)
        walk(f, next)
      case p.Stmt.While(c, b) => walk(b, ctrl && term(c) && !divergentExit(b, varying = false))
      case p.Stmt.ForRange(i, lb, ub, st, b) =>
        val next = ctrl && term(lb) && term(ub) && term(st) && !divergentExit(b, varying = false)
        if (!next) drop(i.symbol)
        walk(b, next)
      case p.Stmt.Annotated(inner, _, _) => walk(List(inner), ctrl)
      case _                             => ()
    }
    while ({
      changed = false
      walk(body, ctrl = true)
      changed
    }) ()
    term
  }

  private def peel(s: p.Stmt): (List[Note], p.Stmt) = s match {
    case p.Stmt.Annotated(inner, pos, comment) =>
      val (notes, base) = peel(inner)
      ((pos, comment) :: notes, base)
    case other => (Nil, other)
  }

  private def plan(stmts: List[p.Stmt], agrees: p.Term => Boolean): Option[List[Piece]] = {
    val out  = List.newBuilder[Piece]
    val run  = List.newBuilder[p.Stmt]
    var open = false
    var ok   = true
    def flush(): Unit = if (open) {
      out += Piece.Run(run.result())
      run.clear()
      open = false
    }
    def nested(body: List[p.Stmt]): List[Piece] = plan(body, agrees) match {
      case Some(pieces) => pieces
      case None =>
        ok = false
        Nil
    }
    stmts.foreach { s =>
      if (ok) {
        val (notes, base) = peel(s)
        base match {
          case p.Stmt.Var(_, Some(p.Expr.SpecOp(op)), _) if isBarrier(op) => flush()
          case _ if !hasBarrier(base) =>
            if (escapes(base) || ReturnUnwind.containsReturn(base)) ok = false
            else {
              run += s
              open = true
            }
          case p.Stmt.Cond(c, t, f) if agrees(c) =>
            flush()
            val (tp, fp) = (nested(t), nested(f))
            out += Piece.If(c, tp, fp, notes)
          case p.Stmt.While(c, b) if agrees(c) =>
            flush()
            out += Piece.Loop(c, nested(b), notes)
          case p.Stmt.ForRange(i, lb, ub, st, b) if agrees(lb) && agrees(ub) && agrees(st) && !writes(i, b) =>
            flush()
            out += Piece.Range(i, lb, ub, st, nested(b), notes)
          case _ => ok = false
        }
      }
    }
    flush()
    Option.when(ok)(out.result())
  }

  private def units(pieces: List[Piece]): List[List[p.Stmt]] = pieces.flatMap {
    case Piece.Run(stmts) => List(stmts)
    case Piece.If(c, t, f, _) =>
      List(p.Stmt.Cond(c, Nil, Nil)) :: units(t) ::: units(f)
    case Piece.Loop(c, body, _) => List(p.Stmt.While(c, Nil)) :: units(body)
    case Piece.Range(i, lb, ub, st, body, _) =>
      List(p.Stmt.ForRange(i, lb, ub, st, Nil)) :: units(body)
  }

  private def runs(pieces: List[Piece]): Int = pieces.map {
    case Piece.Run(_)           => 1
    case Piece.If(_, t, f, _)   => runs(t) + runs(f)
    case Piece.Loop(_, body, _) => runs(body)
    case Piece.Range(body = b)  => runs(b)
  }.sum

  def apply(prepared: GroupVocabulary.Prepared, defs: List[p.StructDef]): Either[String, GroupRegions.Lowered] = {
    val body = prepared.body match {
      case init :+ p.Stmt.Return(p.Expr.Alias(_)) => Right(init)
      case _                                      => Left("the kernel does not end in a plain return")
    }
    for {
      stmts   <- body
      pieces  <- plan(stmts, agreeing(prepared, defs)).toRight("a barrier sits under a lane-varying construct")
      classes <- classify(prepared, defs, units(pieces).toVector, pinned(prepared.body))
    } yield {
      val access = Access(prepared, classes)
      def annotate(s: p.Stmt, notes: List[Note]): p.Stmt = notes.foldRight(s) { case ((pos, comment), inner) =>
        p.Stmt.Annotated(inner, pos, comment)
      }
      def emit(pieces: List[Piece]): List[p.Stmt] = pieces.map {
        case Piece.Run(stmts) =>
          val scope = access.scope(stmts)
          p.Stmt.ForRange(
            Lid,
            u32(0),
            sel(Lanes),
            u32(1),
            scope.prelude ::: access.slots(scope) ::: stmts.flatMap(access.statement(_, scope))
          )
        case Piece.If(c, t, f, notes) => annotate(p.Stmt.Cond(access.control(c), emit(t), emit(f)), notes)
        case Piece.Loop(c, b, notes)  => annotate(p.Stmt.While(access.control(c), emit(b)), notes)
        case Piece.Range(i, lb, ub, st, b, notes) =>
          annotate(p.Stmt.ForRange(i, access.control(lb), access.control(ub), access.control(st), emit(b)), notes)
      }
      val setup = access.shared ::: (if (access.perLane.isEmpty) Nil else access.state(Nil)._1)
      GroupRegions.Lowered(
        prepared.prologue,
        setup ::: emit(pieces) ::: List(p.Stmt.Return(alias(p.Term.Unit0Const))),
        classes.wrappers,
        runs(pieces)
      )
    }
  }
}
