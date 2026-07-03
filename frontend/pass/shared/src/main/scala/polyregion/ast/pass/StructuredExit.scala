package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ListBuffer

import polyregion.ast.{Log, PolyAST as p, *, given}

// lowers `assert(code, message)` to a structured exit: an asserting lane raises a flag, fences its
// remaining per-lane work, and drains out - but still runs every collective barrier (and the code that feeds
// it) so a divergent assert cannot deadlock the workgroup. runs post-mono after inlining but before the arena
// passes, so a runtime message's copy loop is arena-lowered with the rest; a no-op without asserts. an entry
// that asserts gains `var #asserted = false` and a leading `#error: i8*` arg (leading so the arena passes'
// view args cannot displace it) the dispatch fills with a zeroed `[code:u32 little-endian][message: NUL-
// terminated, up to AssertMessageLimit]` buffer. the flag + break + guard are pure control flow and lower
// identically on every backend
// examples:
//   assert(c, m)             ->  #asserted = true; #error[0..4) = c (le);        // then m's bytes inlined
//   while (b) { p[i]=v; a }  ->  while (b) { if (#asserted) break; p[i]=v; a }   // a barrier-free loop drains
//   p[i]=v; a; q[i]=w        ->  p[i]=v; a; if (!#asserted) { q[i]=w }           // the tail is fenced
//   a; barrier(); q          ->  a; barrier(); if (!#asserted) { q }             // the barrier stays unfenced
// edge cases:
//   value-returning entry        ->  drains to `return poison`; the host discards it once it reads the buffer
//   string-literal message       ->  inlined as byte stores (rusticl's SPIR-V loader panics on a string global)
//   assert in a barrier loop     ->  rejected: draining it would skip the barrier and hang the workgroup
object StructuredExit extends ProgramPass {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private val AssertedSym  = "#asserted"
  private val ErrorSym     = "#error"
  private val ErrorPtr     = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)
  private val CodeBytes    = 4 // the [code:u32 little-endian] prefix; matches polyrt::assertCodeBytes
  private val MessageLimit = p.Conventions.assertMessageLimit
  private val ctr          = new AtomicLong(0L)

  private val asserted = sel(p.Named(AssertedSym, p.Type.Bool1))
  private val error    = sel(p.Named(ErrorSym, ErrorPtr))

  private def fresh(tpe: p.Type): p.Named = p.Named(s"#as${ctr.incrementAndGet()}", tpe)
  private def let(tpe: p.Type, e: p.Expr, into: ListBuffer[p.Stmt]): p.Term = {
    val n = fresh(tpe); into += p.Stmt.Var(n, Some(e), isMutable = false); sel(n)
  }

  private def isAssert(s: p.Stmt): Boolean = s match {
    case p.Stmt.Var(_, Some(p.Expr.SpecOp(_: p.Spec.Assert)), _) => true
    case _                                                       => false
  }

  private def mayAssert(s: p.Stmt): Boolean = s match {
    case _ if isAssert(s)               => true
    case p.Stmt.Cond(_, t, f)           => t.exists(mayAssert) || f.exists(mayAssert)
    case p.Stmt.While(_, b)             => b.exists(mayAssert)
    case p.Stmt.ForRange(_, _, _, _, b) => b.exists(mayAssert)
    case p.Stmt.Annotated(inner, _, _)  => mayAssert(inner)
    case _                              => false
  }

  private def isBarrier(s: p.Stmt): Boolean = s match {
    case p.Stmt.Var(
          _,
          Some(p.Expr.SpecOp(p.Spec.GpuBarrierGlobal | p.Spec.GpuBarrierLocal | p.Spec.GpuBarrierAll)),
          _
        ) =>
      true
    case _ => false
  }

  private def containsBarrier(s: p.Stmt): Boolean = s match {
    case _ if isBarrier(s)              => true
    case p.Stmt.Cond(_, t, f)           => t.exists(containsBarrier) || f.exists(containsBarrier)
    case p.Stmt.While(_, b)             => b.exists(containsBarrier)
    case p.Stmt.ForRange(_, _, _, _, b) => b.exists(containsBarrier)
    case p.Stmt.Annotated(inner, _, _)  => containsBarrier(inner)
    case _                              => false
  }

  private def barrierBoundAssertLoop(s: p.Stmt): Boolean = s match {
    case p.Stmt.While(_, b) => (b.exists(mayAssert) && b.exists(containsBarrier)) || b.exists(barrierBoundAssertLoop)
    case p.Stmt.ForRange(_, _, _, _, b) =>
      (b.exists(mayAssert) && b.exists(containsBarrier)) || b.exists(barrierBoundAssertLoop)
    case p.Stmt.Cond(_, t, f)          => t.exists(barrierBoundAssertLoop) || f.exists(barrierBoundAssertLoop)
    case p.Stmt.Annotated(inner, _, _) => barrierBoundAssertLoop(inner)
    case _                             => false
  }

  private def lowerAssert(code: p.Term, message: p.Term): List[p.Stmt] = {
    val out = ListBuffer[p.Stmt](p.Stmt.Mut(asserted, p.Expr.Alias(p.Term.Bool1Const(true))))
    (0 until CodeBytes).foreach { k =>
      val shifted = let(p.Type.IntU32, p.Expr.IntrOp(p.Intr.BSR(code, p.Term.IntU32Const(8 * k), p.Type.IntU32)), out)
      val byte    = let(p.Type.IntS8, p.Expr.Cast(shifted, p.Type.IntS8), out)
      out += p.Stmt.Update(error, p.Term.IntU32Const(k), byte)
    }
    message match {
      case p.Term.StringConst(s) =>
        s.take(MessageLimit).zipWithIndex.foreach { (c, k) =>
          out += p.Stmt.Update(error, p.Term.IntU32Const(CodeBytes + k), p.Term.IntS8Const(c.toByte))
        }
      case _ =>
        val msg   = let(message.tpe, p.Expr.Alias(message), out)
        val i     = fresh(p.Type.IntU32)
        val body  = ListBuffer.empty[p.Stmt]
        val ch    = let(p.Type.IntS8, p.Expr.Index(msg, sel(i), p.Type.IntS8), body)
        val atNul = let(p.Type.Bool1, p.Expr.IntrOp(p.Intr.LogicEq(ch, p.Term.IntS8Const(0))), body)
        body += p.Stmt.Cond(atNul, List(p.Stmt.Break), Nil)
        val off =
          let(p.Type.IntU32, p.Expr.IntrOp(p.Intr.Add(p.Term.IntU32Const(CodeBytes), sel(i), p.Type.IntU32)), body)
        body += p.Stmt.Update(error, off, ch)
        out += p.Stmt.ForRange(
          i,
          p.Term.IntU32Const(0),
          p.Term.IntU32Const(MessageLimit),
          p.Term.IntU32Const(1),
          body.toList
        )
    }
    out.toList
  }

  private def lower(s: p.Stmt): List[p.Stmt] = s match {
    case p.Stmt.Var(_, Some(p.Expr.SpecOp(p.Spec.Assert(code, message))), _) => lowerAssert(code, message)
    case p.Stmt.While(c, b)                                                  => List(p.Stmt.While(c, drainTop(b)))
    case p.Stmt.ForRange(i, lb, ub, st, b) => List(p.Stmt.ForRange(i, lb, ub, st, drainTop(b)))
    case p.Stmt.Cond(c, t, f)              => List(p.Stmt.Cond(c, guard(t), guard(f)))
    case p.Stmt.Annotated(inner, pos, c)   => lower(inner).map(p.Stmt.Annotated(_, pos, c))
    case other                             => List(other)
  }

  private def drainTop(body: List[p.Stmt]): List[p.Stmt] = {
    val lowered = guard(body)
    if (body.exists(mayAssert)) p.Stmt.Cond(asserted, List(p.Stmt.Break), Nil) :: lowered else lowered
  }

  private def guard(stmts: List[p.Stmt]): List[p.Stmt] = stmts match {
    case Nil => Nil
    case s :: rest =>
      val head = lower(s)
      if (mayAssert(s)) head ::: fence(rest)
      else head ::: guard(rest)
  }

  private def fence(stmts: List[p.Stmt]): List[p.Stmt] = {
    val (free, rest) = stmts.span(s => !containsBarrier(s))
    rest match {
      case Nil             => if (free.isEmpty) Nil else List(p.Stmt.Cond(asserted, Nil, guard(free)))
      case barrier :: tail => guard(free) ::: lower(barrier) ::: fence(tail)
    }
  }

  override def apply(program: p.Program, log: Log): p.Program =
    if (!program.entry.body.exists(mayAssert)) program
    else if (program.entry.body.exists(barrierBoundAssertLoop))
      throw RuntimeException(
        "assert inside a loop that also carries a collective barrier is unsupported: draining the loop would " +
          "skip the barrier and deadlock the workgroup"
      )
    else {
      val e = program.entry
      val flagDecl =
        p.Stmt.Var(p.Named(AssertedSym, p.Type.Bool1), Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
      val errorArg = p.Arg(p.Named(ErrorSym, ErrorPtr))
      val sentinel = if (e.rtn == p.Type.Unit0) p.Term.Unit0Const else p.Term.Poison(e.rtn)
      val exit     = p.Stmt.Return(p.Expr.Alias(sentinel))
      log.info(s"${e.signatureRepr}: lowered asserts to a structured drain + error buffer")
      program.copy(entry = e.copy(args = errorArg +: e.args, body = (flagDecl :: guard(e.body)) :+ exit))
    }

}
