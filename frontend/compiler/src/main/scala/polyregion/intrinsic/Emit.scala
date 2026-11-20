package polyregion.intrinsic

import polyregion.ast.{PolyAST as p, *}

import scala.collection.mutable.ListBuffer

final class Emit(val params: List[p.Term.Select]) {

  private var scopes = List(ListBuffer.empty[p.Stmt])
  private var next   = 0

  def statements: List[p.Stmt] = scopes.head.toList

  private def push(s: p.Stmt): Unit = scopes.head += s

  private def fresh(t: p.Type): p.Named = { next += 1; p.Named(s"t$next", t) }

  def select(n: p.Named): p.Term.Select = p.Term.Select(n, Nil, n.tpe)

  def bind(e: p.Expr): p.Term.Select = {
    val n = fresh(e.tpe)
    push(p.Stmt.Var(n, Some(e)))
    select(n)
  }

  def variable(e: p.Expr): p.Term.Select = {
    val n = fresh(e.tpe)
    push(p.Stmt.Var(n, Some(e), isMutable = true))
    select(n)
  }

  def set(target: p.Term.Select, e: p.Expr): Unit = push(p.Stmt.Mut(target, e))

  def branch(cond: p.Term)(onTrue: => Unit)(onFalse: => Unit): Unit = {
    def scoped(f: => Unit): List[p.Stmt] = {
      val inner = ListBuffer.empty[p.Stmt]
      scopes = inner :: scopes
      f
      scopes = scopes.tail
      inner.toList
    }
    val t = scoped(onTrue)
    val f = scoped(onFalse)
    push(p.Stmt.Cond(cond, t, f))
  }

  def when(cond: p.Term)(onTrue: => Unit): Unit = branch(cond)(onTrue)(())

  def raise(code: p.Enums.AssertCode, message: String): Unit = {
    val assertion = p.Expr.SpecOp(p.Spec.Assert(p.Term.IntU32Const(code.value), p.Term.StringConst(message)))
    push(p.Stmt.Var(fresh(p.Type.Unit0), Some(assertion)))
  }

  def real(t: p.Type, v: Double): p.Term = t match {
    case p.Type.Float32 => p.Term.Float32Const(v.toFloat)
    case p.Type.Float64 => p.Term.Float64Const(v)
    case other          => throw IllegalArgumentException(s"No real constant of ${other.repr}")
  }

  def const(t: p.Type, v: Long): p.Term = t match {
    case p.Type.IntS8  => p.Term.IntS8Const(v.toByte)
    case p.Type.IntS16 => p.Term.IntS16Const(v.toShort)
    case p.Type.IntU16 => p.Term.IntU16Const(v.toChar)
    case p.Type.IntS32 => p.Term.IntS32Const(v.toInt)
    case p.Type.IntS64 => p.Term.IntS64Const(v)
    case p.Type.IntU32 => p.Term.IntU32Const(v.toInt)
    case p.Type.IntU64 => p.Term.IntU64Const(v)
    case p.Type.Bool1  => p.Term.Bool1Const(v != 0)
    case other         => throw IllegalArgumentException(s"No constant of ${other.repr}")
  }

  private def intr(i: p.Intr): p.Term.Select = bind(p.Expr.IntrOp(i))

  def add(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Add(x, y, x.tpe))
  def sub(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Sub(x, y, x.tpe))
  def mul(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Mul(x, y, x.tpe))
  def div(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Div(x, y, x.tpe))
  def rem(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Rem(x, y, x.tpe))
  def neg(x: p.Term): p.Term.Select             = intr(p.Intr.Neg(x, x.tpe))
  def min(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Min(x, y, x.tpe))
  def max(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.Max(x, y, x.tpe))
  def and(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.BAnd(x, y, x.tpe))
  def or(x: p.Term, y: p.Term): p.Term.Select   = intr(p.Intr.BOr(x, y, x.tpe))
  def xor(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.BXor(x, y, x.tpe))
  def shl(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.BSL(x, y, x.tpe))
  def ushr(x: p.Term, y: p.Term): p.Term.Select = intr(p.Intr.BZSR(x, y, x.tpe))
  def popCount(x: p.Term): p.Term.Select        = intr(p.Intr.PopCount(x, x.tpe))

  def eq(x: p.Term, y: p.Term): p.Term.Select   = intr(p.Intr.LogicEq(x, y))
  def neq(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.LogicNeq(x, y))
  def lt(x: p.Term, y: p.Term): p.Term.Select   = intr(p.Intr.LogicLt(x, y))
  def lte(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.LogicLte(x, y))
  def gt(x: p.Term, y: p.Term): p.Term.Select   = intr(p.Intr.LogicGt(x, y))
  def gte(x: p.Term, y: p.Term): p.Term.Select  = intr(p.Intr.LogicGte(x, y))
  def both(x: p.Term, y: p.Term): p.Term.Select = intr(p.Intr.LogicAnd(x, y))
  def not(x: p.Term): p.Term.Select             = intr(p.Intr.LogicNot(x))

  def magnitudeBits(x: p.Term): p.Term.Select = {
    val wide = x.tpe == p.Type.Float64
    val bits = if (wide) p.Type.IntS64 else p.Type.IntS32
    and(bitCast(x, bits), const(bits, if (wide) Long.MaxValue else Int.MaxValue.toLong))
  }

  private def exponentMask(x: p.Term): p.Term =
    if (x.tpe == p.Type.Float64) const(p.Type.IntS64, 0x7ff0000000000000L) else const(p.Type.IntS32, 0x7f800000L)

  def isNaN(x: p.Term): p.Term.Select      = gt(magnitudeBits(x), exponentMask(x))
  def isInfinite(x: p.Term): p.Term.Select = eq(magnitudeBits(x), exponentMask(x))
  def isFinite(x: p.Term): p.Term.Select   = lt(magnitudeBits(x), exponentMask(x))

  def wrapNeg(x: p.Term): p.Term = {
    val u = if (x.tpe == p.Type.IntS64) p.Type.IntU64 else p.Type.IntU32
    cast(sub(const(u, 0), cast(x, u)), x.tpe)
  }

  def cast(x: p.Term, as: p.Type): p.Term =
    if (x.tpe == as) x else bind(p.Expr.Cast(x, as))
  def bitCast(x: p.Term, as: p.Type): p.Term.Select = bind(p.Expr.BitCast(x, as))
  def math(m: p.Math): p.Term.Select                = bind(p.Expr.MathOp(m))
}
