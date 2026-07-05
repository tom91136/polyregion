package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}

object Fold {

  def isConstTerm(t: p.Term): Boolean = t match {
    case _: p.Term.Float16Const | _: p.Term.Float32Const | _: p.Term.Float64Const | _: p.Term.IntU8Const |
        _: p.Term.IntU16Const | _: p.Term.IntU32Const | _: p.Term.IntU64Const | _: p.Term.IntS8Const |
        _: p.Term.IntS16Const | _: p.Term.IntS32Const | _: p.Term.IntS64Const | _: p.Term.Bool1Const |
        p.Term.Unit0Const | _: p.Term.NullPtrConst =>
      true
    case _ => false
  }

  def asLong(t: p.Term): Option[Long] = t match {
    case p.Term.IntS8Const(v)  => Some(v.toLong)
    case p.Term.IntS16Const(v) => Some(v.toLong)
    case p.Term.IntS32Const(v) => Some(v.toLong)
    case p.Term.IntS64Const(v) => Some(v)
    case p.Term.IntU8Const(v)  => Some(v.toLong & 0xffL)
    case p.Term.IntU16Const(v) => Some(v.toLong & 0xffffL)
    case p.Term.IntU32Const(v) => Some(v.toLong & 0xffffffffL)
    case p.Term.IntU64Const(v) => Some(v)
    case _                     => None
  }

  def asDouble(t: p.Term): Option[Double] = t match {
    case p.Term.Float16Const(v) => Some(v.toDouble)
    case p.Term.Float32Const(v) => Some(v.toDouble)
    case p.Term.Float64Const(v) => Some(v)
    case _                      => None
  }

  def asBool(t: p.Term): Option[Boolean] = t match {
    case p.Term.Bool1Const(v) => Some(v)
    case _                    => None
  }

  def intTerm(tpe: p.Type, v: Long): Option[p.Term] = tpe match {
    case p.Type.IntS8  => Some(p.Term.IntS8Const(v.toByte))
    case p.Type.IntS16 => Some(p.Term.IntS16Const(v.toShort))
    case p.Type.IntS32 => Some(p.Term.IntS32Const(v.toInt))
    case p.Type.IntS64 => Some(p.Term.IntS64Const(v))
    case p.Type.IntU8  => Some(p.Term.IntU8Const(v.toByte))
    case p.Type.IntU16 => Some(p.Term.IntU16Const(v.toChar))
    case p.Type.IntU32 => Some(p.Term.IntU32Const(v.toInt))
    case p.Type.IntU64 => Some(p.Term.IntU64Const(v))
    case _             => None
  }

  def floatTerm(tpe: p.Type, v: Double): Option[p.Term] = tpe match {
    case p.Type.Float16 => Some(p.Term.Float16Const(v.toFloat))
    case p.Type.Float32 => Some(p.Term.Float32Const(v.toFloat))
    case p.Type.Float64 => Some(p.Term.Float64Const(v))
    case _              => None
  }

  private def foldUnaryNumeric(x: p.Term, t: p.Type)(longF: Long => Long)(doubleF: Double => Double): Option[p.Term] =
    asLong(x).flatMap(v => intTerm(t, longF(v))).orElse(asDouble(x).flatMap(v => floatTerm(t, doubleF(v))))

  private def foldBinNumeric(x: p.Term, y: p.Term, t: p.Type)(longF: (Long, Long) => Long)(
      doubleF: (Double, Double) => Double
  ): Option[p.Term] =
    (asLong(x), asLong(y)) match {
      case (Some(a), Some(b)) if t.kind == p.Type.Kind.Integral => intTerm(t, longF(a, b))
      case _ =>
        (asDouble(x), asDouble(y)) match {
          case (Some(a), Some(b)) => floatTerm(t, doubleF(a, b))
          case _                  => None
        }
    }

  // like foldBinNumeric but refuses to fold an integral divide/remainder by zero (left to trap at runtime)
  private def foldDivLike(x: p.Term, y: p.Term, t: p.Type)(longF: (Long, Long) => Long)(
      doubleF: (Double, Double) => Double
  ): Option[p.Term] =
    (asLong(x), asLong(y)) match {
      case (Some(_), Some(0L)) if t.kind == p.Type.Kind.Integral => None
      case (Some(a), Some(b)) if t.kind == p.Type.Kind.Integral  => intTerm(t, longF(a, b))
      case _ =>
        (asDouble(x), asDouble(y)) match {
          case (Some(a), Some(b)) => floatTerm(t, doubleF(a, b))
          case _                  => None
        }
    }

  private def foldBits(x: p.Term, y: p.Term, t: p.Type)(f: (Long, Long) => Long): Option[p.Term] =
    for {
      a <- asLong(x)
      b <- asLong(y)
      r <- intTerm(t, f(a, b))
    } yield r

  private def foldCmp(x: p.Term, y: p.Term)(longF: (Long, Long) => Boolean)(
      doubleF: (Double, Double) => Boolean
  ): Option[p.Term] =
    (asLong(x), asLong(y)) match {
      case (Some(a), Some(b)) => Some(p.Term.Bool1Const(longF(a, b)))
      case _ =>
        (asDouble(x), asDouble(y)) match {
          case (Some(a), Some(b)) => Some(p.Term.Bool1Const(doubleF(a, b)))
          case _                  => None
        }
    }

  private def boolEq(x: p.Term, y: p.Term): Option[p.Term.Bool1Const] = (asBool(x), asBool(y)) match {
    case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a == b))
    case _                  => None
  }

  def tryFoldIntr(op: p.Intr): Option[p.Term] = op match {
    case p.Intr.Neg(x, t)   => foldUnaryNumeric(x, t)(_.unary_-)(-_)
    case p.Intr.Pos(x, t)   => foldUnaryNumeric(x, t)(identity)(identity)
    case p.Intr.BNot(x, t)  => asLong(x).flatMap(v => intTerm(t, ~v))
    case p.Intr.LogicNot(x) => asBool(x).map(v => p.Term.Bool1Const(!v))

    case p.Intr.Add(x, y, t) => foldBinNumeric(x, y, t)(_ + _)(_ + _)
    case p.Intr.Sub(x, y, t) => foldBinNumeric(x, y, t)(_ - _)(_ - _)
    case p.Intr.Mul(x, y, t) => foldBinNumeric(x, y, t)(_ * _)(_ * _)
    case p.Intr.Div(x, y, t) =>
      foldDivLike(x, y, t)((a, b) => if (t.isSigned) a / b else java.lang.Long.divideUnsigned(a, b))(_ / _)
    case p.Intr.Rem(x, y, t) =>
      foldDivLike(x, y, t)((a, b) => if (t.isSigned) a % b else java.lang.Long.remainderUnsigned(a, b))(_ % _)
    case p.Intr.Min(x, y, t) => foldBinNumeric(x, y, t)(math.min(_, _))(math.min(_, _))
    case p.Intr.Max(x, y, t) => foldBinNumeric(x, y, t)(math.max(_, _))(math.max(_, _))

    case p.Intr.BAnd(x, y, t) => foldBits(x, y, t)(_ & _)
    case p.Intr.BOr(x, y, t)  => foldBits(x, y, t)(_ | _)
    case p.Intr.BXor(x, y, t) => foldBits(x, y, t)(_ ^ _)
    case p.Intr.BSL(x, y, t)  => foldBits(x, y, t)(_ << _)
    case p.Intr.BSR(x, y, t)  => foldBits(x, y, t)((a, b) => if (t.isSigned) a >> b else a >>> b)
    case p.Intr.BZSR(x, y, t) => foldBits(x, y, t)(_ >>> _)

    case p.Intr.LogicAnd(x, y) =>
      (asBool(x), asBool(y)) match { case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a && b)); case _ => None }
    case p.Intr.LogicOr(x, y) =>
      (asBool(x), asBool(y)) match { case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a || b)); case _ => None }
    case p.Intr.LogicEq(x, y) => foldCmp(x, y)(_ == _)(_ == _).orElse(boolEq(x, y))
    case p.Intr.LogicNeq(x, y) =>
      foldCmp(x, y)(_ != _)(_ != _).orElse(boolEq(x, y).map(b => p.Term.Bool1Const(!b.value)))
    case p.Intr.LogicLt(x, y)  => foldCmp(x, y)(_ < _)(_ < _)
    case p.Intr.LogicLte(x, y) => foldCmp(x, y)(_ <= _)(_ <= _)
    case p.Intr.LogicGt(x, y)  => foldCmp(x, y)(_ > _)(_ > _)
    case p.Intr.LogicGte(x, y) => foldCmp(x, y)(_ >= _)(_ >= _)
  }

  def tryFoldCast(from: p.Term, as: p.Type): Option[p.Term] =
    if (from.tpe == as) Some(from)
    else
      asLong(from)
        .flatMap { v =>
          as match {
            case p.Type.Float16 => Some(p.Term.Float16Const(v.toFloat))
            case p.Type.Float32 => Some(p.Term.Float32Const(v.toFloat))
            case p.Type.Float64 => Some(p.Term.Float64Const(v.toDouble))
            case _              => intTerm(as, v)
          }
        }
        .orElse(asDouble(from).flatMap { v =>
          as match {
            case p.Type.Float16 | p.Type.Float32 | p.Type.Float64 => floatTerm(as, v)
            case _                                                => intTerm(as, v.toLong)
          }
        })
        .orElse(asBool(from).flatMap { v =>
          as match {
            case p.Type.Bool1 => Some(p.Term.Bool1Const(v))
            case _            => intTerm(as, if (v) 1L else 0L)
          }
        })

  private def isZeroI(t: p.Term): Boolean  = asLong(t).contains(0L)
  private def isOneI(t: p.Term): Boolean   = asLong(t).contains(1L)
  private def isTrueB(t: p.Term): Boolean  = t == p.Term.Bool1Const(true)
  private def isFalseB(t: p.Term): Boolean = t == p.Term.Bool1Const(false)
  private def sameSelect(x: p.Term, y: p.Term): Boolean =
    x == y && (x match { case _: p.Term.Select => true; case _ => false })

  def trySimplifyIntr(op: p.Intr): Option[p.Term] = op match {
    case p.Intr.Add(x, y, _) => if (isZeroI(y)) Some(x) else if (isZeroI(x)) Some(y) else None
    case p.Intr.Sub(x, y, t) => if (isZeroI(y)) Some(x) else if (sameSelect(x, y)) intTerm(t, 0L) else None
    case p.Intr.Mul(x, y, t) =>
      if (isZeroI(x) || isZeroI(y)) intTerm(t, 0L)
      else if (isOneI(y)) Some(x)
      else if (isOneI(x)) Some(y)
      else None
    case p.Intr.Div(x, y, _) => if (isOneI(y)) Some(x) else None
    case p.Intr.Rem(x, y, t) => if (isOneI(y)) intTerm(t, 0L) else None
    case p.Intr.BAnd(x, y, t) =>
      if (isZeroI(x) || isZeroI(y)) intTerm(t, 0L) else if (sameSelect(x, y)) Some(x) else None
    case p.Intr.BOr(x, y, _) =>
      if (isZeroI(y)) Some(x) else if (isZeroI(x)) Some(y) else if (sameSelect(x, y)) Some(x) else None
    case p.Intr.BXor(x, y, t) =>
      if (isZeroI(y)) Some(x) else if (isZeroI(x)) Some(y) else if (sameSelect(x, y)) intTerm(t, 0L) else None
    case p.Intr.BSL(x, y, _)  => if (isZeroI(y)) Some(x) else None
    case p.Intr.BSR(x, y, _)  => if (isZeroI(y)) Some(x) else None
    case p.Intr.BZSR(x, y, _) => if (isZeroI(y)) Some(x) else None
    case p.Intr.Min(x, y, _)  => if (sameSelect(x, y)) Some(x) else None
    case p.Intr.Max(x, y, _)  => if (sameSelect(x, y)) Some(x) else None

    case p.Intr.LogicAnd(x, y) =>
      if (isFalseB(x) || isFalseB(y)) Some(p.Term.Bool1Const(false))
      else if (isTrueB(x)) Some(y)
      else if (isTrueB(y)) Some(x)
      else if (sameSelect(x, y)) Some(x)
      else None
    case p.Intr.LogicOr(x, y) =>
      if (isTrueB(x) || isTrueB(y)) Some(p.Term.Bool1Const(true))
      else if (isFalseB(x)) Some(y)
      else if (isFalseB(y)) Some(x)
      else if (sameSelect(x, y)) Some(x)
      else None

    // x < x and x > x are false for every value (including NaN); the others would be wrong for NaN
    case p.Intr.LogicLt(x, y)  => if (sameSelect(x, y)) Some(p.Term.Bool1Const(false)) else None
    case p.Intr.LogicGt(x, y)  => if (sameSelect(x, y)) Some(p.Term.Bool1Const(false)) else None
    case p.Intr.LogicEq(x, y)  => if (sameSelect(x, y) && !x.tpe.isFractional) Some(p.Term.Bool1Const(true)) else None
    case p.Intr.LogicNeq(x, y) => if (sameSelect(x, y) && !x.tpe.isFractional) Some(p.Term.Bool1Const(false)) else None
    case p.Intr.LogicLte(x, y) => if (sameSelect(x, y) && !x.tpe.isFractional) Some(p.Term.Bool1Const(true)) else None
    case p.Intr.LogicGte(x, y) => if (sameSelect(x, y) && !x.tpe.isFractional) Some(p.Term.Bool1Const(true)) else None

    case _ => None
  }

}
