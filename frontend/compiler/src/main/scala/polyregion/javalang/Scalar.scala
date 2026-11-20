package polyregion.javalang

import polyregion.ast.PolyAST as p

object Scalar {

  def of[A](c: A): Option[Scalar] = (c: Any) match {
    case v: Byte   => Some(I(v.toInt))
    case v: Short  => Some(I(v.toInt))
    case v: Int    => Some(I(v))
    case v: Long   => Some(L(v))
    case v: Float  => Some(F(v))
    case v: Double => Some(D(v))
    case _         => None
  }

  def constant(tpe: p.Type): Option[Scalar => p.Term] = tpe match {
    case p.Type.Bool1   => Some(n => p.Term.Bool1Const(n.toInt != 0))
    case p.Type.IntS8   => Some(n => p.Term.IntS8Const(n.toInt.toByte))
    case p.Type.IntS16  => Some(n => p.Term.IntS16Const(n.toInt.toShort))
    case p.Type.IntU16  => Some(n => p.Term.IntU16Const(n.toInt.toChar))
    case p.Type.IntS32  => Some(n => p.Term.IntS32Const(n.toInt))
    case p.Type.IntS64  => Some(n => p.Term.IntS64Const(n.toLong))
    case p.Type.Float32 => Some(n => p.Term.Float32Const(n.toFloat))
    case p.Type.Float64 => Some(n => p.Term.Float64Const(n.toDouble))
    case _              => None
  }
}

enum Scalar {
  case I(v: Int)
  case L(v: Long)
  case F(v: Float)
  case D(v: Double)

  def toInt: Int = this match {
    case I(v) => v
    case L(v) => v.toInt
    case F(v) => v.toInt
    case D(v) => v.toInt
  }

  def toLong: Long = this match {
    case I(v) => v.toLong
    case L(v) => v
    case F(v) => v.toLong
    case D(v) => v.toLong
  }

  def toFloat: Float = this match {
    case I(v) => v.toFloat
    case L(v) => v.toFloat
    case F(v) => v
    case D(v) => v.toFloat
  }

  def toDouble: Double = this match {
    case I(v) => v.toDouble
    case L(v) => v.toDouble
    case F(v) => v.toDouble
    case D(v) => v
  }
}
