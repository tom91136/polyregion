package polyregion.ast

import scala.collection.immutable.ArraySeq

object PolyAST {

  object Type {
    enum Space derives MsgPack.Codec { case Global, Local                   }
    enum Kind derives MsgPack.Codec  { case None, Ref, Integral, Fractional }
  }

  case class SourcePosition(file: String, line: Int, col: Option[Int]) derives MsgPack.Codec
  case class Named(symbol: String, tpe: Type) derives MsgPack.Codec

  enum Type(val kind: Type.Kind) derives MsgPack.Codec {
    case Float16 extends Type(Type.Kind.Fractional)
    case Float32 extends Type(Type.Kind.Fractional)
    case Float64 extends Type(Type.Kind.Fractional)

    case IntU8  extends Type(Type.Kind.Integral)
    case IntU16 extends Type(Type.Kind.Integral)
    case IntU32 extends Type(Type.Kind.Integral)
    case IntU64 extends Type(Type.Kind.Integral)

    case IntS8  extends Type(Type.Kind.Integral)
    case IntS16 extends Type(Type.Kind.Integral)
    case IntS32 extends Type(Type.Kind.Integral)
    case IntS64 extends Type(Type.Kind.Integral)

    case Nothing extends Type(Type.Kind.None)
    case Unit0   extends Type(Type.Kind.None)
    case Bool1   extends Type(Type.Kind.Integral)

    case Struct(name: String)                                                       extends Type(Type.Kind.Ref)
    case Ptr(component: Type, length: Option[Int], space: Type.Space)               extends Type(Type.Kind.Ref)
    case Annotated(tpe: Type, pos: Option[SourcePosition], comment: Option[String]) extends Type(tpe.kind)
  }

  enum Expr(val tpe: Type) derives MsgPack.Codec {
    case Float16Const(value: Float)  extends Expr(Type.Float16)
    case Float32Const(value: Float)  extends Expr(Type.Float32)
    case Float64Const(value: Double) extends Expr(Type.Float64)

    case IntU8Const(value: Byte)  extends Expr(Type.IntU8)
    case IntU16Const(value: Char) extends Expr(Type.IntU16)
    case IntU32Const(value: Int)  extends Expr(Type.IntU32)
    case IntU64Const(value: Long) extends Expr(Type.IntU64)

    case IntS8Const(value: Byte)   extends Expr(Type.IntS8)
    case IntS16Const(value: Short) extends Expr(Type.IntS16)
    case IntS32Const(value: Int)   extends Expr(Type.IntS32)
    case IntS64Const(value: Long)  extends Expr(Type.IntS64)

    case Unit0Const                 extends Expr(Type.Unit0)
    case Bool1Const(value: Boolean) extends Expr(Type.Bool1)

    case SpecOp(op: Spec) extends Expr(op.tpe)
    case MathOp(op: Math) extends Expr(op.tpe)
    case IntrOp(op: Intr) extends Expr(op.tpe)

    case Select(init: List[Named], last: Named) extends Expr(last.tpe)
    case Poison(t: Type)                        extends Expr(t)

    case Cast(from: Expr, as: Type)                           extends Expr(as)
    case Index(lhs: Expr, idx: Expr, component: Type)         extends Expr(component)
    case RefTo(lhs: Expr, idx: Option[Expr], component: Type) extends Expr(Type.Ptr(component, None, Type.Space.Global))
    case Alloc(component: Type, size: Expr)                   extends Expr(Type.Ptr(component, None, Type.Space.Global))
    case Invoke(name: String, args: List[Expr], rtn: Type)    extends Expr(rtn)

    case Annotated(expr: Expr, pos: Option[SourcePosition], comment: Option[String]) extends Expr(expr.tpe)
  }

  enum Stmt derives MsgPack.Codec {
    case Block(stmts: List[Stmt])
    case Comment(value: String)
    case Var(name: Named, expr: Option[Expr])
    case Mut(name: Expr, expr: Expr)
    case Update(lhs: Expr, idx: Expr, value: Expr)
    case While(tests: List[Stmt], cond: Expr, body: List[Stmt])
    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
    case Annotated(expr: Stmt, pos: Option[SourcePosition] = None, comment: Option[String] = None)
  }

  case class Overload(args: List[Type], rtn: Type) derives MsgPack.Codec

  object Spec {
    inline def GpuIndex    = List(Overload(List(Type.IntU32), Type.IntU32))
    inline def NullaryUnit = List(Overload(List[Type](), Type.Unit0))
  }
  enum Spec(val overloads: List[Overload], val exprs: List[Expr], val tpe: Type) derives MsgPack.Codec {
    // nullary misc
    case Assert extends Spec(Spec.NullaryUnit, List[Expr](), Type.Nothing)
    // nullary GPU control
    case GpuBarrierGlobal extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    case GpuBarrierLocal  extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    case GpuBarrierAll    extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    case GpuFenceGlobal   extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    case GpuFenceLocal    extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    case GpuFenceAll      extends Spec(Spec.NullaryUnit, List[Expr](), Type.Unit0)
    //  unary GPU indexing
    case GpuGlobalIdx(dim: Expr)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGlobalSize(dim: Expr) extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGroupIdx(dim: Expr)   extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGroupSize(dim: Expr)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuLocalIdx(dim: Expr)   extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuLocalSize(dim: Expr)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
  }

  object Intr {
    inline def UnaryUniformNumeric = List(
      Overload(List(Type.Float16), Type.Float16),
      Overload(List(Type.Float32), Type.Float32),
      Overload(List(Type.Float64), Type.Float64),
      Overload(List(Type.IntU8), Type.IntU8),
      Overload(List(Type.IntU16), Type.IntU16),
      Overload(List(Type.IntU32), Type.IntU32),
      Overload(List(Type.IntU64), Type.IntU64),
      Overload(List(Type.IntS8), Type.IntS8),
      Overload(List(Type.IntS16), Type.IntS16),
      Overload(List(Type.IntS32), Type.IntS32),
      Overload(List(Type.IntS64), Type.IntS64)
    )

    inline def BinaryUniformNumeric = List(
      Overload(List(Type.Float16, Type.Float16), Type.Float16),
      Overload(List(Type.Float32, Type.Float32), Type.Float32),
      Overload(List(Type.Float64, Type.Float64), Type.Float64),
      Overload(List(Type.IntU8, Type.IntU8), Type.IntU8),
      Overload(List(Type.IntU16, Type.IntU16), Type.IntU16),
      Overload(List(Type.IntU32, Type.IntU32), Type.IntU32),
      Overload(List(Type.IntU64, Type.IntU64), Type.IntU64),
      Overload(List(Type.IntS8, Type.IntS8), Type.IntS8),
      Overload(List(Type.IntS16, Type.IntS16), Type.IntS16),
      Overload(List(Type.IntS32, Type.IntS32), Type.IntS32),
      Overload(List(Type.IntS64, Type.IntS64), Type.IntS64)
    )

    inline def BinaryUniformIntegral = List(
      Overload(List(Type.IntU8, Type.IntU8), Type.IntU8),
      Overload(List(Type.IntU16, Type.IntU16), Type.IntU16),
      Overload(List(Type.IntU32, Type.IntU32), Type.IntU32),
      Overload(List(Type.IntU64, Type.IntU64), Type.IntU64),
      Overload(List(Type.IntS8, Type.IntS8), Type.IntS8),
      Overload(List(Type.IntS16, Type.IntS16), Type.IntS16),
      Overload(List(Type.IntS32, Type.IntS32), Type.IntS32),
      Overload(List(Type.IntS64, Type.IntS64), Type.IntS64)
    )

    inline def UnaryUniformIntegral = List(
      Overload(List(Type.IntU8), Type.IntU8),
      Overload(List(Type.IntU16), Type.IntU16),
      Overload(List(Type.IntU32), Type.IntU32),
      Overload(List(Type.IntU64), Type.IntU64),
      Overload(List(Type.IntS8), Type.IntS8),
      Overload(List(Type.IntS16), Type.IntS16),
      Overload(List(Type.IntS32), Type.IntS32),
      Overload(List(Type.IntS64), Type.IntS64)
    )

    inline def BinaryUniformLogic = List(
      Overload(List(Type.Float16, Type.Float16), Type.Bool1),
      Overload(List(Type.Float32, Type.Float32), Type.Bool1),
      Overload(List(Type.Float64, Type.Float64), Type.Bool1),
      Overload(List(Type.IntU8, Type.IntU8), Type.Bool1),
      Overload(List(Type.IntU16, Type.IntU16), Type.Bool1),
      Overload(List(Type.IntU32, Type.IntU32), Type.Bool1),
      Overload(List(Type.IntU64, Type.IntU64), Type.Bool1),
      Overload(List(Type.IntS8, Type.IntS8), Type.Bool1),
      Overload(List(Type.IntS16, Type.IntS16), Type.Bool1),
      Overload(List(Type.IntS32, Type.IntS32), Type.Bool1),
      Overload(List(Type.IntS64, Type.IntS64), Type.Bool1)
    )
    inline def UnaryUniformSIntegral = List(
      Overload(List(Type.IntS8), Type.IntS8),
      Overload(List(Type.IntS16), Type.IntS16),
      Overload(List(Type.IntS32), Type.IntS32),
      Overload(List(Type.IntS64), Type.IntS64)
    )

    inline def BinaryUniformBool = List(Overload(List(Type.Bool1, Type.Bool1), Type.Bool1))
    inline def GpuIndex          = List(Overload(List(Type.IntU32), Type.IntU32))
    inline def NullaryUnit       = List(Overload(List[Type](), Type.Unit0))
  }
  enum Intr(val overloads: List[Overload], val exprs: List[Expr], val tpe: Type) derives MsgPack.Codec {
    // unary
    case BNot(x: Expr, rtn: Type) extends Intr(Intr.UnaryUniformIntegral, List(x), rtn)
    case LogicNot(x: Expr)        extends Intr(Intr.BinaryUniformBool, List(x), Type.Bool1)
    case Pos(x: Expr, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)
    case Neg(x: Expr, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)
    // binary math ops
    case Add(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Sub(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Mul(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Div(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Rem(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Min(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Max(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    // binary bit manip.
    case BAnd(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BOr(x: Expr, y: Expr, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BXor(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSL(x: Expr, y: Expr, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSR(x: Expr, y: Expr, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BZSR(x: Expr, y: Expr, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    // binary logic
    case LogicAnd(x: Expr, y: Expr) extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicOr(x: Expr, y: Expr)  extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicEq(x: Expr, y: Expr)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicNeq(x: Expr, y: Expr) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLte(x: Expr, y: Expr) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGte(x: Expr, y: Expr) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLt(x: Expr, y: Expr)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGt(x: Expr, y: Expr)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
  }

  object Math {
    inline def UnaryUniformFractional = List(
      Overload(List(Type.Float16), Type.Float16),
      Overload(List(Type.Float32), Type.Float32),
      Overload(List(Type.Float64), Type.Float64)
    )
    inline def BinaryUniformFractional = List(
      Overload(List(Type.Float16, Type.Float16), Type.Float16),
      Overload(List(Type.Float32, Type.Float32), Type.Float32),
      Overload(List(Type.Float64, Type.Float64), Type.Float64)
    )
  }
  enum Math(val overloads: List[Overload], val exprs: List[Expr], val tpe: Type) derives MsgPack.Codec {
    // --- unary ---
    case Abs(x: Expr, rtn: Type)    extends Math(Intr.BinaryUniformNumeric, List(x), rtn)
    case Sin(x: Expr, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cos(x: Expr, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tan(x: Expr, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Asin(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Acos(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Atan(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sinh(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cosh(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tanh(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Signum(x: Expr, rtn: Type) extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Round(x: Expr, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Ceil(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Floor(x: Expr, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Rint(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sqrt(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cbrt(x: Expr, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Exp(x: Expr, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Expm1(x: Expr, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log(x: Expr, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log1p(x: Expr, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log10(x: Expr, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    // --- binary ---
    case Pow(x: Expr, y: Expr, rtn: Type)   extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Atan2(x: Expr, y: Expr, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Hypot(x: Expr, y: Expr, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
  }

  case class Signature(name: String, args: List[Type], rtn: Type) derives MsgPack.Codec
  case class Arg(named: Named, pos: Option[SourcePosition] = None) derives MsgPack.Codec
  case class StructDef(name: String, members: List[Named]) derives MsgPack.Codec

  object Function {
    enum Attr derives MsgPack.Codec { case Internal, Exported, FPRelaxed, FPStrict, Entry }
  }
  case class Function(           //
      name: String,              //
      args: List[Arg],           //
      rtn: Type,                 //
      body: List[Stmt],          //
      attrs: List[Function.Attr] //
  ) derives MsgPack.Codec

  case class Program(structs: List[StructDef], functions: List[Function]) derives MsgPack.Codec

  case class StructLayoutMember(name: Named, offsetInBytes: Long, sizeInBytes: Long) derives MsgPack.Codec
  case class StructLayout( //
      name: String,
      sizeInBytes: Long,
      alignment: Long,
      members: List[StructLayoutMember]
  ) derives MsgPack.Codec

  case class CompileEvent( //
      epochMillis: Long,
      elapsedNanos: Long,
      name: String,
      data: String
  ) derives MsgPack.Codec

  case class CompileResult(
      binary: Option[ArraySeq[Byte]],
      features: List[String],
      events: List[CompileEvent],
      layouts: List[StructLayout],
      messages: String
  ) derives MsgPack.Codec

}
