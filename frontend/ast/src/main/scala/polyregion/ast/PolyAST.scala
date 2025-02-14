package polyregion.ast

import scala.collection.immutable.ArraySeq
import polyregion.ast.PolyAST.Type.Space
import polyregion.ast.PolyAST.Type.Kind
import polyregion.ast.PolyAST.Function.Attr

object PolyAST {

  object Type {
    enum Space derives MsgPack.Codec { case Global, Local, Private          }
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
    case Ptr(comp: Type, length: Option[Int], space: Type.Space)                    extends Type(Type.Kind.Ref)
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

    case Unit0Const                                  extends Expr(Type.Unit0)
    case Bool1Const(value: Boolean)                  extends Expr(Type.Bool1)
    case NullPtrConst(comp: Type, space: Type.Space) extends Expr(Type.Ptr(comp, None, space))

    case SpecOp(op: Spec) extends Expr(op.tpe)
    case MathOp(op: Math) extends Expr(op.tpe)
    case IntrOp(op: Intr) extends Expr(op.tpe)

    case Select(init: List[Named], last: Named) extends Expr(last.tpe)
    case Poison(t: Type)                        extends Expr(t)

    case Cast(from: Expr, as: Type)                                         extends Expr(as)
    case Index(lhs: Expr, idx: Expr, comp: Type)                            extends Expr(comp)
    case RefTo(lhs: Expr, idx: Option[Expr], comp: Type, space: Type.Space) extends Expr(Type.Ptr(comp, None, space))
    case Alloc(comp: Type, size: Expr, space: Type.Space)                   extends Expr(Type.Ptr(comp, None, space))
    case Invoke(name: String, args: List[Expr], rtn: Type)                  extends Expr(rtn)

    case Annotated(expr: Expr, pos: Option[SourcePosition], comment: Option[String]) extends Expr(expr.tpe)
  }

  enum Stmt derives MsgPack.Codec {
    case Block(stmts: List[Stmt])
    case Comment(value: String)
    case Var(name: Named, expr: Option[Expr])
    case Mut(name: Expr, expr: Expr)
    case Update(lhs: Expr, idx: Expr, value: Expr)
    case While(tests: List[Stmt], cond: Expr, body: List[Stmt])
    case ForRange(induction: Expr.Select, lbIncl: Expr, ubExcl: Expr, step: Expr, body: List[Stmt])

    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
    case Annotated(stmt: Stmt, pos: Option[SourcePosition] = None, comment: Option[String] = None)
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
  case class Function(          //
      name: String,             //
      args: List[Arg],          //
      rtn: Type,                //
      body: List[Stmt],         //
      attrs: Set[Function.Attr] //
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

  // ==========

  extension (t: SourcePosition) {
    inline def repr: String = s"${t.file}:${t.line}${t.col.map(c => s":$c").getOrElse("")}"
  }

  extension (t: Type.Space) {
    inline def repr: String = t match {
      case Space.Global  => ""
      case Space.Local   => "^Local"
      case Space.Private => "^Private"
    }
  }

  extension (k: Type.Kind) {
    inline def repr: String = k match {
      case Kind.None       => "None"
      case Kind.Ref        => "Ref"
      case Kind.Integral   => "Integral"
      case Kind.Fractional => "Fractional"
    }
  }

  extension (t: Type) {
    inline def repr: String = t match {
      case Type.Float16 => "F16"
      case Type.Float32 => "F32"
      case Type.Float64 => "F64"

      case Type.IntU8  => "U8"
      case Type.IntU16 => "U16"
      case Type.IntU32 => "U32"
      case Type.IntU64 => "U64"

      case Type.IntS8  => "I8"
      case Type.IntS16 => "I16"
      case Type.IntS32 => "I32"
      case Type.IntS64 => "I64"

      case Type.Nothing => "Nothing"
      case Type.Unit0   => "Unit0"
      case Type.Bool1   => "Bool1"

      case Type.Struct(name) => s"$name"
      case Type.Ptr(comp, length, space) =>
        s"${comp.repr}${length.map(_.toString).map(l => s"[$l]").getOrElse("*")}${space.repr}"
      case Type.Annotated(tpe, pos, comment) =>
        s"(${tpe.repr}${pos.map(s => s"/* ${s.repr} */").getOrElse("")}${comment.map(s => s"/* $s */").getOrElse("")})"
    }
  }

  extension (n: Named) {
    inline def repr: String = s"${n.symbol}"
  }

  extension (e: Expr) {
    inline def repr: String = e match {
      case Expr.Float16Const(x) => s"f16($x)"
      case Expr.Float32Const(x) => s"f32($x)"
      case Expr.Float64Const(x) => s"f64($x)"

      case Expr.IntU8Const(x)  => s"u8($x)"
      case Expr.IntU16Const(x) => s"u16($x)"
      case Expr.IntU32Const(x) => s"u32($x)"
      case Expr.IntU64Const(x) => s"u64($x)"

      case Expr.IntS8Const(x)  => s"i8($x)"
      case Expr.IntS16Const(x) => s"i16($x)"
      case Expr.IntS32Const(x) => s"i32($x)"
      case Expr.IntS64Const(x) => s"i64($x)"

      case Expr.Unit0Const             => "unit0(())"
      case Expr.Bool1Const(x)          => s"bool1($x)"
      case Expr.NullPtrConst(x, space) => s"nullptr[${x.repr}, ${space.repr}]"

      case Expr.SpecOp(op) =>
        op match {
          case Spec.Assert             => "'assert"
          case Spec.GpuBarrierGlobal   => "'gpuBarrierGlobal"
          case Spec.GpuBarrierLocal    => "'gpuBarrierLocal"
          case Spec.GpuBarrierAll      => "'gpuBarrierAll"
          case Spec.GpuFenceGlobal     => "'gpuFenceGlobal"
          case Spec.GpuFenceLocal      => "'gpuFenceLocal"
          case Spec.GpuFenceAll        => "'gpuFenceAll"
          case Spec.GpuGlobalIdx(dim)  => s"'gpuGlobalIdx(${dim.repr})"
          case Spec.GpuGlobalSize(dim) => s"'gpuGlobalSize(${dim.repr})"
          case Spec.GpuGroupIdx(dim)   => s"'gpuGroupIdx(${dim.repr})"
          case Spec.GpuGroupSize(dim)  => s"'gpuGroupSize(${dim.repr})"
          case Spec.GpuLocalIdx(dim)   => s"'gpuLocalIdx(${dim.repr})"
          case Spec.GpuLocalSize(dim)  => s"'gpuLocalSize(${dim.repr})"
        }
      case Expr.MathOp(op) =>
        op match {
          case Math.Abs(x, tpe)      => s"'abs(${x.repr})"
          case Math.Sin(x, tpe)      => s"'sin(${x.repr})"
          case Math.Cos(x, tpe)      => s"'cos(${x.repr})"
          case Math.Tan(x, tpe)      => s"'tan(${x.repr})"
          case Math.Asin(x, tpe)     => s"'asin(${x.repr})"
          case Math.Acos(x, tpe)     => s"'acos(${x.repr})"
          case Math.Atan(x, tpe)     => s"'atan(${x.repr})"
          case Math.Sinh(x, tpe)     => s"'sinh(${x.repr})"
          case Math.Cosh(x, tpe)     => s"'cosh(${x.repr})"
          case Math.Tanh(x, tpe)     => s"'tanh(${x.repr})"
          case Math.Signum(x, tpe)   => s"'signum(${x.repr})"
          case Math.Round(x, tpe)    => s"'round(${x.repr})"
          case Math.Ceil(x, tpe)     => s"'ceil(${x.repr})"
          case Math.Floor(x, tpe)    => s"'floor(${x.repr})"
          case Math.Rint(x, tpe)     => s"'rint(${x.repr})"
          case Math.Sqrt(x, tpe)     => s"'sqrt(${x.repr})"
          case Math.Cbrt(x, tpe)     => s"'cbrt(${x.repr})"
          case Math.Exp(x, tpe)      => s"'exp(${x.repr})"
          case Math.Expm1(x, tpe)    => s"'expm1(${x.repr})"
          case Math.Log(x, tpe)      => s"'log(${x.repr})"
          case Math.Log1p(x, tpe)    => s"'log1p(${x.repr})"
          case Math.Log10(x, tpe)    => s"'log10(${x.repr})"
          case Math.Pow(x, y, tpe)   => s"'pow(${x.repr}, ${y.repr})"
          case Math.Atan2(x, y, tpe) => s"'atan2(${x.repr}, ${y.repr})"
          case Math.Hypot(x, y, tpe) => s"'hypot(${x.repr}, ${y.repr})"

        }
      case Expr.IntrOp(op) =>
        op match {
          case Intr.BNot(x, tpe) => s"('~${x.repr})"
          case Intr.LogicNot(x)  => s"('!${x.repr})"
          case Intr.Pos(x, tpe)  => s"('+${x.repr})"
          case Intr.Neg(x, tpe)  => s"('-${x.repr})"

          case Intr.Add(x, y, tpe)  => s"(${x.repr} '+ ${y.repr})"
          case Intr.Sub(x, y, tpe)  => s"(${x.repr} '- ${y.repr})"
          case Intr.Mul(x, y, tpe)  => s"(${x.repr} '* ${y.repr})"
          case Intr.Div(x, y, tpe)  => s"(${x.repr} '/ ${y.repr})"
          case Intr.Rem(x, y, tpe)  => s"(${x.repr} '% ${y.repr})"
          case Intr.Min(x, y, tpe)  => s"'min(${x.repr}, ${y.repr})"
          case Intr.Max(x, y, tpe)  => s"'max(${x.repr}, ${y.repr})"
          case Intr.BAnd(x, y, tpe) => s"(${x.repr} '& ${y.repr})"
          case Intr.BOr(x, y, tpe)  => s"(${x.repr} '| ${y.repr})"
          case Intr.BXor(x, y, tpe) => s"(${x.repr} '^ ${y.repr})"
          case Intr.BSL(x, y, tpe)  => s"(${x.repr} '<< ${y.repr})"
          case Intr.BSR(x, y, tpe)  => s"(${x.repr} '>> ${y.repr})"
          case Intr.BZSR(x, y, tpe) => s"(${x.repr} '>>> ${y.repr})"

          case Intr.LogicAnd(x, y) => s"(${x.repr} '&& ${y.repr})"
          case Intr.LogicOr(x, y)  => s"(${x.repr} '|| ${y.repr})"
          case Intr.LogicEq(x, y)  => s"(${x.repr} '== ${y.repr})"
          case Intr.LogicNeq(x, y) => s"(${x.repr} '!= ${y.repr})"
          case Intr.LogicLte(x, y) => s"(${x.repr} '<= ${y.repr})"
          case Intr.LogicGte(x, y) => s"(${x.repr} '>= ${y.repr})"
          case Intr.LogicLt(x, y)  => s"(${x.repr} '< ${y.repr})"
          case Intr.LogicGt(x, y)  => s"(${x.repr} '> ${y.repr})"
        }
      case Expr.Select(init, last) =>
        (init :+ last)
          .map(x => s"${x.symbol}: ${x.tpe.repr}")
          .reduceLeftOption((acc, x) => s"($acc).$x")
          .getOrElse("")
      case Expr.Poison(t) => s"__poison__ /* poison of type ${t.repr} */"

      case Expr.Cast(from, as)        => s"(${from.repr}).to[${as.repr}]"
      case Expr.Index(lhs, idx, comp) => s"(${lhs.repr}).index[${comp.repr}](${idx.repr})"
      case Expr.RefTo(lhs, idx, comp, space) =>
        s"(${lhs.repr}).refTo[${comp.repr}, ${space.repr}](${idx.map(_.repr).getOrElse("")})"
      case Expr.Alloc(comp, size, space) => s"alloc[${comp.repr}, ${space.repr}](${size.repr})"
      case Expr.Invoke(name, args, rtn)  => s"$name(${args.map(_.repr).mkString(", ")}): ${rtn.repr}"
      case Expr.Annotated(expr, pos, comment) =>
        s"(${expr.repr}${pos.map(s => s"/* ${s.repr} */").getOrElse("")}${comment.map(s => s"/* $s */").getOrElse("")})"

    }
  }

  extension (stmt: Stmt) {
    inline def repr: String = stmt match {
      case Stmt.Block(xs) =>
        s"${"{"}\n${xs.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
      case Stmt.Comment(value)          => s" /* $value */"
      case Stmt.Var(name, rhs)          => s"var ${name.symbol}: ${name.tpe.repr} = ${rhs.map(_.repr).getOrElse("_")}"
      case Stmt.Mut(name, expr)         => s"${name.repr} = ${expr.repr}"
      case Stmt.Update(lhs, idx, value) => s"(${lhs.repr}).update(${idx.repr}) = ${value.repr}"
      case Stmt.While(tests, cond, body) =>
        s"while(${"{"}${(tests.map(_.repr) :+ cond.repr)
            .mkString(";")}${"}"})${"{"}\n${body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
      case Stmt.ForRange(ind, lb, ub, step, body) =>
        s"for(${ind.repr} = ${lb.repr}; ${ind.repr} < ${ub.repr}; ${ind.repr} += ${step.repr})${"{"}\n${body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
      case Stmt.Break        => s"break;"
      case Stmt.Cont         => s"continue;"
      case Stmt.Return(expr) => s"return ${expr.repr}"
      case Stmt.Cond(cond, trueBr, falseBr) =>
        s"if(${cond.repr}) ${"{"}\n${trueBr.map(_.repr).mkString("\n").indent(2)}\n${"}"}${
            if (falseBr.isEmpty) ""
            else
              s" else ${"{"}\n${falseBr
                  .map(_.repr)
                  .mkString("\n")
                  .indent(2)}\n${"}"}"
          }"
      case Stmt.Annotated(stmt, pos, comment) =>
        s"${stmt.repr}${pos.map(s => s"/* ${s.repr} */").getOrElse("")}${comment.map(s => s"/* $s */").getOrElse("")}"
    }
  }

  extension (a: Arg) {
    inline def repr: String =
      s"${a.named.symbol}: ${a.named.tpe.repr}${a.pos.map(s => s"/* ${s.repr} */").getOrElse("")}"
  }

  extension (a: Function.Attr) {
    inline def repr: String = a match {
      case Attr.Internal  => "Internal"
      case Attr.Exported  => "Exported"
      case Attr.FPRelaxed => "FPRelaxed"
      case Attr.FPStrict  => "FPStrict"
      case Attr.Entry     => "Entry"
    }
  }

  extension (f: Signature) {
    inline def repr: String = s"def ${f.name}(${f.args.map(_.repr).mkString(", ")}: ${f.rtn.repr}"
  }

  extension (f: Function) {
    inline def repr: String = s"def ${f.name}(${f.args
        .map(a => s"${a.named.symbol}: ${a.named.tpe.repr}${a.pos.map(s => s"/* ${s.repr} */").getOrElse("")}")
        .mkString(", ")}): ${f.rtn.repr} /* ${f.attrs
        .map(_.repr)
        .mkString(", ")} */ ${"{"}\n${f.body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
  }

  extension (s: StructDef) {
    inline def repr: String = s"class ${s.name}(${s.members.map(m => s"${m.symbol}: ${m.tpe.repr}").mkString(", ")})"
  }

  extension (s: Program) {
    inline def repr: String = s"${s.structs.map(_.repr).mkString("\n")}\n${s.functions.map(_.repr).mkString("\n")}"
  }

  extension (l: StructLayout) {
    inline def repr: String =
      s"StructLayout[${l.name}, sizeInBytes=${l.sizeInBytes}, align=${l.alignment}]${"{"}\n${l.members
          .map(m => s"${m.name.symbol}: ${m.name.tpe.repr} (+${m.offsetInBytes},${m.sizeInBytes})")
          .mkString("\n")
          .indent(2)}\n${"}"}"
  }

}
