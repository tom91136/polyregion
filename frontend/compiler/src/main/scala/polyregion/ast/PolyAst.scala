package polyregion.ast

object PolyAst {

  case class Sym(fqn: List[String]) derives MsgPack.Codec {
    def repr: String             = fqn.mkString(".")
    infix def :+(s: String): Sym = Sym(fqn :+ s)
    infix def ~(s: Sym): Sym     = Sym(fqn ++ s.fqn)
    def last: String             = fqn.last
  }
  object Sym {
    def apply(raw: String): Sym = {
      require(raw.trim.nonEmpty)
      // normalise dollar
      Sym(raw.trim.split('.').toList)
    }

    def unapply(xs: List[String]): Option[(Sym, String)] =
      xs.lastOption.map(x => Sym(xs.init) -> x)

  }

  enum TypeKind derives MsgPack.Codec {
    case None, Ref, Integral, Fractional /*Add Erased*/
  }

  enum Type(val kind: TypeKind) derives MsgPack.Codec {
    // case Float  extends Type(TypeKind.Fractional)
    // case Double extends Type(TypeKind.Fractional)

    // case Bool  extends Type(TypeKind.Integral)
    // case Byte  extends Type(TypeKind.Integral)
    // case Char  extends Type(TypeKind.Integral)
    // case Short extends Type(TypeKind.Integral)
    // case Int   extends Type(TypeKind.Integral)
    // case Long  extends Type(TypeKind.Integral)

    case Float16 extends Type(TypeKind.Fractional)
    case Float32 extends Type(TypeKind.Fractional)
    case Float64 extends Type(TypeKind.Fractional)

    case IntU8  extends Type(TypeKind.Integral)
    case IntU16 extends Type(TypeKind.Integral)
    case IntU32 extends Type(TypeKind.Integral)
    case IntU64 extends Type(TypeKind.Integral)
    case IntS8  extends Type(TypeKind.Integral)
    case IntS16 extends Type(TypeKind.Integral)
    case IntS32 extends Type(TypeKind.Integral)
    case IntS64 extends Type(TypeKind.Integral)

    case Nothing extends Type(TypeKind.None)
    case Unit0   extends Type(TypeKind.None)
    case Bool1   extends Type(TypeKind.Integral)

    // specialisations
    case Struct(name: Sym, tpeVars: List[String], args: List[Type], parents: List[Sym]) extends Type(TypeKind.Ref)
    case Array(component: Type, space: Type.Space)                                      extends Type(TypeKind.Ref)

    //
    case Var(name: String)                                        extends Type(TypeKind.None)
    case Exec(tpeVars: List[String], args: List[Type], rtn: Type) extends Type(TypeKind.None)

    // def Exec(tpeVars: List[String], args: List[Type], rtn: Type) = Struct(Sym("" :: "Poly":: Nil), tpeVars, args, rtn, Nil)
  }
  object Type {
    enum Space derives MsgPack.Codec { case Global, Local }
  }

  //  val Intersection = Type.Struct(Sym("__intersection"), Nil, ???, Nil)

  case class Named(symbol: String, tpe: Type) derives MsgPack.Codec

  enum Term(val tpe: Type) derives MsgPack.Codec {
    case Select(init: List[Named], last: Named) extends Term(last.tpe)
    case Poison(t: Type)                        extends Term(t)

    case Float16Const(value: Float)  extends Term(Type.Float16)
    case Float32Const(value: Float)  extends Term(Type.Float32)
    case Float64Const(value: Double) extends Term(Type.Float64)

    case IntU8Const(value: Byte)   extends Term(Type.IntU8)
    case IntU16Const(value: Char)  extends Term(Type.IntU16)
    case IntU32Const(value: Int)   extends Term(Type.IntU32)
    case IntU64Const(value: Long)  extends Term(Type.IntU64)
    case IntS8Const(value: Byte)   extends Term(Type.IntS8)
    case IntS16Const(value: Short) extends Term(Type.IntS16)
    case IntS32Const(value: Int)   extends Term(Type.IntS32)
    case IntS64Const(value: Long)  extends Term(Type.IntS64)

    case Unit0Const                 extends Term(Type.Unit0)
    case Bool1Const(value: Boolean) extends Term(Type.Bool1)

    // case UnitConst                              extends Term(Type.Unit)
    // case Bool1Const(value: Boolean)              extends Term(Type.Bool)
    // case ByteConst(value: Byte)                 extends Term(Type.Byte)
    // case CharConst(value: Char)                 extends Term(Type.Char)
    // case ShortConst(value: Short)               extends Term(Type.Short)
    // case IntConst(value: Int)                   extends Term(Type.Int)
    // case LongConst(value: Long)                 extends Term(Type.Long)
    // case FloatConst(value: Float)               extends Term(Type.Float)
    // case DoubleConst(value: Double)             extends Term(Type.Double)
  }

  case class SourcePosition(file: String, line: Int, col: Option[Int]) derives MsgPack.Codec

  // enum BinaryIntrinsicKind {
  //   case Add, Sub, Mul, Div, Rem
  //   case Pow
  //   case Min, Max
  //   case Atan2, Hypot
  //   case BAnd, BOr, BXor, BSL, BSR, BZSR
  //   case LogicEq, LogicNeq, LogicAnd, LogicOr, LogicLte, LogicGte, LogicLt, LogicGt
  // }

  // enum UnaryIntrinsicKind {
  //   case Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh
  //   case Signum, Abs, Round, Ceil, Floor, Rint
  //   case Sqrt, Cbrt, Exp, Expm1, Log, Log1p, Log10
  //   case BNot
  //   case Pos, Neg
  //   case LogicNot
  // }

  // enum NullaryIntrinsicKind {
  //   // Int
  //   case GpuGlobalIdxX, GpuGlobalIdxY, GpuGlobalIdxZ
  //   case GpuGlobalSizeX, GpuGlobalSizeY, GpuGlobalSizeZ
  //   case GpuGroupIdxX, GpuGroupIdxY, GpuGroupIdxZ
  //   case GpuGroupSizeX, GpuGroupSizeY, GpuGroupSizeZ
  //   case GpuLocalIdxX, GpuLocalIdxY, GpuLocalIdxZ
  //   case GpuLocalSizeX, GpuLocalSizeY, GpuLocalSizeZ

  //   // Unit
  //   case GpuGroupBarrier // __syncthreads() or barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
  //   case GpuGroupFence   // __threadfence_block() or mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

  //   case Assert
  // }

  case class Overload(args: List[Type], rtn: Type) derives MsgPack.Codec


  object Spec {
    inline def GpuIndex          = List(Overload(List(Type.IntU32), Type.IntU32))
    inline def NullaryUnit       = List(Overload(List[Type](), Type.Unit0))
  }
  enum Spec(val overloads: List[Overload], val terms: List[Term], val tpe: Type) derives MsgPack.Codec {
    // nullary misc
    case Assert extends Spec(Spec.NullaryUnit, List[Term](), Type.Nothing)
    // nullary GPU control
    case GpuBarrierGlobal extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuBarrierLocal  extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuBarrierAll    extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceGlobal   extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceLocal    extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceAll      extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    //  unary GPU indexing
    case GpuGlobalIdx(dim: Term)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGlobalSize(dim: Term) extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGroupIdx(dim: Term)   extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuGroupSize(dim: Term)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuLocalIdx(dim: Term)   extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
    case GpuLocalSize(dim: Term)  extends Spec(Spec.GpuIndex, List(dim), Type.IntU32)
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
  enum Intr(val overloads: List[Overload], val terms: List[Term], val tpe: Type) derives MsgPack.Codec {

    // // --- unary ---

    // // nullary misc
    // case Assert extends Intr(Intr.NullaryUnit, List[Term](), Type.Nothing)
    // // nullary GPU control
    // case GpuBarrierGlobal extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // case GpuBarrierLocal  extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // case GpuBarrierAll    extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // case GpuFenceGlobal   extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // case GpuFenceLocal    extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // case GpuFenceAll      extends Intr(Intr.NullaryUnit, List[Term](), Type.Unit0)
    // //  unary GPU indexing
    // case GpuGlobalIdx(dim: Term)  extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // case GpuGlobalSize(dim: Term) extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // case GpuGroupIdx(dim: Term)   extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // case GpuGroupSize(dim: Term)  extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // case GpuLocalIdx(dim: Term)   extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // case GpuLocalSize(dim: Term)  extends Intr(Intr.GpuIndex, List(dim), Type.IntU32)
    // unary math ops
    // /**/case Abs(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformSIntegral, List(x), rtn)
    // /**/case Pos(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformSIntegral, List(x), rtn)
    // /**/case Neg(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformSIntegral, List(x), rtn)
    // /**/case Sin(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Cos(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Tan(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Asin(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Acos(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Atan(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Sinh(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Cosh(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Tanh(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Signum(x: Term, rtn: Type) extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Round(x: Term, rtn: Type)  extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Ceil(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Floor(x: Term, rtn: Type)  extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Rint(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Sqrt(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Cbrt(x: Term, rtn: Type)   extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Exp(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Expm1(x: Term, rtn: Type)  extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Log(x: Term, rtn: Type)    extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Log1p(x: Term, rtn: Type)  extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // /**/case Log10(x: Term, rtn: Type)  extends Intr(Intr.UnaryUniformFractional, List(x), rtn)
    // unary bit manip
    case BNot(x: Term, rtn: Type) extends Intr(Intr.UnaryUniformIntegral, List(x), rtn)
    // unary logic
    case LogicNot(x: Term) extends Intr(Intr.BinaryUniformBool, List(x), Type.Bool1)

    case Pos(x: Term, rtn: Type)    extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)
    case Neg(x: Term, rtn: Type)    extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)

    // --- binary ---

    // binary math ops
    case Add(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Sub(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Mul(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Div(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Rem(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Min(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Max(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    
    // /**/case Pow(x: Term, y: Term, rtn: Type)   extends Intr(Intr.BinaryUniformFractional, List(x, y), rtn)
    // /**/case Atan2(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformFractional, List(x, y), rtn)
    // /**/case Hypot(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformFractional, List(x, y), rtn)
    // binary bit manip.
    case BAnd(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BOr(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BXor(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSL(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSR(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BZSR(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    // binary logic
    case LogicAnd(x: Term, y: Term) extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicOr(x: Term, y: Term)  extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicEq(x: Term, y: Term)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicNeq(x: Term, y: Term) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLte(x: Term, y: Term) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGte(x: Term, y: Term) extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLt(x: Term, y: Term)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGt(x: Term, y: Term)  extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)

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
  enum Math(val overloads: List[Overload], val terms: List[Term], val tpe: Type) derives MsgPack.Codec {
    // --- unary ---
    
    case Abs(x: Term, rtn: Type)    extends Math(Intr.BinaryUniformNumeric, List(x), rtn)
    case Sin(x: Term, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cos(x: Term, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tan(x: Term, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Asin(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Acos(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Atan(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sinh(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cosh(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tanh(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Signum(x: Term, rtn: Type) extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Round(x: Term, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Ceil(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Floor(x: Term, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Rint(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sqrt(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cbrt(x: Term, rtn: Type)   extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Exp(x: Term, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Expm1(x: Term, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log(x: Term, rtn: Type)    extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log1p(x: Term, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log10(x: Term, rtn: Type)  extends Math(Math.UnaryUniformFractional, List(x), rtn)
    // --- binary ---
    case Pow(x: Term, y: Term, rtn: Type)   extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Atan2(x: Term, y: Term, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Hypot(x: Term, y: Term, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
  }

  

  enum Expr(val tpe: Type) derives MsgPack.Codec {

    // case NullaryIntrinsic(kind: NullaryIntrinsicKind, rtn: Type)                     extends Expr(rtn)
    // case UnaryIntrinsic(lhs: Term, kind: UnaryIntrinsicKind, rtn: Type)              extends Expr(rtn)
    // case BinaryIntrinsic(lhs: Term, rhs: Term, kind: BinaryIntrinsicKind, rtn: Type) extends Expr(rtn)

    case SpecOp(op: Spec) extends Expr(op.tpe) 
    case MathOp(op: Math) extends Expr(op.tpe)
    case IntrOp(op: Intr) extends Expr(op.tpe)

    case Cast(from: Term, as: Type)                   extends Expr(as)
    case Alias(ref: Term)                             extends Expr(ref.tpe)
    case Index(lhs: Term, idx: Term, component: Type) extends Expr(component)
    case Alloc(component: Type, size: Term)           extends Expr(Type.Array(component, Type.Space.Global))

    case Invoke(
        name: Sym,
        tpeArgs: List[Type],
        receiver: Option[Term],
        args: List[Term],
        captures: List[Term],
        rtn: Type
    ) extends Expr(rtn)

    // case Suspend(args: List[Named], stmts: List[Stmt], rtn: Type, shape: Type.Exec) extends Expr(shape)

  }

  case class Arg(named: Named, pos: Option[SourcePosition] = None) derives MsgPack.Codec

  enum Stmt derives MsgPack.Codec {
    case Block(stmts: List[Stmt])
    case Comment(value: String)
    case Var(name: Named, expr: Option[Expr])
    case Mut(
        name: Term,
        expr: Expr,
        copy: Boolean // FIXME do we need this copy thing now that we have value/ref semantics???
    )
    case Update(lhs: Term, idx: Term, value: Term)
    case While(tests: List[Stmt], cond: Term, body: List[Stmt])
    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
  }

  case class StructMember(named: Named, isMutable: Boolean) derives MsgPack.Codec
  case class StructDef(            //
      name: Sym,                   //
      isReference: Boolean,        //
      tpeVars: List[String],       //
      members: List[StructMember], //
      parents: List[Sym]           //
  ) derives MsgPack.Codec

  case class Mirror(                //
      source: Sym,                  //
      sourceParents: List[Sym],     //
      struct: StructDef,            //
      functions: List[Function],    //
      dependencies: List[StructDef] //
  ) derives MsgPack.Codec

  case class Signature(
      name: Sym,
      tpeVars: List[String],
      receiver: Option[Type],
      args: List[Type],
      moduleCaptures: List[Type],
      termCaptures: List[Type],
      rtn: Type
  ) derives MsgPack.Codec

  case class InvokeSignature(
      name: Sym,
      tpeVars: List[Type],
      receiver: Option[Type],
      args: List[Type],
      captures: List[Type],
      rtn: Type
  ) derives MsgPack.Codec

  case class Function(           //
      name: Sym,                 //
      tpeVars: List[String],     //
      receiver: Option[Arg],     //
      args: List[Arg],           //
      moduleCaptures: List[Arg], //
      termCaptures: List[Arg],   //
      rtn: Type,                 //
      body: List[Stmt]           //
  ) derives MsgPack.Codec
  object Function {
    enum Kind derives MsgPack.Codec { case Internal, Exported  }
    enum Attr derives MsgPack.Codec { case FPRelaxed, FPStrict }
  } //

  case class Program(
      entry: Function, // TODO merge entry with the rest when we add internal/export attrs
      functions: List[Function],
      defs: List[StructDef]
  ) derives MsgPack.Codec

}
