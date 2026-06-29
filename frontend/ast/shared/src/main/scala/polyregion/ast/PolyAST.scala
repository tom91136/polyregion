package polyregion.ast

import scala.collection.immutable.ArraySeq
import polyregion.ast.PolyAST.Type.Space
import polyregion.ast.PolyAST.Type.Kind

object PolyAST {

  case class Sym(fqn: List[String]) derives MsgPack.Codec {
    infix def :+(s: String): Sym = Sym(fqn :+ s)
    infix def ~(s: Sym): Sym     = Sym(fqn ++ s.fqn)
    def last: String             = fqn.last
  }
  object Sym {
    def apply(raw: String): Sym = {
      require(raw.trim.nonEmpty)
      Sym(raw.trim.split('.').toList)
    }
    def unapply(xs: List[String]): Option[(Sym, String)] =
      xs.lastOption.map(x => Sym(xs.init) -> x)
  }

  object Type {
    enum Space derives MsgPack.Codec { case Global, Local, Private, Constant }
    enum Kind derives MsgPack.Codec  { case None, Ref, Integral, Fractional  }
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

    case Struct(name: Sym, args: List[Type])             extends Type(Type.Kind.Ref)
    case Ptr(comp: Type, space: Type.Space)              extends Type(Type.Kind.Ref)
    case Arr(comp: Type, length: Int, space: Type.Space) extends Type(Type.Kind.Ref)

    case Var(name: String)                                        extends Type(Type.Kind.None)
    case Exec(tpeVars: List[String], args: List[Type], rtn: Type) extends Type(Type.Kind.None)
  }

  enum Region derives MsgPack.Codec {
    case Rooted(root: Named)
    case Opaque
  }

  enum PathStep derives MsgPack.Codec {
    case Field(name: String)
    case Deref
    case Index(idx: Int)
    case IndexDyn(idx: Term)
  }

  enum Term(val tpe: Type) derives MsgPack.Codec {
    case Float16Const(value: Float)  extends Term(Type.Float16)
    case Float32Const(value: Float)  extends Term(Type.Float32)
    case Float64Const(value: Double) extends Term(Type.Float64)

    case IntU8Const(value: Byte)  extends Term(Type.IntU8)
    case IntU16Const(value: Char) extends Term(Type.IntU16)
    case IntU32Const(value: Int)  extends Term(Type.IntU32)
    case IntU64Const(value: Long) extends Term(Type.IntU64)

    case IntS8Const(value: Byte)   extends Term(Type.IntS8)
    case IntS16Const(value: Short) extends Term(Type.IntS16)
    case IntS32Const(value: Int)   extends Term(Type.IntS32)
    case IntS64Const(value: Long)  extends Term(Type.IntS64)

    case Unit0Const                                                  extends Term(Type.Unit0)
    case Bool1Const(value: Boolean)                                  extends Term(Type.Bool1)
    case NullPtrConst(comp: Type, space: Type.Space, region: Region) extends Term(Type.Ptr(comp, space))
    case StringConst(value: String) extends Term(Type.Ptr(Type.IntS8, Type.Space.Constant))
    case Poison(t: Type)            extends Term(t)

    case Select(root: Named, steps: List[PathStep], override val tpe: Type) extends Term(tpe)
  }

  enum Expr(val tpe: Type) derives MsgPack.Codec {
    case Alias(ref: Term)                        extends Expr(ref.tpe)
    case SpecOp(op: Spec)                        extends Expr(op.tpe)
    case MathOp(op: Math)                        extends Expr(op.tpe)
    case IntrOp(op: Intr)                        extends Expr(op.tpe)
    case Cast(from: Term, as: Type)              extends Expr(as)
    case Index(lhs: Term, idx: Term, comp: Type) extends Expr(comp)
    case RefTo(lhs: Term, idx: Option[Term], comp: Type, space: Type.Space, region: Region)
        extends Expr(Type.Ptr(comp, space))
    case Alloc(comp: Type, size: Term, space: Type.Space, region: Region) extends Expr(Type.Ptr(comp, space))
    case Invoke(
        name: Sym,
        tpeArgs: List[Type],
        receiver: Option[Term],
        args: List[Term],
        rtn: Type
    )                                                           extends Expr(rtn)
    case ForeignCall(name: String, args: List[Term], rtn: Type) extends Expr(rtn)
    case OffsetOf(structTpe: Type, field: String)               extends Expr(Type.IntU64)
    case SizeOf(forTpe: Type)                                   extends Expr(Type.IntU64)
  }

  enum Stmt derives MsgPack.Codec {
    case Var(name: Named, expr: Option[Expr], isMutable: Boolean = false)
    case Mut(name: Term.Select, expr: Expr)
    case Update(lhs: Term.Select, idx: Term, value: Term)
    case While(cond: Term, body: List[Stmt])
    case ForRange(induction: Named, lbIncl: Term, ubExcl: Term, step: Term, body: List[Stmt])

    case Break
    case Cont
    case Cond(cond: Term, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
    case Annotated(inner: Stmt, pos: Option[SourcePosition] = None, comment: Option[String] = None)
  }

  case class Overload(args: List[Type], rtn: Type) derives MsgPack.Codec

  object Spec {
    inline def GpuIndex    = List(Overload(List(Type.IntU32), Type.IntU32))
    inline def NullaryUnit = List(Overload(List[Type](), Type.Unit0))
  }
  enum Spec(val overloads: List[Overload], val terms: List[Term], val tpe: Type) derives MsgPack.Codec {
    case Assert                   extends Spec(Spec.NullaryUnit, List[Term](), Type.Nothing)
    case GpuBarrierGlobal         extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuBarrierLocal          extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuBarrierAll            extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceGlobal           extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceLocal            extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
    case GpuFenceAll              extends Spec(Spec.NullaryUnit, List[Term](), Type.Unit0)
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
    case BNot(x: Term, rtn: Type)          extends Intr(Intr.UnaryUniformIntegral, List(x), rtn)
    case LogicNot(x: Term)                 extends Intr(Intr.BinaryUniformBool, List(x), Type.Bool1)
    case Pos(x: Term, rtn: Type)           extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)
    case Neg(x: Term, rtn: Type)           extends Intr(Intr.BinaryUniformNumeric, List(x), rtn)
    case Add(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Sub(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Mul(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Div(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Rem(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Min(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case Max(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformNumeric, List(x, y), rtn)
    case BAnd(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BOr(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BXor(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSL(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BSR(x: Term, y: Term, rtn: Type)  extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case BZSR(x: Term, y: Term, rtn: Type) extends Intr(Intr.BinaryUniformIntegral, List(x, y), rtn)
    case LogicAnd(x: Term, y: Term)        extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicOr(x: Term, y: Term)         extends Intr(Intr.BinaryUniformBool, List(x, y), Type.Bool1)
    case LogicEq(x: Term, y: Term)         extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicNeq(x: Term, y: Term)        extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLte(x: Term, y: Term)        extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGte(x: Term, y: Term)        extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicLt(x: Term, y: Term)         extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
    case LogicGt(x: Term, y: Term)         extends Intr(Intr.BinaryUniformLogic, List(x, y), Type.Bool1)
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
    case Abs(x: Term, rtn: Type)            extends Math(Intr.BinaryUniformNumeric, List(x), rtn)
    case Sin(x: Term, rtn: Type)            extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cos(x: Term, rtn: Type)            extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tan(x: Term, rtn: Type)            extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Asin(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Acos(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Atan(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sinh(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cosh(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Tanh(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Signum(x: Term, rtn: Type)         extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Round(x: Term, rtn: Type)          extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Ceil(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Floor(x: Term, rtn: Type)          extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Rint(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Sqrt(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Cbrt(x: Term, rtn: Type)           extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Exp(x: Term, rtn: Type)            extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Expm1(x: Term, rtn: Type)          extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log(x: Term, rtn: Type)            extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log1p(x: Term, rtn: Type)          extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Log10(x: Term, rtn: Type)          extends Math(Math.UnaryUniformFractional, List(x), rtn)
    case Pow(x: Term, y: Term, rtn: Type)   extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Atan2(x: Term, y: Term, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
    case Hypot(x: Term, y: Term, rtn: Type) extends Math(Math.BinaryUniformFractional, List(x, y), rtn)
  }

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
      rtn: Type
  ) derives MsgPack.Codec

  case class Arg(named: Named, pos: Option[SourcePosition] = None) derives MsgPack.Codec
  case class StructDef(
      name: Sym,
      tpeVars: List[String],
      members: List[Named],
      parents: List[Type.Struct],
      isUnion: Boolean = false
  ) derives MsgPack.Codec

  case class Mirror(
      source: Sym,
      sourceParents: List[Sym],
      structDef: StructDef,
      functions: List[Function],
      dependencies: List[StructDef]
  ) derives MsgPack.Codec

  object Function {
    enum Visibility derives MsgPack.Codec { case Internal, Exported }
    enum FpMode derives MsgPack.Codec     { case Relaxed, Strict    }
    enum Affinity derives MsgPack.Codec   { case Offload, Host      }
  }
  case class Function(
      name: Sym,
      tpeVars: List[String],
      receiver: Option[Arg],
      args: List[Arg],
      moduleCaptures: List[Arg],
      termCaptures: List[Arg],
      rtn: Type,
      body: List[Stmt],
      visibility: Function.Visibility,
      fpMode: Function.FpMode,
      isEntry: Boolean,
      affinity: Function.Affinity = Function.Affinity.Offload
  ) derives MsgPack.Codec

  enum PassPhase derives MsgPack.Codec { case Initial, PostMono }

  case class MetaEntry(key: String, value: String) derives MsgPack.Codec

  case class Program(
      entry: Function,
      functions: List[Function],
      defs: List[StructDef],
      phase: PassPhase = PassPhase.Initial,
      metadata: List[MetaEntry] = Nil
  ) derives MsgPack.Codec {
    def hostFunctions: List[Function]     = functions.filter(_.affinity == Function.Affinity.Host)
    def offloadFunctions: List[Function]  = functions.filter(_.affinity == Function.Affinity.Offload)
    def meta(key: String): Option[String] = metadata.find(_.key == key).map(_.value)
  }

  case class StructLayoutMember(name: Named, offsetInBytes: Long, sizeInBytes: Long) derives MsgPack.Codec
  case class StructLayout(
      name: String,
      sizeInBytes: Long,
      alignment: Long,
      members: List[StructLayoutMember]
  ) derives MsgPack.Codec

  case class CompileEvent(
      epochMillis: Long,
      elapsedNanos: Long,
      name: String,
      data: String,
      items: List[CompileEvent] = Nil
  ) derives MsgPack.Codec

  case class PassArg(name: String, value: String) derives MsgPack.Codec
  case class PassSpec(name: String, args: List[PassArg]) derives MsgPack.Codec
  case class PassPipeline(steps: List[PassSpec]) derives MsgPack.Codec
  case class PassRunResult(program: Program, event: CompileEvent) derives MsgPack.Codec

  case class CompileResult(
      binary: Option[ArraySeq[Byte]],
      features: List[String],
      events: List[CompileEvent],
      layouts: List[StructLayout],
      messages: String
  ) derives MsgPack.Codec

  object PolyPassAbi {

    inline val Version    = 1
    inline val Prefix     = "polypass_"
    inline val EnvPlugins = "POLYPASS_PLUGINS"

    object Status {
      inline val Ok            = 0
      inline val AllocFailed   = 1
      inline val PipelineError = 2
      inline val UnknownPass   = 3
      inline val AbiMismatch   = 4
    }
    def docs(d: String): Nothing = ???

    object AbiVersion {
      def apply(): Int = docs("ABI version the plugin was built against; polyc refuses mismatched plugins.")
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object PassCount {
      def apply(): Long                   = docs("Number of passes this plugin contributes.")
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object PassName {
      def apply(i: Long): String =
        docs("Bare identifier of the i-th pass (e.g. \"FullOpt\"). Process-lifetime; NULL if i out of range.")
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object PassDescr {
      def apply(i: Long): String =
        docs("Optional human-readable description of the i-th pass; may return NULL or \"\".")
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object RunPasses {
      def apply(steps: List[String], in: Array[Byte]): Array[Byte] = docs(
        "Run the NULL-terminated `steps` list against `in` (msgpack Program); steps share in-process state. On Ok, *out is a malloc'd PassRunResult; caller frees via " +
          Free.Name + "."
      )
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object LastError {
      def apply(): String = docs(
        "NUL-terminated diagnostic for the most recent non-Ok status. Valid until the next " + RunPasses.Name +
          " call; NULL when no error is set."
      )
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }

    object Free {
      def apply(ptr: Any): Unit           = docs("Release a buffer returned by " + RunPasses.Name + ".")
      transparent inline def Name: String = AbiMacros.cName[this.type](Prefix)
    }
  }

  object Conventions {
    inline val EntryName               = "_main"
    inline val ThisReceiver            = "#this"
    inline val CaptureArg              = "#capture"
    inline val BaseFieldPrefix         = "#base"
    inline val EmptyStructStorageField = "#empty_struct_storage"
    inline val KernelBundleType        = "KernelBundle"
    object Macros {
      inline val PolyreflectTrackAnnotation     = "polyreflect-track"
      inline val PolyreflectRtProtectAnnotation = "polyreflect-rt-protect"
      inline val PolyreflectRtOdrAnnotation     = "polyreflect-rt-odr"
      inline val PolyregionLocalAnnotation      = "__polyregion_local"
    }
    object RuntimeAbi {
      inline val SmaAlloc       = "polyrt_sma_alloc"
      inline val SmaEnsure      = "polyrt_sma_ensure"
      inline val SmaEnsureMin   = "polyrt_sma_ensure_min"
      inline val SmaEnsureDeep  = "polyrt_sma_ensure_deep"
      inline val SmaPointeeSize = "polyrt_sma_pointee_size"
      inline val SmaPatch       = "polyrt_sma_patch"
      inline val SmaReadAlloc   = "polyrt_sma_read_alloc"
      inline val SmaReadDeep    = "polyrt_sma_read_deep"
      inline val SmaVisitClear  = "polyrt_sma_visit_clear"
      inline val SmaMirrorGraph = "polyrt_sma_mirror_graph"
      inline val SmaReadGraph   = "polyrt_sma_read_graph"
      inline val SmaPoolReset   = "polyrt_sma_pool_reset"
      inline val SmaPoolPtr     = "polyrt_sma_pool_ptr"
    }
    object Reflect {
      inline val MirrorBitcodeGlobal = "polyregion_mirror_bc"
      inline val MirrorPrelude       = "__polyregion_mirror_prelude"
      inline val MirrorPostlude      = "__polyregion_mirror_postlude"

      inline val FlagVerbose     = "polyreflect-verbose"
      inline val FlagEarly       = "polyreflect-early"
      inline val FlagLate        = "polyreflect-late"
      inline val PassRecordAlloc = "polyreflect-record-alloc"
      inline val PassStack       = "polyreflect-stack"
      inline val PassMem         = "polyreflect-mem"
      inline val PassLinkMirror  = "polyreflect-link-mirror"
      inline val PassProtectRt   = "polyreflect-protect-rt"
      inline val PassInterpose   = "polyreflect-interpose"
    }
  }

  object Enums {
    case class JavaMirror(pkg: String, name: String)
    trait Variant {
      def value: Int
      def namespace: String
      def cppExport: Boolean
      def javaMirror: Option[JavaMirror] = None
      def java: Option[String]           = None
      def javaSize: Option[String]       = None
      final def name: String             = toString
      final def cppName: String          = getClass.getSuperclass.getSimpleName
    }

    enum Backend(val value: Int) extends Variant {
      def namespace = "invoke"
      def cppExport = true
      case CUDA              extends Backend(0)
      case HIP               extends Backend(1)
      case HSA               extends Backend(2)
      case OpenCL            extends Backend(3)
      case Vulkan            extends Backend(4)
      case Metal             extends Backend(5)
      case SharedObject      extends Backend(6)
      case RelocatableObject extends Backend(7)
      case LevelZero         extends Backend(8)
    }

    enum Access(val value: Int, override val java: Option[String]) extends Variant {
      def namespace           = "invoke"
      def cppExport           = false
      override def javaMirror = Some(JavaMirror("runtime", "Access"))
      case RW extends Access(1, Some("RW"))
      case RO extends Access(2, Some("RO"))
      case WO extends Access(3, Some("WO"))
    }

    enum Target(val value: Int, override val java: Option[String]) extends Variant {
      def namespace           = "compiletime"
      def cppExport           = true
      override def javaMirror = Some(JavaMirror("compiler", "Target"))
      case Object_LLVM_HOST            extends Target(10, Some("LLVM_HOST"))
      case Object_LLVM_x86_64          extends Target(11, Some("LLVM_X86_64"))
      case Object_LLVM_AArch64         extends Target(12, Some("LLVM_AARCH64"))
      case Object_LLVM_ARM             extends Target(13, Some("LLVM_ARM"))
      case Object_LLVM_NVPTX64         extends Target(20, Some("LLVM_NVPTX64"))
      case Object_LLVM_AMDGCN          extends Target(21, Some("LLVM_AMDGCN"))
      case Object_LLVM_SPIRV32_Kernel  extends Target(22, Some("LLVM_SPIRV32_KERNEL"))
      case Object_LLVM_SPIRV64_Kernel  extends Target(23, Some("LLVM_SPIRV64_KERNEL"))
      case Object_LLVM_SPIRV_GLCompute extends Target(24, Some("LLVM_SPIRV_GLCOMPUTE"))
      case Source_C_C11                extends Target(30, Some("C_C11"))
      case Source_C_OpenCL1_1          extends Target(31, Some("C_OpenCL1_1"))
      case Source_C_Metal1_0           extends Target(32, Some("C_Metal1_0"))
    }

    enum OptLevel(val value: Int, override val java: Option[String]) extends Variant {
      def namespace           = "compiletime"
      def cppExport           = true
      override def javaMirror = Some(JavaMirror("compiler", "Opt"))
      case O0    extends OptLevel(10, Some("O0"))
      case O1    extends OptLevel(11, Some("O1"))
      case O2    extends OptLevel(12, Some("O2"))
      case O3    extends OptLevel(13, Some("O3"))
      case Ofast extends OptLevel(14, Some("Ofast"))
    }

    enum Type(val value: Int, override val java: Option[String] = None, override val javaSize: Option[String] = None)
        extends Variant {
      def namespace           = "runtime"
      def cppExport           = true
      override def javaMirror = Some(JavaMirror("runtime", "Type"))
      case Void    extends Type(1, Some("VOID"), Some("0"))
      case Bool1   extends Type(2, Some("BOOL"), Some("Byte.BYTES"))
      case IntU8   extends Type(3)
      case IntU16  extends Type(4, Some("CHAR"), Some("Character.BYTES"))
      case IntU32  extends Type(5)
      case IntU64  extends Type(6)
      case IntS8   extends Type(7, Some("BYTE"), Some("Byte.BYTES"))
      case IntS16  extends Type(8, Some("SHORT"), Some("Short.BYTES"))
      case IntS32  extends Type(9, Some("INT"), Some("Integer.BYTES"))
      case IntS64  extends Type(10, Some("LONG"), Some("Long.BYTES"))
      case Float16 extends Type(11)
      case Float32 extends Type(12, Some("FLOAT"), Some("Float.BYTES"))
      case Float64 extends Type(13, Some("DOUBLE"), Some("Double.BYTES"))
      case Ptr     extends Type(14, Some("PTR"), Some("Long.BYTES"))
      case Scratch extends Type(15)
    }

    enum PlatformKind(val value: Int) extends Variant {
      def namespace = "runtime"
      def cppExport = true
      case HostThreaded extends PlatformKind(1)
      case Managed      extends PlatformKind(2)
    }

    enum ModuleFormat(val value: Int) extends Variant {
      def namespace = "runtime"
      def cppExport = true
      case Source          extends ModuleFormat(1)
      case Object          extends ModuleFormat(2)
      case DSO             extends ModuleFormat(3)
      case PTX             extends ModuleFormat(4)
      case HSACO           extends ModuleFormat(5)
      case SPIRV_Kernel    extends ModuleFormat(6)
      case SPIRV_GLCompute extends ModuleFormat(7)
    }

    val All: List[Variant] =
      Backend.values.toList ++ Access.values.toList ++ Target.values.toList ++ OptLevel.values.toList ++
        Type.values.toList ++ PlatformKind.values.toList ++ ModuleFormat.values.toList
  }

  extension (s: Sym) {
    def repr: String = s.fqn.mkString(".")
  }

  extension (t: SourcePosition) {
    def repr: String = s"${t.file}:${t.line}${t.col.map(c => s":$c").getOrElse("")}"
  }

  extension (t: Type.Space) {
    def repr: String = t match {
      case Space.Global   => ""
      case Space.Local    => "^Local"
      case Space.Private  => "^Private"
      case Space.Constant => "^Constant"
    }
  }

  extension (r: Region) {
    def repr: String = r match {
      case Region.Rooted(root) => s"@${root.symbol}"
      case Region.Opaque       => "@opaque"
    }
  }

  extension (k: Type.Kind) {
    def repr: String = k match {
      case Kind.None       => "None"
      case Kind.Ref        => "Ref"
      case Kind.Integral   => "Integral"
      case Kind.Fractional => "Fractional"
    }
  }

  extension (s: PathStep) {
    def repr: String = s match {
      case PathStep.Field(name)   => s".$name"
      case PathStep.Deref         => "->*"
      case PathStep.Index(idx)    => s"[$idx]"
      case PathStep.IndexDyn(idx) => s"[${idx.repr}]"
    }
  }

  extension (t: Type) {
    def repr: String = t match {
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

      case Type.Struct(name, args) =>
        s"${name.repr}<${args.map(_.repr).mkString(",")}>"
      case Type.Ptr(comp, space)         => s"${comp.repr}*${space.repr}"
      case Type.Arr(comp, length, space) => s"${comp.repr}[$length]${space.repr}"
      case Type.Var(name)                => s"#$name"
      case Type.Exec(tpeVars, args, rtn) =>
        s"<${tpeVars.mkString(",")}>(${args.map(_.repr).mkString(",")}) => ${rtn.repr}"
    }
  }

  extension (n: Named) {
    def repr: String = s"${n.symbol}"
  }

  extension (t: Term) {
    def repr: String = t match {
      case Term.Float16Const(x) => s"f16($x)"
      case Term.Float32Const(x) => s"f32($x)"
      case Term.Float64Const(x) => s"f64($x)"

      case Term.IntU8Const(x)  => s"u8($x)"
      case Term.IntU16Const(x) => s"u16($x)"
      case Term.IntU32Const(x) => s"u32($x)"
      case Term.IntU64Const(x) => s"u64($x)"

      case Term.IntS8Const(x)  => s"i8($x)"
      case Term.IntS16Const(x) => s"i16($x)"
      case Term.IntS32Const(x) => s"i32($x)"
      case Term.IntS64Const(x) => s"i64($x)"

      case Term.Unit0Const                     => "unit0(())"
      case Term.Bool1Const(x)                  => s"bool1($x)"
      case Term.NullPtrConst(x, space, region) => s"nullptr[${x.repr}, ${space.repr}${region.repr}]"
      case Term.StringConst(value)             => s"str($value)"
      case Term.Poison(t)                      => s"__poison__ /* poison of type ${t.repr} */"
      case Term.Select(root, steps, tpe) =>
        s"${root.symbol}: ${root.tpe.repr}${steps.map(_.repr).mkString("")}"
    }
  }

  extension (e: Expr) {
    def repr: String = e match {
      case Expr.Alias(ref) => ref.repr
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

      case Expr.Cast(from, as)        => s"(${from.repr}).to[${as.repr}]"
      case Expr.Index(lhs, idx, comp) => s"(${lhs.repr}).index[${comp.repr}](${idx.repr})"
      case Expr.RefTo(lhs, idx, comp, space, region) =>
        s"(${lhs.repr}).refTo[${comp.repr}, ${space.repr}${region.repr}](${idx.map(_.repr).getOrElse("")})"
      case Expr.Alloc(comp, size, space, region) => s"alloc[${comp.repr}, ${space.repr}${region.repr}](${size.repr})"
      case Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
        s"${receiver.map(r => s"${r.repr}.").getOrElse("")}${name.repr}<${tpeArgs.map(_.repr).mkString(",")}>(${args
            .map(_.repr)
            .mkString(", ")}): ${rtn.repr}"
      case Expr.ForeignCall(name, args, rtn) => s"$name(${args.map(_.repr).mkString(", ")}): ${rtn.repr}"
      case Expr.OffsetOf(tpe, field)         => s"offsetof(${tpe.repr}, $field)"
      case Expr.SizeOf(tpe)                  => s"sizeof(${tpe.repr})"
    }
  }

  extension (stmt: Stmt) {
    def repr: String = stmt match {
      case Stmt.Var(name, rhs, isMutable) =>
        s"${if (isMutable) "var" else "val"} ${name.symbol}: ${name.tpe.repr} = ${rhs.map(_.repr).getOrElse("_")}"
      case Stmt.Mut(name, expr)         => s"${name.repr} = ${expr.repr}"
      case Stmt.Update(lhs, idx, value) => s"(${lhs.repr}).update(${idx.repr}) = ${value.repr}"
      case Stmt.While(cond, body) =>
        s"while(${cond.repr})${"{"}\n${body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
      case Stmt.ForRange(ind, lb, ub, step, body) =>
        s"for(${ind.symbol}: ${ind.tpe.repr} = ${lb.repr}; < ${ub.repr}; += ${step.repr})${"{"}\n${body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
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
      case Stmt.Annotated(inner, pos, comment) =>
        s"${inner.repr}${pos.map(p => s" /* ${p.repr} */").getOrElse("")}${comment.map(c => s" /* $c */").getOrElse("")}"
    }
  }

  extension (a: Arg) {
    def repr: String =
      s"${a.named.symbol}: ${a.named.tpe.repr}${a.pos.map(s => s"/* ${s.repr} */").getOrElse("")}"
  }

  extension (v: Function.Visibility) {
    def repr: String = v match {
      case Function.Visibility.Internal => "Internal"
      case Function.Visibility.Exported => "Exported"
    }
  }

  extension (m: Function.FpMode) {
    def repr: String = m match {
      case Function.FpMode.Relaxed => "FPRelaxed"
      case Function.FpMode.Strict  => "FPStrict"
    }
  }

  extension (f: Signature) {
    def repr: String =
      s"def ${f.receiver.map(r => s"${r.repr}.").getOrElse("")}${f.name.repr}<${f.tpeVars
          .mkString(",")}>(${f.args.map(_.repr).mkString(", ")}): ${f.rtn.repr} /* mod=${f.moduleCaptures
          .map(_.repr)
          .mkString(",")} term=${f.termCaptures.map(_.repr).mkString(",")} */"
  }

  extension (f: Function) {
    def repr: String =
      s"def ${f.receiver.map(r => s"${r.repr}.").getOrElse("")}${f.name.repr}<${f.tpeVars.mkString(",")}>(${f.args
          .map(a => s"${a.named.symbol}: ${a.named.tpe.repr}")
          .mkString(", ")}): ${f.rtn.repr} /* vis=${f.visibility.repr} fp=${f.fpMode.repr} entry=${f.isEntry} mod=${f.moduleCaptures
          .map(_.repr)
          .mkString(",")} term=${f.termCaptures
          .map(_.repr)
          .mkString(",")} */ ${"{"}\n${f.body.map(_.repr).mkString("\n").indent(2)}\n${"}"}"
  }

  extension (s: StructDef) {
    def repr: String =
      s"class ${s.name.repr}<${s.tpeVars.mkString(",")}>(${s.members
          .map(m => s"${m.symbol}: ${m.tpe.repr}")
          .mkString(", ")}) <: ${s.parents.map(_.repr).mkString(", ")}"
  }

  extension (s: Program) {
    def repr: String =
      s"${s.defs.map(_.repr).mkString("\n")}\n${s.entry.repr}\n${s.functions.map(_.repr).mkString("\n")}"
  }

  extension (l: StructLayout) {
    def repr: String =
      s"StructLayout[${l.name}, sizeInBytes=${l.sizeInBytes}, align=${l.alignment}]${"{"}\n${l.members
          .map(m => s"${m.name.symbol}: ${m.name.tpe.repr} (+${m.offsetInBytes},${m.sizeInBytes})")
          .mkString("\n")
          .indent(2)}\n${"}"}"
  }

}
