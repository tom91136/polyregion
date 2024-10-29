package polyregion.ast

import scala.collection.immutable.ArraySeq

object ScalaSRIR {

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
    case Float  extends Type(TypeKind.Fractional)
    case Double extends Type(TypeKind.Fractional)

    case Bool  extends Type(TypeKind.Integral)
    case Byte  extends Type(TypeKind.Integral)
    case Char  extends Type(TypeKind.Integral)
    case Short extends Type(TypeKind.Integral)
    case Int   extends Type(TypeKind.Integral)
    case Long  extends Type(TypeKind.Integral)

    case Nothing extends Type(TypeKind.None)
    case Unit   extends Type(TypeKind.None)

    case Struct(
        name: Sym,             //
        tpeVars: List[String], //
        args: List[Type],      //
        parents: List[Sym]
    )                                                                 extends Type(TypeKind.Ref)
    case Ptr(component: Type, length: Option[Int], space: Type.Space) extends Type(TypeKind.Ref)

    case Var(name: String)                                        extends Type(TypeKind.None)
    case Exec(tpeVars: List[String], args: List[Type], rtn: Type) extends Type(TypeKind.None)

    // def Exec(tpeVars: List[String], args: List[Type], rtn: Type) = Struct(Sym("" :: "Poly":: Nil), tpeVars, args, rtn, Nil)
  }
  object Type {
    enum Space derives MsgPack.Codec { case Global, Local }
  }

  case class Named(symbol: String, tpe: Type) derives MsgPack.Codec

  enum Term(val tpe: Type) derives MsgPack.Codec {
    case Select(init: List[Named], last: Named) extends Term(last.tpe)
    case Poison(t: Type)                        extends Term(t)
    case UnitConst                              extends Term(Type.Unit)
    case BoolConst(value: Boolean)              extends Term(Type.Bool)
    case ByteConst(value: Byte)                 extends Term(Type.Byte)
    case CharConst(value: Char)                 extends Term(Type.Char)
    case ShortConst(value: Short)               extends Term(Type.Short)
    case IntConst(value: Int)                   extends Term(Type.Int)
    case LongConst(value: Long)                 extends Term(Type.Long)
    case FloatConst(value: Float)               extends Term(Type.Float)
    case DoubleConst(value: Double)             extends Term(Type.Double)
  }

  case class SourcePosition(file: String, line: Int, col: Option[Int]) derives MsgPack.Codec

  enum Expr(val tpe: Type) derives MsgPack.Codec {
    case Cast(from: Term, as: Type)                           extends Expr(as)
    case Alias(ref: Term)                                     extends Expr(ref.tpe)
    case Index(lhs: Term, idx: Term, component: Type)         extends Expr(component)
    case RefTo(lhs: Term, idx: Option[Term], component: Type) extends Expr(Type.Ptr(component, None, Type.Space.Global))
    case Alloc(component: Type, size: Term)                   extends Expr(Type.Ptr(component, None, Type.Space.Global))

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
      body: List[Stmt],           //
      kind : Function.Kind
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

  case class CompileLayoutMember( //
      name: Named,
      offsetInBytes: Long,
      sizeInBytes: Long
  ) derives MsgPack.Codec
  case class CompileLayout( //
      name: Sym,
      sizeInBytes: Long,
      alignment: Long,
      members: List[CompileLayoutMember]
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
      layouts: List[CompileLayout],
      messages: String
  ) derives MsgPack.Codec

}
