package polyregion.ast

import polyregion.data.MsgPack
import cats.data.Op

object PolyAst {

  case class Sym(fqn: List[String]) derives MsgPack.Codec {
    def repr: String             = fqn.mkString(".")
    infix def :+(s: String): Sym = Sym(fqn :+ s)
    infix def ~(s: Sym): Sym     = Sym(fqn ++ s.fqn)

  }
  object Sym {
    def apply(raw: String): Sym = {
      require(!raw.isBlank)
      // normalise dollar
      Sym(raw.split('.').toList)
    }

    def unapply(xs: List[String]): Option[(Sym, String)] =
      xs.lastOption.map(x => Sym(xs.init) -> x)

  }

  enum TypeKind derives MsgPack.Codec {
    case None, Ref, Integral, Fractional
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

    case Unit extends Type(TypeKind.None)

    // specialisations
    case String                 extends Type(TypeKind.Ref)
    case Struct(name: Sym)      extends Type(TypeKind.Ref)
    case Array(component: Type) extends Type(TypeKind.Ref)
  }

  case class Named(symbol: String, tpe: Type) derives MsgPack.Codec

  enum Term(val tpe: Type) derives MsgPack.Codec {
    case Select(init: List[Named], last: Named) extends Term(last.tpe)
    case UnitConst                              extends Term(Type.Unit)
    case BoolConst(value: Boolean)              extends Term(Type.Bool)
    case ByteConst(value: Byte)                 extends Term(Type.Byte)
    case CharConst(value: Char)                 extends Term(Type.Char)
    case ShortConst(value: Short)               extends Term(Type.Short)
    case IntConst(value: Int)                   extends Term(Type.Int)
    case LongConst(value: Long)                 extends Term(Type.Long)
    case FloatConst(value: Float)               extends Term(Type.Float)
    case DoubleConst(value: Double)             extends Term(Type.Double)
    case StringConst(value: String)             extends Term(Type.String)
  }

  case class Position(file: String, line: Int, col: Int) derives MsgPack.Codec

  enum BinaryIntrinsicKind {
    case Add, Sub, Mul, Div, Rem
    case Pow
    case BAnd, BOr, BXor, BSL, BSR
  }

  enum BinaryLogicIntrinsicKind {
    case Eq, Neq, And, Or, Lte, Gte, Lt, Gt
  }

  enum UnaryIntrinsicKind {
    case Sin, Cos, Tan, Abs
    case BNot
  }

  enum UnaryLogicIntrinsicKind {
    case Not
  }

  enum Expr(val tpe: Type) derives MsgPack.Codec {

    case UnaryIntrinsic(lhs: Term, kind: UnaryIntrinsicKind, rtn: Type)              extends Expr(rtn)
    case BinaryIntrinsic(lhs: Term, rhs: Term, kind: BinaryIntrinsicKind, rtn: Type) extends Expr(rtn)

    case UnaryLogicIntrinsic(lhs: Term, kind: UnaryLogicIntrinsicKind)              extends Expr(Type.Bool)
    case BinaryLogicIntrinsic(lhs: Term, rhs: Term, kind: BinaryLogicIntrinsicKind) extends Expr(Type.Bool)

    case Cast(from: Term, as: Type)                                             extends Expr(as)
    case Alias(ref: Term)                                                       extends Expr(ref.tpe)
    case Invoke(name: Sym, receiver: Option[Term], args: List[Term], rtn: Type) extends Expr(rtn)
    case Index(lhs: Term.Select, idx: Term, component: Type)                    extends Expr(component)
    case Alloc(witness: Type.Array, size: Term)                                 extends Expr(witness)
  }

  enum Stmt derives MsgPack.Codec {
    case Comment(value: String)
    case Var(name: Named, expr: Option[Expr])
    case Mut(name: Term.Select, expr: Expr, copy: Boolean)
    case Update(lhs: Term.Select, idx: Term, value: Term)
    case While(cond: Expr, body: List[Stmt])
    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
  }

  case class StructDef(name: Sym, members: List[Named]) derives MsgPack.Codec

  case class Function(    //
      name: Sym,          //
      args: List[Named],  //
      rtn: Type,          //
      body: List[Stmt]    //
  ) derives MsgPack.Codec //

  case class Program(
      entry: Function,
      functions: List[Function],
      defs: List[StructDef]
  ) derives MsgPack.Codec

}
