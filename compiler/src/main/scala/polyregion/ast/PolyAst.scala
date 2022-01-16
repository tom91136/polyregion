package polyregion.ast

import polyregion.data.MsgPack

object PolyAst {

  case class Sym(fqn: List[String]) derives MsgPack.Codec {
    def repr: String = fqn.mkString(".")
  }
  object Sym {
    def apply(raw: String): Sym = {
      require(!raw.isBlank)
      // normalise dollar
      Sym(raw.split('.').toList)
    }
  }

  enum TypeKind derives MsgPack.Codec {
    case Ref, Integral, Fractional
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

    case String                              extends Type(TypeKind.Ref)
    case Unit                                extends Type(TypeKind.Ref)
    case Struct(name: Sym, args: List[Type]) extends Type(TypeKind.Ref)
    case Array(component: Type)              extends Type(TypeKind.Ref)

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

  enum Expr(tpe: Type) derives MsgPack.Codec {
    case Sin(lhs: Term, rtn: Type) extends Expr(rtn)
    case Cos(lhs: Term, rtn: Type) extends Expr(rtn)
    case Tan(lhs: Term, rtn: Type) extends Expr(rtn)

    case Add(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)
    case Sub(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)
    case Mul(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)
    case Div(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)
    case Mod(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)
    case Pow(lhs: Term, rhs: Term, rtn: Type) extends Expr(rtn)

    case Inv(lhs: Term)            extends Expr(Type.Bool)
    case Eq(lhs: Term, rhs: Term)  extends Expr(Type.Bool)
    case Lte(lhs: Term, rhs: Term) extends Expr(Type.Bool)
    case Gte(lhs: Term, rhs: Term) extends Expr(Type.Bool)
    case Lt(lhs: Term, rhs: Term)  extends Expr(Type.Bool)
    case Gt(lhs: Term, rhs: Term)  extends Expr(Type.Bool)

    case Alias(ref: Term)                                             extends Expr(ref.tpe)
    case Invoke(lhs: Term, name: String, args: List[Term], rtn: Type) extends Expr(rtn)
    case Index(lhs: Term.Select, idx: Term, component: Type)          extends Expr(component)
  }

  enum Stmt derives MsgPack.Codec {
    case Comment(value: String)
    case Var(name: Named, expr: Expr)
    case Mut(name: Term.Select, expr: Expr)
    case Update(lhs: Term.Select, idx: Term, value: Term)
    case Effect(lhs: Term.Select, name: String, args: List[Term])
    case While(cond: Expr, body: List[Stmt])
    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
  }

  case class Function(name: String, args: List[Named], rtn: Type, body: List[Stmt]) derives MsgPack.Codec

  case class StructDef(
      members: List[Named]
      // TODO methods
  ) derives MsgPack.Codec

}
