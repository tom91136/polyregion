package polyregion

object PolyAstUnused {
  // struct Sym { std::vector<std::string>> data } ;

  case class Sym(fqn: List[String]) {
    def repr: String = fqn.mkString(".")
  }
  object Sym {
    def apply(raw: String): Sym = {
      require(!raw.isBlank)
      // normalise dollar
      Sym(raw.split('.').toList)
    }
  }

  enum Kind {
    case Ref, Integral, Fractional
  }

  enum Type(val kind: Kind) {
    case Float extends Type(Kind.Fractional)
    case Double extends Type(Kind.Fractional)

    case Bool extends Type(Kind.Integral)
    case Byte extends Type(Kind.Integral)
    case Char extends Type(Kind.Integral)
    case Short extends Type(Kind.Integral)
    case Int extends Type(Kind.Integral)
    case Long extends Type(Kind.Integral)

    case Struct(name: Sym, args: List[Type]) extends Type(Kind.Ref)
    case Array(component: Type) extends Type(Kind.Ref)
    case String extends Type(Kind.Ref)
    case Unit extends Type(Kind.Ref)
  }

  case class Named(symbol: String, tpe: Type)

  // struct RefBase { Type type; }
  // struct BoolConst : RefBase {
  //   bool value;
  //   explicit BootConst(bool value) : RefBase {Type::Fractional::Float}
  // }

  case class N(z: String)

  enum U(val N: String) {
    case X(n: N) extends U(n.z)
  }

  enum Ref(val tpe: Type) {
    case Select(init: List[Named], last: Named) extends Ref(Type.Bool)
    case BoolConst(value: Boolean) extends Ref(Type.Bool)
    // case ByteConst(value: Byte) extends Ref(Type.Byte)
    // case CharConst(value: Char) extends Ref(Type.Char)
    // case ShortConst(value: Short) extends Ref(Type.Short)
    // case IntConst(value: Int) extends Ref(Type.Int)
    // case LongConst(value: Long) extends Ref(Type.Long)
    // case FloatConst(value: Float) extends Ref(Type.Float)
    // case DoubleConst(value: Double) extends Ref(Type.Double)
    // case StringConst(value: String) extends Ref(Type.String)
  }

  case class Position(file: String, line: Int, col: Int)

  sealed abstract class Tree(val tpe: Type)

  enum Intr(tpe: Type) extends Tree(tpe) {
    case Inv(lhs: Ref, rtn: Type) extends Intr(rtn)
    case Sin(lhs: Ref, rtn: Type) extends Intr(rtn)
    case Cos(lhs: Ref, rtn: Type) extends Intr(rtn)
    case Tan(lhs: Ref, rtn: Type) extends Intr(rtn)

    case Add(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)
    case Sub(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)
    case Div(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)
    case Mul(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)
    case Mod(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)
    case Pow(lhs: Ref, rhs: Ref, rtn: Type) extends Intr(rtn)

  }

  // Expr
  //  Alias {  ref : Ref  }
  enum Expr(tpe: Type) extends Tree(tpe) {
    case Alias(ref: Ref) extends Expr(ref.tpe)
    case Invoke(lhs: Ref, name: String, args: List[Ref], rtn: Type) extends Expr(rtn)
    case Index(lhs: Ref.Select, idx: Ref, component: Type) extends Expr(component)
  }

  enum Stmt extends Tree(Type.Unit) {
    case Comment(value: String)
    case Var(name: Named, rhs: Expr)
    case Mut(name: Ref.Select, expr: Expr)
    case Update(lhs: Ref.Select, idx: Ref, value: Ref)
    case Effect(lhs: Ref.Select, name: String, args: List[Ref])
    case While(cond: Expr, body: List[Stmt])
    case Break
    case Cont
    case Cond(cond: Expr, trueBr: List[Stmt], falseBr: List[Stmt])
    case Return(value: Expr)
  }

  case class Function(name: String, args: List[Named], rtn: Type, body: List[Stmt])

  //    case class Type(sym: Sym, args: List[Type]) {
  //      def repr: String = args match {
  //        case Nil => sym.repr
  //        case xs  => s"${sym.repr}[${xs.map(_.repr).mkString(",")}]"
  //      }
  //      def args(xs: Type*): Type = copy(args = xs.toList)
  //      def ctor: Type            = copy(args = Nil)
  //    }
  //    object Type {
  //
  //      def apply[T <: AnyRef](using tag: ClassTag[T]): Type = {
  //        // normalise naming differences
  //        // Java        => package.Companion$Member
  //        // Scala Macro => package.Companion$.Member
  //        @tailrec def go(cls: Class[_], xs: List[String] = Nil, companion: Boolean = false): List[String] = {
  //          val name = cls.getSimpleName + (if (companion) "$" else "")
  //          cls.getEnclosingClass match {
  //            case null => cls.getPackageName :: name :: xs
  //            case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
  //          }
  //        }
  //        Type(Sym(go(tag.runtimeClass)), Nil)
  //      }
  //
  //      // XXX we can't do [T: ClassTag] becase it resolves to the unboxed class
  //      def apply(name: String): Type = try {
  //        Class.forName(name) // resolve it first to make sure it's actually there
  //        Type(Sym(name), Nil)
  //      } catch { t => throw new AssertionError(s"Cannot resolve ${name} for Type constant: ${t.getMessage}") }
  //    }

  case class StructDef(
      members: List[(String, Type)]
      //TODO methods
  )

  //    object Primitives {
  //      val Unit    = Type("scala.Unit")
  //      val Boolean = Type("scala.Boolean")
  //      val Byte    = Type("scala.Byte")
  //      val Short   = Type("scala.Short")
  //      val Int     = Type("scala.Int")
  //      val Long    = Type("scala.Long")
  //      val Float   = Type("scala.Float")
  //      val Double  = Type("scala.Double")
  //      val Char    = Type("scala.Char")
  //      val String  = Type("java.lang.String")
  //      val All     = List(Unit, Boolean, Byte, Short, Int, Long, Float, Double, Char, String)
  //    }
  //
  //    object Intrinsics {
  //      val Buffer       = Type[Buffer[_]]
  //      val DoubleBuffer = Buffer.args(Primitives.Double)
  //      val FloatBuffer  = Buffer.args(Primitives.Float)
  //      val LongBuffer   = Buffer.args(Primitives.Long)
  //      val IntBuffer    = Buffer.args(Primitives.Int)
  //      val ShortBuffer  = Buffer.args(Primitives.Short)
  //      val ByteBuffer   = Buffer.args(Primitives.Byte)
  //      val CharBuffer   = Buffer.args(Primitives.Char)
  //
  //    }

  //    case class Path(name: String, tpe: Type) {
  //      def repr: String = s"($name:${tpe.repr})"
  //    }

  /*

  struct Ref{
	virtual bar::Type tpe();

  }
  struct Select : Ref {

  }




   */

  //    enum Ref(show: => String, val tpe: Type) {
  //      case Select(head: Path, tail: List[Path] = Nil)
  //          extends Ref((head :: tail).map(_.repr).mkString("."), tail.lastOption.getOrElse(head).tpe)
  //
  //      case ByteConst(value: Byte) extends Ref(s"Byte(`$value)`", Primitives.Byte)
  //      case CharConst(value: Char) extends Ref(s"Char(`$value`)", Primitives.Char)
  //      case ShortConst(value: Short) extends Ref(s"Short(`$value)`", Primitives.Short)
  //      case IntConst(value: Int) extends Ref(s"Int(`$value)`", Primitives.Int)
  //      case LongConst(value: Long) extends Ref(s"Long(`$value)`", Primitives.Long)
  //
  //      case FloatConst(value: Float) extends Ref(s"Float(`$value)`", Primitives.Float)
  //      case DoubleConst(value: Double) extends Ref(s"Double(`$value)`", Primitives.Double)
  //
  //      case BoolConst(value: Boolean) extends Ref(s"Boolean(`$value)`", Primitives.Boolean)
  //
  //      case StringConst(value: String) extends Ref(s"String(`$value`)", Primitives.String)
  //      case UnitConst() extends Ref("()", Primitives.Unit)
  //      case NullConst(resolved: Type)
  //          extends Ref(s"(null: ${resolved.repr})", resolved) // null is Nothing which will be concrete after Typer?
  //      def repr: String              = show
  //      override def toString: String = repr
  //
  //    }
  //
  //    sealed trait Tree {
  //      def repr: String
  //      override def toString: String = repr
  //    }
  //
  //    enum Expr(show: => String, tpe: Type) extends Tree {
  //      case Alias(ref: Ref) extends Expr(s"(~>${ref.repr})", ref.tpe)
  //      case Invoke(lhs: Ref, name: String, args: Vector[Ref], tpe: Type)
  //          extends Expr(s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")})", tpe)
  //      def repr: String = show
  //    }
  //
  //    // enum
  //
  //    enum Stmt(show: => String) extends Tree {
  //      case Comment(value: String) extends Stmt(s" // $value") // discard at backend
  //
  //      case Var(key: String, tpe: Type, rhs: Expr) extends Stmt(s"var $key : ${tpe.repr} = ${rhs.repr}")
  //      case Effect(lhs: Ref, name: String, args: Vector[Ref])
  //          extends Stmt(s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : Unit")
  //      case Mut(lhs: Ref, ref: Expr) extends Stmt(s"${lhs.repr} := ${ref.repr}")
  //      case While(cond: Expr, body: Vector[Tree])
  //          extends Stmt(s"while(${cond.repr}{\n${body.map(_.repr).mkString("\n")}\n}")
  ////      case Block(exprs: List[Tree]) extends Stmt(exprs.map(_.repr).mkString("{\n", "\n", "\n}"))
  //      def repr: String = show
  //    }

}
