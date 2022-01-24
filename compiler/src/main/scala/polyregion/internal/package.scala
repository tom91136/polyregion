package polyregion.internal

import cats.Eval
import cats.data.EitherT
import polyregion.ast.PolyAst

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag

// type VNil[+A] = scala.collection.immutable.Vector[A]
// val VNil = scala.collection.immutable.Vector

class CompilerException(m: String) extends Exception(m) {}

type Result[A] = Either[Throwable, A]

type Deferred[A] = EitherT[Eval, Throwable, A]

extension [A](a: Result[A]) {
  def deferred: Deferred[A]       = EitherT.fromEither[Eval](a)
  def withFilter(p: A => Boolean) = a.flatMap(x => (if (p(x)) Right(x) else Left(new MatchError(x))))
}

extension [A](a: Deferred[A]) {
  def resolve: Result[A]          = a.value.value
  def withFilter(p: A => Boolean) = a.subflatMap(x => (if (p(x)) Right(x) else Left(new MatchError(x))))
}

extension [A](a: A) {
  def success: Result[A] = Right(a)
  def pure: Deferred[A]  = Right(a).deferred
}
extension (message: => String) {
  def fail[A]: Result[A] = Left(new CompilerException(message))
}
extension [A](m: Option[A]) {
  def failIfEmpty(message: => String): Result[A] = m.fold(message.fail[A])(Right(_))
}
extension (e: => Throwable) {
  def failE[A]: Result[A] = Left(e)
}

extension (e: => PolyAst.Sym.type) {
  def apply[T <: AnyRef](using tag: ClassTag[T]): PolyAst.Sym = {
    // normalise naming differences
    // Java        => package.Companion$Member
    // Scala Macro => package.Companion$.Member
    @tailrec def go(cls: Class[_], xs: List[String] = Nil, companion: Boolean = false): List[String] = {
      val name = cls.getSimpleName + (if (companion) "$" else "")
      cls.getEnclosingClass match {
        case null => cls.getPackageName :: name :: xs
        case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
      }
    }
    PolyAst.Sym(go(tag.runtimeClass))
  }
}

extension (e: => PolyAst.Sym) {
  def repr: String = e.fqn.mkString(".")
}

extension (p: PolyAst.Named) {
  def repr: String = s"(${p.symbol}:${p.tpe.repr})"
}

extension (e: PolyAst.Term) {

  def repr: String = {
    import PolyAst.Term.*
    e match {
      case Select(init, last) => (init :+ last).map(_.repr).mkString(".")
      case UnitConst          => s"Unit()"
      case BoolConst(value)   => s"Bool($value)"
      case ByteConst(value)   => s"Byte($value)"
      case CharConst(value)   => s"Char($value)"
      case ShortConst(value)  => s"Short($value)"
      case IntConst(value)    => s"Int($value)"
      case LongConst(value)   => s"Long($value)"
      case FloatConst(value)  => s"Float($value)"
      case DoubleConst(value) => s"Double($value)"
      case StringConst(value) => s"String($value)"
    }
  }
}

extension (e: PolyAst.Type) {

  def repr: String = {
    import PolyAst.Type.*
    e match {
      case Struct(sym)    => s"Struct[${sym.repr}]"
      case Array(comp, n) => s"Array[${comp.repr}${n.map(x => s";$x").getOrElse("")}]"
      case Bool           => "Bool"
      case Byte           => "Byte"
      case Char           => "Char"
      case Short          => "Short"
      case Int            => "Int"
      case Long           => "Long"
      case Float          => "Float"
      case Double         => "Double"
      case String         => "String"
      case Unit           => "Unit"
    }
  }
}

extension (e: PolyAst.Expr) {
  def repr: String = {
    import PolyAst.Expr.*
    e match {

      case Sin(lhs, rtn) => s"sin(${lhs.repr})"
      case Cos(lhs, rtn) => s"cos(${lhs.repr})"
      case Tan(lhs, rtn) => s"tan(${lhs.repr})"
      case Abs(lhs, rtn) => s"abs(${lhs.repr})"

      case Add(lhs, rhs, rtn) => s"${lhs.repr} + ${rhs.repr}"
      case Sub(lhs, rhs, rtn) => s"${lhs.repr} - ${rhs.repr}"
      case Mul(lhs, rhs, rtn) => s"${lhs.repr} * ${rhs.repr}"
      case Div(lhs, rhs, rtn) => s"${lhs.repr} / ${rhs.repr}"
      case Rem(lhs, rhs, rtn) => s"${lhs.repr} % ${rhs.repr}"

      case Pow(lhs, rhs, rtn) => s"${lhs.repr} ** ${rhs.repr}"

      case BNot(lhs, _)      => s"~${lhs.repr}"
      case BAnd(lhs, rhs, _) => s"${lhs.repr} & ${rhs.repr}"
      case BOr(lhs, rhs, _)  => s"${lhs.repr} | ${rhs.repr}"
      case BXor(lhs, rhs, _) => s"${lhs.repr} ^ ${rhs.repr}"
      case BSL(lhs, rhs, _)  => s"${lhs.repr} < <${rhs.repr}"
      case BSR(lhs, rhs, _)  => s"${lhs.repr} > >${rhs.repr}"

      case Not(lhs)      => s"!(${lhs.repr})"
      case Eq(lhs, rhs)  => s"${lhs.repr} == ${rhs.repr}"
      case Neq(lhs, rhs) => s"${lhs.repr} != ${rhs.repr}"
      case And(lhs, rhs) => s"${lhs.repr} && ${rhs.repr}"
      case Or(lhs, rhs)  => s"${lhs.repr} || ${rhs.repr}"
      case Lte(lhs, rhs) => s"${lhs.repr} <= ${rhs.repr}"
      case Gte(lhs, rhs) => s"${lhs.repr} >= ${rhs.repr}"
      case Lt(lhs, rhs)  => s"${lhs.repr} < ${rhs.repr}"
      case Gt(lhs, rhs)  => s"${lhs.repr} > ${rhs.repr}"

      case Alias(ref)                   => s"(~>${ref.repr})"
      case Invoke(lhs, name, args, tpe) => s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : ${tpe.repr}"
      case Index(lhs, idx, tpe)         => s"${lhs.repr}[${idx.repr}] : ${tpe.repr}"
      // case Block(xs, x)                 => s"{\n${xs.map(_.repr).mkString("\n")}\n${x.repr}\n}"
    }
  }
}

extension (e: PolyAst.Stmt) {
  def repr: String = {
    import PolyAst.Stmt.*
    e match {
      case Comment(value)          => s"// $value"
      case Var(name, rhs)          => s"var ${name.repr} = ${rhs.fold("_")(_.repr)}"
      case Mut(name, expr)         => s"${name.repr} := ${expr.repr}"
      case Update(lhs, idx, value) => s"${lhs.repr}[${idx.repr}] := ${value.repr}"
      case Effect(lhs, name, args) => s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : Unit"
      case While(cond, body)       => s"while(${cond.repr}){\n${body.map(_.repr).mkString("\n")}\n}"
      case Break                   => s"break;"
      case Cont                    => s"continue;"
      case Return(expr)            => s"return ${expr.repr}"
      case Cond(cond, trueBr, falseBr) =>
        s"if(${cond.repr}) {\n${trueBr.map("  " + _.repr).mkString("\n")}\n} else {\n${falseBr.map("  " + _.repr).mkString("\n")}\n}"
    }

  }
}
