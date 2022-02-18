package polyregion

import cats.Eval
import cats.data.EitherT
import polyregion.ast.PolyAst as p

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag

class CompilerException(m: String) extends Exception(m)

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

extension (e: => p.Sym.type) {

  def apply[T](using tag: ClassTag[T]): p.Sym = apply(tag.runtimeClass)

  def apply[T](cls: Class[T]): p.Sym = {
    // normalise naming differences
    // Java        => package.Companion$Member
    // Scala Macro => package.Companion$.Member
    @tailrec def go(cls: Class[_], xs: List[String] = Nil, companion: Boolean = false): List[String] = {
      val name = cls.getSimpleName + (if (companion) "$" else "")
      cls.getEnclosingClass match {
        case null => cls.getPackageName.split("\\.").toList ::: name :: xs
        case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
      }
    }
    p.Sym(go(cls))
  }

}

extension (e: p.Sym) {
  def repr: String = e.fqn.mkString(".")
}

extension (n: p.Named) {
  def repr: String = s"(${n.symbol}:${n.tpe.repr})"
}

extension (e: p.Term) {
  def repr: String = e match {
    case p.Term.Select(init, last) => (init :+ last).map(_.repr).mkString(".")
    case p.Term.UnitConst          => s"Unit()"
    case p.Term.BoolConst(value)   => s"Bool($value)"
    case p.Term.ByteConst(value)   => s"Byte($value)"
    case p.Term.CharConst(value)   => s"Char($value)"
    case p.Term.ShortConst(value)  => s"Short($value)"
    case p.Term.IntConst(value)    => s"Int($value)"
    case p.Term.LongConst(value)   => s"Long($value)"
    case p.Term.FloatConst(value)  => s"Float($value)"
    case p.Term.DoubleConst(value) => s"Double($value)"
    case p.Term.StringConst(value) => s"String($value)"
  }
}

extension (e: p.Type) {
  def repr: String = e match {
    case p.Type.Struct(sym) => s"Struct[${sym.repr}]"
    case p.Type.Array(comp) => s"Array[${comp.repr}]"
    case p.Type.Bool        => "Bool"
    case p.Type.Byte        => "Byte"
    case p.Type.Char        => "Char"
    case p.Type.Short       => "Short"
    case p.Type.Int         => "Int"
    case p.Type.Long        => "Long"
    case p.Type.Float       => "Float"
    case p.Type.Double      => "Double"
    case p.Type.String      => "String"
    case p.Type.Unit        => "Unit"
  }
}

extension (e: p.Expr) {
  def repr: String = e match {
    case p.Expr.UnaryIntrinsic(lhs, kind, rtn) =>
      val fn = kind match {
        case p.UnaryIntrinsicKind.Sin  => "sin"
        case p.UnaryIntrinsicKind.Cos  => "cos"
        case p.UnaryIntrinsicKind.Tan  => "tan"
        case p.UnaryIntrinsicKind.Abs  => "abs"
        case p.UnaryIntrinsicKind.BNot => "~"
      }
      s"$fn(${lhs.repr})"
    case p.Expr.UnaryLogicIntrinsic(lhs, kind) =>
      val fn = kind match {
        case p.UnaryLogicIntrinsicKind.Not => "!"
      }
      s"$fn(${lhs.repr})"
    case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) =>
      val op = kind match {
        case p.BinaryIntrinsicKind.Add  => s"+"
        case p.BinaryIntrinsicKind.Sub  => s"-"
        case p.BinaryIntrinsicKind.Mul  => s"*"
        case p.BinaryIntrinsicKind.Div  => s"/"
        case p.BinaryIntrinsicKind.Rem  => s"%"
        case p.BinaryIntrinsicKind.Pow  => s"**"
        case p.BinaryIntrinsicKind.BAnd => s"&"
        case p.BinaryIntrinsicKind.BOr  => s"|"
        case p.BinaryIntrinsicKind.BXor => s"^"
        case p.BinaryIntrinsicKind.BSL  => s"<<"
        case p.BinaryIntrinsicKind.BSR  => s">>"
      }
      s"${lhs.repr} ${op} ${rhs.repr}"

    case p.Expr.BinaryLogicIntrinsic(lhs, rhs, kind) =>
      val op = kind match {
        case p.BinaryLogicIntrinsicKind.Eq  => "=="
        case p.BinaryLogicIntrinsicKind.Neq => "!="
        case p.BinaryLogicIntrinsicKind.And => "&&"
        case p.BinaryLogicIntrinsicKind.Or  => "||"
        case p.BinaryLogicIntrinsicKind.Lte => "<="
        case p.BinaryLogicIntrinsicKind.Gte => ">="
        case p.BinaryLogicIntrinsicKind.Lt  => "<"
        case p.BinaryLogicIntrinsicKind.Gt  => ">"
      }
      s"${lhs.repr} ${op} ${rhs.repr}"
    case p.Expr.Cast(from, to) => s"${from.repr}.to[${to.repr}]"
    case p.Expr.Alias(ref)     => s"(~>${ref.repr})"

    case p.Expr.Invoke(name, recv, args, tpe) =>
      s"${recv.map(_.repr).getOrElse("<module>")}.${name.repr}(${args.map(_.repr).mkString(",")}) : ${tpe.repr}"
    case p.Expr.Index(lhs, idx, tpe) => s"${lhs.repr}[${idx.repr}] : ${tpe.repr}"
    case p.Expr.Alloc(tpe, size)     => s"new [${tpe.component.repr}*${size.repr}]"
  }
}

extension (e: p.Stmt) {

  def mapAccExpr[A](f: p.Expr => (p.Expr, List[p.Stmt], List[A])): (List[p.Stmt], List[A]) = e match {
    case x @ p.Stmt.Comment(_) => (x :: Nil, Nil)
    case p.Stmt.Var(name, rhs) =>
      rhs match {
        case None => (p.Stmt.Var(name, None) :: Nil, Nil)
        case Some(x) =>
          val (y, xs, as) = f(x)
          (xs :+ p.Stmt.Var(name, Some(y)), as)
      }
    case p.Stmt.Mut(name, expr, copy) => val (y, xs, as) = f(expr); (xs :+ p.Stmt.Mut(name, y, copy), as)
    case x @ p.Stmt.Update(_, _, _)   => (x :: Nil, Nil)
    case p.Stmt.While(cond, body) =>
      val (y, xs, as) = f(cond)
      val (bss, bass) = body.map(_.mapAccExpr(f)).unzip
      (xs :+ p.Stmt.While(y, bss.flatten), (as :: bass).flatten)
    case x @ p.Stmt.Break => (x :: Nil, Nil)
    case x @ p.Stmt.Cont  => (x :: Nil, Nil)
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      val (y, xs, as) = f(cond)
      val (tss, tass) = trueBr.map(_.mapAccExpr(f)).unzip
      val (fss, fass) = falseBr.map(_.mapAccExpr(f)).unzip
      (xs :+ p.Stmt.Cond(y, tss.flatten, fss.flatten), (as :: tass ::: fass).flatten)
    case p.Stmt.Return(expr) => val (y, xs, as) = f(expr); (xs :+ p.Stmt.Return(y), as)
  }

  def mapExpr(f: p.Expr => (p.Expr, List[p.Stmt])): List[p.Stmt] = e.mapAccExpr[Unit](f(_) ++ Nil *: EmptyTuple)._1

  def mapTerm(g: p.Term.Select => p.Term.Select, f: p.Term => p.Term): List[p.Stmt] = e
    .map {
      case p.Stmt.Mut(name, expr, copy)   => p.Stmt.Mut(g(name), expr, copy) :: Nil
      case p.Stmt.Update(lhs, idx, value) => p.Stmt.Update(g(lhs), f(idx), f(value)) :: Nil
      case x                              => x :: Nil
    }
    .flatMap(_.mapExpr {
      case p.Expr.UnaryIntrinsic(lhs, kind, rtn)       => (p.Expr.UnaryIntrinsic(f(lhs), kind, rtn), Nil)
      case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) => (p.Expr.BinaryIntrinsic(f(lhs), f(rhs), kind, rtn), Nil)

      case p.Expr.UnaryLogicIntrinsic(lhs, kind)       => (p.Expr.UnaryLogicIntrinsic(f(lhs), kind), Nil)
      case p.Expr.BinaryLogicIntrinsic(lhs, rhs, kind) => (p.Expr.BinaryLogicIntrinsic(f(lhs), f(rhs), kind), Nil)

      case p.Expr.Cast(from, to) => (p.Expr.Cast(f(from), to), Nil)
      case p.Expr.Alias(ref) =>
        val h = ref match {
          // case x @ p.Term.Select(_, _) => g(x)
          case x => f(x)
        }
        (p.Expr.Alias(h), Nil)
      case p.Expr.Invoke(name, receiver, args, rtn) => (p.Expr.Invoke(name, receiver.map(f), args.map(f), rtn), Nil)
      case p.Expr.Index(lhs, idx, component)        => (p.Expr.Index(g(lhs), f(idx), component), Nil)
      // case p.Expr.MkArray(elem)                     => (p.Expr.MkArray(elem), Nil)

    })

  def mapAcc[A](f: p.Stmt => (List[p.Stmt], List[A])): (List[p.Stmt], List[A]) = e match {
    case x @ p.Stmt.Comment(_)      => f(x)
    case x @ p.Stmt.Var(_, _)       => f(x)
    case x @ p.Stmt.Mut(_, _, _)    => f(x)
    case x @ p.Stmt.Update(_, _, _) => f(x)
    case p.Stmt.While(cond, body) =>
      val (sss0, xss) = body.map(_.mapAcc(f)).unzip
      val (sss1, yss) = f(p.Stmt.While(cond, sss0.flatten))
      (sss1, xss.flatten ::: yss)
    case x @ p.Stmt.Break => f(x)
    case x @ p.Stmt.Cont  => f(x)
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      val (tss, tass) = trueBr.map(_.mapAcc(f)).unzip
      val (fss, fass) = falseBr.map(_.mapAcc(f)).unzip
      val (ss, ass)   = f(p.Stmt.Cond(cond, tss.flatten, fss.flatten))
      (ss, (tass ::: fass).flatten ::: ass)
    case x @ p.Stmt.Return(_) => f(x)
  }

  def acc[A](f: p.Stmt => List[A]): List[A] = e.mapAcc[A](x => (x :: Nil, f(x)))._2

  def map(f: p.Stmt => List[p.Stmt]): List[p.Stmt] = e.mapAcc[Unit](x => (f(x), Nil))._1

  def repr: String = e match {
    case p.Stmt.Comment(value)          => s" /* $value */"
    case p.Stmt.Var(name, rhs)          => s"var ${name.repr} = ${rhs.fold("_")(_.repr)}"
    case p.Stmt.Mut(name, expr, copy)   => s"${name.repr} ${if (copy) ":=!" else ":="} ${expr.repr}"
    case p.Stmt.Update(lhs, idx, value) => s"${lhs.repr}[${idx.repr}] := ${value.repr}"
    case p.Stmt.While(cond, body)       => s"while(${cond.repr}){\n${body.map(_.repr).mkString("\n")}\n}"
    case p.Stmt.Break                   => s"break;"
    case p.Stmt.Cont                    => s"continue;"
    case p.Stmt.Return(expr)            => s"return ${expr.repr}"
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      s"if(${cond.repr}) {\n${trueBr.map("  " + _.repr).mkString("\n")}\n} else {\n${falseBr.map("  " + _.repr).mkString("\n")}\n}"
  }

}

extension (f: p.Function) {
  def repr: String =
    s"""${f.name.repr}(${f.args.map(_.repr).mkString(", ")}) : ${f.rtn.repr} = {
       |${f.body.map(_.repr).map("  " + _).mkString("\n")}
		   |}""".stripMargin
}
