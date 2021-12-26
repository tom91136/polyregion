package polyregion

import cats.Eval
import cats.data.EitherT
import polyregion.PolyAst

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag

package object internal {

  type VNil[+A] = scala.collection.immutable.Vector[A]
  val VNil = scala.collection.immutable.Vector

  type Result[A] = Either[Throwable, A]

  type Deferred[A] = EitherT[Eval, Throwable, A]

  extension [A](a: Result[A]) {
    def deferred: Deferred[A] = EitherT.fromEither[Eval](a)
  }

  extension [A](a: Deferred[A]) {
    def resolve: Result[A]          = a.value.value
    def withFilter(p: A => Boolean) = a.subflatMap(x => (if (p(x)) Right(x) else Left(new MatchError(x))))
  }

  extension [A](a: A) {
    def success: Result[A] = Right(a)
  }
  extension (message: => String) {
    def fail[A]: Result[A] = Left(new Exception(message))
  }
  extension (e: => Throwable) {
    def failE[A]: Result[A] = Left(e)
  }

  extension (e: => PolyAst.Sym.type) {
    def apply(raw: String): PolyAst.Sym = {
      require(!raw.isBlank)
      // normalise dollar
      PolyAst.Sym(raw.split('.').toVector)
    }
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
      PolyAst.Sym(go(tag.runtimeClass).toVector)
    }
  }

  extension (e: => PolyAst.Sym) {
    def repr: String = e.fqn.mkString(".")
  }

  extension (p: PolyAst.Named) {
    def repr: String = s"(${p.symbol}:${p.tpe.repr})"
  }

  extension (e: PolyAst.Refs.Ref) {

    def tpe: PolyAst.Types.Type = {
      import polyregion.PolyAst.Refs.*
      e match {
        case Select(head, tail) => tail.lastOption.getOrElse(head).tpe
        case BoolConst(value)   => PolyAst.Types.BoolTpe()
        case ByteConst(value)   => PolyAst.Types.ByteTpe()
        case CharConst(value)   => PolyAst.Types.CharTpe()
        case ShortConst(value)  => PolyAst.Types.ShortTpe()
        case IntConst(value)    => PolyAst.Types.IntTpe()
        case LongConst(value)   => PolyAst.Types.LongTpe()
        case FloatConst(value)  => PolyAst.Types.FloatTpe()
        case DoubleConst(value) => PolyAst.Types.DoubleTpe()
        case StringConst(value) => PolyAst.Types.StringTpe()
        case Ref.Empty          => PolyAst.Types.Type.Empty
      }
    }

    def repr: String = {
      import polyregion.PolyAst.Refs.*
      e match {
        case Select(head, tail) => (head +: tail).map(_.repr).mkString(".")
        case BoolConst(value)   => s"Bool($value)"
        case ByteConst(value)   => s"Byte($value)"
        case CharConst(value)   => s"Char($value)"
        case ShortConst(value)  => s"Short($value)"
        case IntConst(value)    => s"Int($value)"
        case LongConst(value)   => s"Long($value)"
        case FloatConst(value)  => s"Float($value)"
        case DoubleConst(value) => s"Double($value)"
        case StringConst(value) => s"String($value)"
        case Ref.Empty          => "Unit()"
      }

    }
  }

  extension (e: PolyAst.Types.Type) {

    def repr: String = {
      import polyregion.PolyAst.Types.*
      e match {
        case RefTpe(head, Vector()) => head.repr
        case RefTpe(head, xs)       => s"${head.repr}[${xs.map(_.repr).mkString(",")}]"
        //
        case ArrayTpe(tpe) => s"Array[${tpe.repr}]"
        case BoolTpe()     => "Bool"
        case ByteTpe()     => "Byte"
        case CharTpe()     => "Char"
        case ShortTpe()    => "Short"
        case IntTpe()      => "Int"
        case LongTpe()     => "Long"
        case FloatTpe()    => "Float"
        case DoubleTpe()   => "Double"
        case StringTpe()   => "String"
        case Type.Empty    => "Unit"
      }

    }
  }

  extension (e: PolyAst.Tree.Expr) {
    def repr: String = {
      import polyregion.PolyAst.Tree.*
      e match {
        case Alias(ref)                   => s"(~>${ref.repr})"
        case Invoke(lhs, name, args, tpe) => s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : ${tpe.repr}"
        case Index(lhs, idx, tpe)         => s"${lhs.repr}[${idx.repr}] : ${tpe.repr}"
        // case Block(xs, x)                 => s"{\n${xs.map(_.repr).mkString("\n")}\n${x.repr}\n}"
        case Expr.Empty => "(empty expr)"
      }
    }
  }

  extension (e: PolyAst.Tree.Stmt) {
    def repr: String = {
      import polyregion.PolyAst.Tree.*
      e match {
        case Comment(value)              => s" // $value"
        case Var(name, rhs)              => s"var ${name.repr} = ${rhs.repr}"
        case Mut(name, expr)             => s"${name.repr} := ${expr.repr}"
        case Update(lhs, idx, value)     => s"${lhs.repr}[${idx.repr}] := ${value.repr}"
        case Effect(lhs, name, args)     => s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : Unit"
        case While(cond, body)           => s"while(${cond.repr}){\n${body.map(_.repr).mkString("\n")}\n}"
        case Break()                     => s"break;"
        case Cond(cond, trueBr, falseBr) => s"if(${cond.repr}) {\n${trueBr.repr}\n} else {\n${falseBr}\n}"
        case Stmt.Empty                  => "(empty stmt)"
      }

    }
  }

}
