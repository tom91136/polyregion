package polyregion.ast

import cats.kernel.Semigroup
import polyregion.ast.PolyAst as p
import polyregion.ast.PolyAst.TypeKind

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag
import scala.util.{Success, Try}

@tailrec def doUntilNotEq[A](x: A)(f: A => A): A = {
  val y = f(x)
  if (y == x) y
  else doUntilNotEq(y)(f)
}

final class CompilerException(m: String, e: Throwable) extends Exception(m, e) {
  def this(s: String) = this(s, null)
}

type Result[A] = Either[Throwable, A]

// type Deferred[A] = Result[A] //  cats.data.EitherT[cats.Eval, Throwable, A]

extension [A](a: Result[A]) {
//  def deferred: Deferred[A]       = cats.data.EitherT.fromEither[cats.Eval](a)
  def withFilter(p: A => Boolean) = a.flatMap(x => if (p(x)) Right(x) else Left(new MatchError(x)))
}

//
case class Log(name: String, lines: Vector[(String, Vector[String]) | Log]) {

  // def mark[A](name: String)(f: Log => Result[(A, Log)]): Result[(A, Log)] =
  //   f(Log(name, Vector.empty)).map { case (x, l) => (x, copy(lines = lines :+ l)) }

  infix def +(log: Log): Log = copy(lines = lines :+ log)

  infix def ~+(log: Log): Result[Log] = copy(lines = lines :+ log).success

  infix def ++(log: Seq[Log]): Log = copy(lines = lines ++ log)

  def info_(message: String, details: String*): Log = copy(lines = lines :+ (message -> details.toVector))

  def info(message: String, details: String*): Result[Log] = info_(message, details*).success

  def render(nesting: Int = 0): Vector[String] =
    Try {

      val colour = Log.Colours(nesting % Log.Colours.size)
      val attr   = colour ++ fansi.Reversed.On ++ fansi.Bold.On
      val indent = colour("┃ ")

      ((colour("┏━") ++ attr(s" ${name} ") ++ colour("")) +: lines
        .flatMap {
          case (log: Log) => log.render(nesting + 1).map(indent ++ _)
          case (l, details) =>
            ((colour ++ fansi.Underlined.On)(s"▓ $l ▓")) +: details.flatMap { l =>
              l.linesIterator.toList match {
                case x :: xs =>
                  ((colour("┃ ╰ ") ++ s"$x") :: xs.map(x => indent ++ s"  ${x}")).toVector
                case Nil => Vector()
              }

            }
        } :+ colour(s"┗━${"━" * (name.size + 2)}"))
        .map(_.render)
    }.recover { case e: Exception =>
      Vector(s"Cannot render:${e}")
    }.get

}
object Log {
  private val Colours: Vector[fansi.Attr] = Vector(
    // fansi.Color.Red,
    fansi.Color.Green,
    fansi.Color.Yellow,
    fansi.Color.Blue,
    fansi.Color.Magenta,
    fansi.Color.Cyan,
    fansi.Color.LightGray,
    fansi.Color.DarkGray,
    fansi.Color.LightRed,
    fansi.Color.LightGreen,
    fansi.Color.LightYellow,
    fansi.Color.LightBlue,
    fansi.Color.LightMagenta,
    fansi.Color.LightCyan
  )

  def apply(name: String): Result[Log] = Log(name, Vector.empty).success

}

extension [A](a: A) {
  def success: Result[A] = Right(a)
//  def pure: Deferred[A]  = Right(a).deferred
}
extension (message: => String) {
  def fail[A]: Result[A] = Left(new CompilerException(message))
  def indent_(n: Int)    = message.linesIterator.map(x => " " * n + x).mkString("\n")
}
extension [A](m: Option[A]) {
  def failIfEmpty(message: => String): Result[A] = m.fold(message.fail[A])(Right(_))
}
extension [A](m: List[A]) {
  def failIfNotSingleton(message: => String): Result[A] = m match {
    case x :: Nil => Right(x)
    case xs       => message.fail[A]
  }
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
        case null => cls.getPackage.getName.split("\\.").toList ::: name :: xs
        case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
      }
    }
    p.Sym(go(cls))
  }

}

object PolyAstToExpr {

  import scala.quoted.*

  given SymToExpr: ToExpr[p.Sym] with { def apply(x: p.Sym)(using Quotes) = '{ p.Sym(${ Expr(x.fqn) }) } }
  given NamedToExpr: ToExpr[p.Named] with {
    def apply(x: p.Named)(using Quotes) = '{ p.Named(${ Expr(x.symbol) }, ${ Expr(x.tpe) }) }
  }
  given StructDefToExpr: ToExpr[p.StructDef] with {
    def apply(x: p.StructDef)(using Quotes) = '{
      p.StructDef(${ Expr(x.name) }, ${ Expr(x.tpeVars) }, ${ Expr(x.members) })
    }
  }
  given TypeToExpr: ToExpr[p.Type] with {
    def apply(x: p.Type)(using Quotes) = x match {
      case p.Type.Var(_) => ???
      case p.Type.Float  => '{ p.Type.Float }
      case p.Type.Double => '{ p.Type.Double }
      case p.Type.Bool   => '{ p.Type.Bool }
      case p.Type.Byte   => '{ p.Type.Byte }
      case p.Type.Char   => '{ p.Type.Char }
      case p.Type.Short  => '{ p.Type.Short }
      case p.Type.Int    => '{ p.Type.Int }
      case p.Type.Long   => '{ p.Type.Long }
      case p.Type.Unit   => '{ p.Type.Unit }
      case p.Type.String => '{ p.Type.String }
      case p.Type.Struct(name, tpeVars, args) =>
        '{ p.Type.Struct(${ Expr(name) }, ${ Expr(tpeVars) }, ${ Expr(args) }) }
      case p.Type.Array(component) => '{ p.Type.Array(${ Expr(component) }) }
    }
  }

}

// extension (e: p.Expr.Invoke) {
//   def signature: p.Signature = p.Signature(e.name, e.tpeArgs, e.receiver.map(_.tpe), e.args.map(_.tpe), e.rtn)
// }

extension (sd: p.StructDef) {
  def repr: String = s"${sd.name.repr}<${sd.tpeVars.mkString(",")}> { ${sd.members.map(_.repr).mkString("; ")} }"
}

extension (e: p.Sym) {
  def repr: String = e.fqn.mkString(".")
}

extension (n: p.Named) {
  def repr: String                          = s"(${n.symbol}: ${n.tpe.repr})"
  def mapType(f: p.Type => p.Type): p.Named = p.Named(n.symbol, n.tpe.map(f))
  def mapAccType[A](f: p.Type => (p.Type, List[A])): (p.Named, List[A]) = {
    val (tpe, as) = n.tpe.mapAcc[A](f)
    (p.Named(n.symbol, tpe), as)
  }
}

extension (e: p.Term) {
  def repr: String = e match {
    case p.Term.Select(xs, x)  => (xs :+ x).map(_.repr).mkString(".")
    case p.Term.Poison(t)      => s"Poison($t)"
    case p.Term.UnitConst      => s"Unit()"
    case p.Term.BoolConst(x)   => s"Bool($x)"
    case p.Term.ByteConst(x)   => s"Byte($x)"
    case p.Term.CharConst(x)   => s"Char($x)"
    case p.Term.ShortConst(x)  => s"Short($x)"
    case p.Term.IntConst(x)    => s"Int($x)"
    case p.Term.LongConst(x)   => s"Long($x)"
    case p.Term.FloatConst(x)  => s"Float($x)"
    case p.Term.DoubleConst(x) => s"Double($x)"
    case p.Term.StringConst(x) => s"String($x)"
  }
}

extension (e: p.Type) {

  def mapAcc[A](f: p.Type => (p.Type, List[A])): (p.Type, List[A]) = e match {
    case p.Type.Array(c) =>
      val (c0, as0) = c.mapAcc(f)
      val (t0, as1) = f(p.Type.Array(c0))
      (t0, as0 ::: as1)
    case p.Type.Struct(name, tpeVars, args) =>
      val (args0, as0) = args.map(_.mapAcc(f)).unzip
      val (t0, as1)    = f(p.Type.Struct(name, tpeVars, args0))
      (t0, as0.flatten ::: as1)
    case p.Type.Exec(tpeVars, args, rtn) =>
      val (args0, as0) = args.map(_.mapAcc(f)).unzip
      val (rtn0, as1)  = rtn.mapAcc(f)
      val (t0, as2)    = f(p.Type.Exec(tpeVars, args0, rtn0))
      (t0, as0.flatten ::: as1 ::: as2)
    case x => f(x)
  }

  def map(f: p.Type => p.Type): p.Type      = mapAcc[Unit](x => f(x) -> Nil)._1
  def acc[A](f: p.Type => List[A]): List[A] = mapAcc[A](x => x -> f(x))._2

  def isNumeric = e.kind match {
    case TypeKind.Integral | TypeKind.Fractional => true
    case _                                       => false
  }
  def repr: String = e match {
    case p.Type.Struct(sym, tpeVars, args) =>
      s"@${sym.repr}${tpeVars.zipAll(args, "???", p.Type.Var("???")).map((v, a) => s"$v=${a.repr}").mkString("<", ",", ">")}"
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
    case p.Type.Nothing     => "Nothing"
    case p.Type.Var(name)   => s"#$name"
    case p.Type.Exec(tpeArgs, args, rtn) =>
      s"<${tpeArgs.mkString(",")}>(${args.map(_.repr).mkString(",")}) => ${rtn.repr}"
  }

  // TODO remove
  def monomorphicName: String = e match {
    case p.Type.Struct(sym, _, args)     => sym.fqn.mkString("_") + args.map(_.monomorphicName).mkString("_", "_", "_")
    case p.Type.Array(comp)              => s"${comp.monomorphicName}[]"
    case p.Type.Bool                     => "Bool"
    case p.Type.Byte                     => "Byte"
    case p.Type.Char                     => "Char"
    case p.Type.Short                    => "Short"
    case p.Type.Int                      => "Int"
    case p.Type.Long                     => "Long"
    case p.Type.Float                    => "Float"
    case p.Type.Double                   => "Double"
    case p.Type.String                   => "String"
    case p.Type.Unit                     => "Unit"
    case p.Type.Nothing                  => "Nothing"
    case p.Type.Var(name)                => s"#$name"
    case p.Type.Exec(tpeArgs, args, rtn) => ???
  }
}

extension (e: p.Expr) {
  def repr: String = e match {
    case p.Expr.NullaryIntrinsic(kind, rtn) =>
      val fn = kind match {
        case p.NullaryIntrinsicKind.GpuGlobalIdxX  => "GlobalIdxX"
        case p.NullaryIntrinsicKind.GpuGlobalIdxY  => "GlobalIdxY"
        case p.NullaryIntrinsicKind.GpuGlobalIdxZ  => "GlobalIdxZ"
        case p.NullaryIntrinsicKind.GpuGlobalSizeX => "GlobalSizeX"
        case p.NullaryIntrinsicKind.GpuGlobalSizeY => "GlobalSizeY"
        case p.NullaryIntrinsicKind.GpuGlobalSizeZ => "GlobalSizeZ"
        case p.NullaryIntrinsicKind.GpuGroupIdxX   => "GroupIdxX"
        case p.NullaryIntrinsicKind.GpuGroupIdxY   => "GroupIdxY"
        case p.NullaryIntrinsicKind.GpuGroupIdxZ   => "GroupIdxZ"
        case p.NullaryIntrinsicKind.GpuGroupSizeX  => "GroupSizeX"
        case p.NullaryIntrinsicKind.GpuGroupSizeY  => "GroupSizeY"
        case p.NullaryIntrinsicKind.GpuGroupSizeZ  => "GroupSizeZ"
        case p.NullaryIntrinsicKind.GpuLocalIdxX   => "LocalIdxX"
        case p.NullaryIntrinsicKind.GpuLocalIdxY   => "LocalIdxY"
        case p.NullaryIntrinsicKind.GpuLocalIdxZ   => "LocalIdxZ"
        case p.NullaryIntrinsicKind.GpuLocalSizeX  => "LocalSizeX"
        case p.NullaryIntrinsicKind.GpuLocalSizeY  => "LocalSizeY"
        case p.NullaryIntrinsicKind.GpuLocalSizeZ  => "LocalSizeZ"

        case p.NullaryIntrinsicKind.GpuGroupBarrier => "GroupBarrier"
        case p.NullaryIntrinsicKind.GpuGroupFence   => "GroupFence"
      }
      s"$fn'"
    case p.Expr.UnaryIntrinsic(lhs, kind, rtn) =>
      val fn = kind match {
        case p.UnaryIntrinsicKind.Sin  => "sin"
        case p.UnaryIntrinsicKind.Cos  => "cos"
        case p.UnaryIntrinsicKind.Tan  => "tan"
        case p.UnaryIntrinsicKind.Asin => "asin"
        case p.UnaryIntrinsicKind.Acos => "acos"
        case p.UnaryIntrinsicKind.Atan => "atan"
        case p.UnaryIntrinsicKind.Sinh => "sinh"
        case p.UnaryIntrinsicKind.Cosh => "cosh"
        case p.UnaryIntrinsicKind.Tanh => "tanh"

        case p.UnaryIntrinsicKind.Signum => "signum"
        case p.UnaryIntrinsicKind.Abs    => "abs"
        case p.UnaryIntrinsicKind.Round  => "round"
        case p.UnaryIntrinsicKind.Ceil   => "ceil"
        case p.UnaryIntrinsicKind.Floor  => "floor"
        case p.UnaryIntrinsicKind.Rint   => "rint"

        case p.UnaryIntrinsicKind.Sqrt  => "sqrt"
        case p.UnaryIntrinsicKind.Cbrt  => "cbrt"
        case p.UnaryIntrinsicKind.Exp   => "exp"
        case p.UnaryIntrinsicKind.Expm1 => "expm1"
        case p.UnaryIntrinsicKind.Log   => "log"
        case p.UnaryIntrinsicKind.Log1p => "log1p"
        case p.UnaryIntrinsicKind.Log10 => "log10"

        case p.UnaryIntrinsicKind.BNot => "~"

        case p.UnaryIntrinsicKind.Pos => "+"
        case p.UnaryIntrinsicKind.Neg => "-"

        case p.UnaryIntrinsicKind.LogicNot => "!"

      }
      s"$fn'(${lhs.repr})"
    case p.Expr.BinaryIntrinsic(lhs, rhs, kind, _) =>
      val op = kind match {
        case p.BinaryIntrinsicKind.Add => "+"
        case p.BinaryIntrinsicKind.Sub => "-"
        case p.BinaryIntrinsicKind.Mul => "*"
        case p.BinaryIntrinsicKind.Div => "/"
        case p.BinaryIntrinsicKind.Rem => "%"

        case p.BinaryIntrinsicKind.Pow => "**"

        case p.BinaryIntrinsicKind.Min => "min"
        case p.BinaryIntrinsicKind.Max => "max"

        case p.BinaryIntrinsicKind.Atan2 => "atan2"
        case p.BinaryIntrinsicKind.Hypot => "hypot"

        case p.BinaryIntrinsicKind.BAnd => "&"
        case p.BinaryIntrinsicKind.BOr  => "|"
        case p.BinaryIntrinsicKind.BXor => "^"
        case p.BinaryIntrinsicKind.BSL  => "<<"
        case p.BinaryIntrinsicKind.BSR  => ">>"
        case p.BinaryIntrinsicKind.BZSR => ">>>"

        case p.BinaryIntrinsicKind.LogicEq  => "=="
        case p.BinaryIntrinsicKind.LogicNeq => "!="
        case p.BinaryIntrinsicKind.LogicAnd => "&&"
        case p.BinaryIntrinsicKind.LogicOr  => "||"
        case p.BinaryIntrinsicKind.LogicLte => "<="
        case p.BinaryIntrinsicKind.LogicGte => ">="
        case p.BinaryIntrinsicKind.LogicLt  => "<"
        case p.BinaryIntrinsicKind.LogicGt  => ">"

      }
      s"${lhs.repr} $op' ${rhs.repr}"

    case p.Expr.Cast(from, to) => s"${from.repr}.to[${to.repr}]"
    case p.Expr.Alias(ref)     => s"(~>${ref.repr})"

    case p.Expr.Invoke(name, tpeArgs, recv, args, tpe) =>
      s"${recv.map(_.repr).getOrElse("<module>")}.${name.repr}<${tpeArgs.map(_.repr).mkString(",")}>(${args.map(_.repr).mkString(",")}) : ${tpe.repr}"
    case p.Expr.Index(lhs, idx, tpe) => s"${lhs.repr}[${idx.repr}] : ${tpe.repr}"
    case p.Expr.Alloc(tpe, size)     => s"new [${tpe.component.repr}*${size.repr}]"
  }
}

extension (e: p.Stmt) {

  def acc[A](f: p.Stmt => List[A]): List[A]        = e.mapAcc[A](x => (x :: Nil, f(x)))._2
  def map(f: p.Stmt => List[p.Stmt]): List[p.Stmt] = e.mapAcc[Unit](x => (f(x), Nil))._1
  def mapAcc[A](f: p.Stmt => (List[p.Stmt], List[A])): (List[p.Stmt], List[A]) = e match {
    case x @ p.Stmt.Comment(_)      => f(x)
    case x @ p.Stmt.Var(_, _)       => f(x)
    case x @ p.Stmt.Mut(_, _, _)    => f(x)
    case x @ p.Stmt.Update(_, _, _) => f(x)
    case p.Stmt.While(tests, cond, body) =>
      val (sss0, xss) = tests.map(_.mapAcc(f)).unzip
      val (sss1, yss) = body.map(_.mapAcc(f)).unzip
      val (sss2, zss) = f(p.Stmt.While(sss0.flatten, cond, sss1.flatten))
      (sss2, xss.flatten ::: yss.flatten ::: zss)
    case x @ p.Stmt.Break => f(x)
    case x @ p.Stmt.Cont  => f(x)
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      val (tss, tass) = trueBr.map(_.mapAcc(f)).unzip
      val (fss, fass) = falseBr.map(_.mapAcc(f)).unzip
      val (ss, ass)   = f(p.Stmt.Cond(cond, tss.flatten, fss.flatten))
      (ss, (tass ::: fass).flatten ::: ass)
    case x @ p.Stmt.Return(_) => f(x)
  }

  def accExpr[A](f: p.Expr => List[A]): List[A]                  = e.mapAccExpr(e => (e, Nil, f(e)))._2
  def mapExpr(f: p.Expr => (p.Expr, List[p.Stmt])): List[p.Stmt] = e.mapAccExpr(f(_) ++ Nil *: EmptyTuple)._1
  def mapAccExpr[A](f: p.Expr => (p.Expr, List[p.Stmt], List[A])): (List[p.Stmt], List[A]) = e match {
    case x @ p.Stmt.Comment(_) => (x :: Nil, Nil)
    case p.Stmt.Var(name, rhs) =>
      rhs match {
        case None => (p.Stmt.Var(name, None) :: Nil, Nil)
        case Some(x) =>
          val (y, xs, as) = f(x)
          (xs :+ p.Stmt.Var(name, Some(y)), as)
      }
    case p.Stmt.Mut(name, expr, copy)   => val (y, xs, as) = f(expr); (xs :+ p.Stmt.Mut(name, y, copy), as)
    case x @ p.Stmt.Update(_, _, _)     => (x :: Nil, Nil)
    case p.Stmt.While(test, cond, body) =>
      // val (y, xs, as) = f(cond)
      val (tss, tass) = test.map(_.mapAccExpr(f)).unzip
      val (bss, bass) = body.map(_.mapAccExpr(f)).unzip
      (p.Stmt.While(tss.flatten, cond, bss.flatten) :: Nil, (tass ::: bass).flatten)
    case x @ p.Stmt.Break => (x :: Nil, Nil)
    case x @ p.Stmt.Cont  => (x :: Nil, Nil)
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      val (y, xs, as) = f(cond)
      val (tss, tass) = trueBr.map(_.mapAccExpr(f)).unzip
      val (fss, fass) = falseBr.map(_.mapAccExpr(f)).unzip
      (xs :+ p.Stmt.Cond(y, tss.flatten, fss.flatten), (as :: tass ::: fass).flatten)
    case p.Stmt.Return(expr) => val (y, xs, as) = f(expr); (xs :+ p.Stmt.Return(y), as)
  }

  def accTerm[A](g: p.Term.Select => List[A], f: p.Term => List[A]): List[A] =
    e.mapAccTerm(s => s -> g(s), t => t -> f(t))._2
  def mapTerm(g: p.Term.Select => p.Term.Select, f: p.Term => p.Term): List[p.Stmt] =
    e.mapAccTerm(s => g(s) -> Nil, t => f(t) -> Nil)._1
  def mapAccTerm[A](
      g: p.Term.Select => (p.Term.Select, List[A]),
      f: p.Term => (p.Term, List[A])
  ): (List[p.Stmt], List[A]) = {
    val (stmts0, as0) = e.mapAcc {
      case p.Stmt.Mut(name, expr, copy) =>
        val (name0, as) = g(name)
        (p.Stmt.Mut(name0, expr, copy) :: Nil, as)
      case p.Stmt.Update(lhs, idx, value) =>
        val (lhs0, as0)   = g(lhs)
        val (idx0, as1)   = f(idx)
        val (value0, as2) = f(value)
        (p.Stmt.Update(lhs0, idx0, value0) :: Nil, as0 ::: as1 ::: as2)
      case p.Stmt.While(tests, cond, body) =>
        val (cond0, as) = f(cond)
        (p.Stmt.While(tests, cond0, body) :: Nil, as)
      case x => (x :: Nil, Nil)
    }
    val (stmts1, as1) = stmts0
      .map(_.mapAccExpr {
        case x @ p.Expr.NullaryIntrinsic(_, _) =>
          (x, Nil, Nil)
        case p.Expr.UnaryIntrinsic(lhs, kind, rtn) =>
          val (lhs0, as) = f(lhs)
          (p.Expr.UnaryIntrinsic(lhs0, kind, rtn), Nil, as)
        case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) =>
          val (lhs0, as0) = f(lhs)
          val (rhs0, as1) = f(rhs)
          (p.Expr.BinaryIntrinsic(lhs0, rhs0, kind, rtn), Nil, as0 ::: as1)
        case p.Expr.Cast(from, to) =>
          val (from0, as0) = f(from)
          (p.Expr.Cast(from0, to), Nil, as0)
        case p.Expr.Alias(ref) =>
          val (ref0, as0) = f(ref)
          (p.Expr.Alias(ref0), Nil, as0)
        case p.Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
          val (receiver0, as0) = receiver.map(f).fold((None, Nil))((x, as) => (Some(x), as))
          val (args0, as1)     = args.map(f).unzip
          (p.Expr.Invoke(name, tpeArgs, receiver0, args0, rtn), Nil, (as0 :: as1).flatten)
        case p.Expr.Index(lhs, idx, component) =>
          val (lhs0, as0) = g(lhs)
          val (idx0, as1) = f(idx)
          (p.Expr.Index(lhs0, idx0, component), Nil, as0 ::: as1)
        case p.Expr.Alloc(witness, term) =>
          val (term0, as0) = f(term)
          (p.Expr.Alloc(witness, term0), Nil, as0)
      })
      .unzip //

    (stmts1.flatten, (as0 :: as1).flatten)
  }

  def accType[A](f: p.Type => List[A]): List[A]  = mapAccType(x => x -> f(x))._2
  def mapType(f: p.Type => p.Type): List[p.Stmt] = mapAccType(x => f(x) -> Nil)._1
  def mapAccType[A](f: p.Type => (p.Type, List[A])): (List[p.Stmt], List[A]) = {
    val (stmts0, as0) = e.mapAccTerm(
      { case p.Term.Select(xs, x) =>
        val (xs0, as0) = xs.map(_.mapAccType(f)).unzip
        val (x0, as1)  = x.mapAccType(f)
        (p.Term.Select(xs0, x0), as0.flatten ::: as1)
      },
      {
        case p.Term.Select(xs, x) =>
          val (xs0, as0) = xs.map(_.mapAccType(f)).unzip
          val (x0, as1)  = x.mapAccType(f)
          (p.Term.Select(xs0, x0), as0.flatten ::: as1)
        case x => (x, Nil)
      }
    )
    val (stmts1, as1) = stmts0
      .map(_.mapAccExpr {
        case p.Expr.NullaryIntrinsic(kind, rtn) =>
          val (rtn0, as0) = f(rtn)
          (p.Expr.NullaryIntrinsic(kind, rtn0), Nil, as0)
        case p.Expr.UnaryIntrinsic(lhs, kind, rtn) =>
          val (rtn0, as0) = f(rtn)
          (p.Expr.UnaryIntrinsic(lhs, kind, rtn0), Nil, as0)
        case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) =>
          val (rtn0, as0) = f(rtn)
          (p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn0), Nil, as0)
        case p.Expr.Cast(from, as) =>
          val (as0, as0_) = f(as)
          (p.Expr.Cast(from, as0), Nil, as0_)
        case e @ p.Expr.Alias(_) =>
          (e, Nil, Nil)
        case p.Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
          val (tpeArgs0, as0) = tpeArgs.map(f).unzip
          val (rtn0, as1)     = f(rtn)
          (p.Expr.Invoke(name, tpeArgs0, receiver, args, rtn0), Nil, as0.flatten ::: as1)
        case p.Expr.Index(lhs, idx, component) =>
          val (component0, as0) = f(component)
          (p.Expr.Index(lhs, idx, component0), Nil, as0)
        case p.Expr.Alloc(witness, size) =>
          val (component0, as0) = f(witness.component)
          (p.Expr.Alloc(p.Type.Array(component0), size), Nil, as0)
      })
      .unzip

    val (stmts2, as2) = stmts1.flatten
      .map(_.mapAcc {
        case p.Stmt.Var(named, x) =>
          val (tpe0, as) = named.mapAccType(f) //
          (p.Stmt.Var(tpe0, x) :: Nil, as)
        case x => (x :: Nil, Nil)
      })
      .unzip

    (stmts2.flatten, (as0 :: as1 ::: as2).flatten)
  }

  def repr: String = e match {
    case p.Stmt.Comment(value)          => s" /* $value */"
    case p.Stmt.Var(name, rhs)          => s"var ${name.repr} = ${rhs.fold("_")(_.repr)}"
    case p.Stmt.Mut(name, expr, copy)   => s"${name.repr} ${if (copy) ":=!" else ":="} ${expr.repr}"
    case p.Stmt.Update(lhs, idx, value) => s"${lhs.repr}[${idx.repr}] := ${value.repr}"
    case p.Stmt.While(tests, cond, body) =>
      s"while({${(tests.map(_.repr) :+ cond.repr).mkString(";")}}){\n${body.map("  " + _.repr).mkString("\n")}\n}"
    case p.Stmt.Break        => s"break;"
    case p.Stmt.Cont         => s"continue;"
    case p.Stmt.Return(expr) => s"return ${expr.repr}"
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      s"if(${cond.repr}) {\n${trueBr.map("  " + _.repr).mkString("\n")}\n} else {\n${falseBr.map("  " + _.repr).mkString("\n")}\n}"
  }

}

extension (f: p.Function) {

  def mangledName = f.receiver.map(_.tpe.repr).getOrElse("") + "!" + f.name.fqn
    .mkString("_") + "!" + f.args.map(_.tpe.repr).mkString("_") + "!" + f.rtn.repr

  def signature = p.Signature(f.name, f.tpeVars, f.receiver.map(_.tpe), f.args.map(_.tpe), f.rtn)

  def signatureRepr = {
    val captures = f.captures.map(_.repr).mkString(",")
    val tpeVars  = f.tpeVars.mkString(",")
    val args     = f.args.map(_.repr).mkString(",")
    s"${f.receiver.fold("")(r => r.repr + ".")}${f.name.repr}<$tpeVars>($args)[$captures] : ${f.rtn.repr}"
  }

  def mapType(tf: p.Type => p.Type): p.Function = f.copy(
    receiver = f.receiver.map(_.mapType(tf)),
    args = f.args.map(_.mapType(tf)),
    rtn = f.rtn.map(tf),
    body = f.body.flatMap(_.mapType(tf))
  )

  def repr: String =
    s"""${f.signatureRepr} = {
       |${f.body.flatMap(_.repr.linesIterator.map("  " + _)).mkString("\n")}
	   |}""".stripMargin
}

extension (f: p.Signature) {

  def mapType(tf: p.Type => p.Type): p.Signature = f.copy(
    receiver = f.receiver.map(_.map(tf)),
    args = f.args.map(_.map(tf)),
    rtn = f.rtn.map(tf)
  )

  def repr: String =
    s"${f.receiver.fold("")(r => r.repr + ".")}${f.name.repr}(${f.args.map(_.repr).mkString(", ")}) : ${f.rtn.repr}"
}
