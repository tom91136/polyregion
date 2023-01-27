package polyregion.ast

import cats.kernel.Semigroup
import polyregion.ast.PolyAst as p
import polyregion.ast.PolyAst.{Type, TypeKind}
import polyregion.ast.Traversal.*

import java.lang.reflect.Modifier
import scala.annotation.{tailrec, targetName}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.{Success, Try}

given Traversal[p.Term, p.Type] = Traversal.derived
given Traversal[p.Expr, p.Type] = Traversal.derived
given Traversal[p.Stmt, p.Type] = Traversal.derived
given Traversal[p.Type, p.Type] = Traversal.derived

given Traversal[p.Type, p.Expr] = Traversal.derived
given Traversal[p.Stmt, p.Expr] = Traversal.derived
given Traversal[p.Term, p.Expr] = Traversal.derived
given Traversal[p.Expr, p.Expr] = Traversal.derived

given Traversal[p.Type, p.Term] = Traversal.derived
given Traversal[p.Expr, p.Term] = Traversal.derived
given Traversal[p.Stmt, p.Term] = Traversal.derived
given Traversal[p.Term, p.Term] = Traversal.derived

given Traversal[p.Type, p.Stmt] = Traversal.derived
given Traversal[p.Expr, p.Stmt] = Traversal.derived
given Traversal[p.Term, p.Stmt] = Traversal.derived
given Traversal[p.Stmt, p.Stmt] = Traversal.derived

given Traversal[p.Signature, p.Type] = Traversal.derived

given Traversal[p.Function, p.Type] = Traversal.derived
given Traversal[p.Function, p.Expr] = Traversal.derived
given Traversal[p.Function, p.Term] = Traversal.derived
given Traversal[p.Function, p.Stmt] = Traversal.derived

given Traversal[p.StructDef, p.Type] = Traversal.derived

@tailrec def doUntilNotEq[A](x: A, n: Int = 0, limit: Int = Int.MaxValue)(f: (Int, A) => A): (Int, A) = {
  val y = f(n, x)
  if (y == x || n >= limit) (n, y)
  else doUntilNotEq(y, n + 1, limit)(f)
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
case class Log(name: String, lines: ArrayBuffer[(String, Vector[String]) | Log]) {

  @targetName("append") infix def subLog(name: String): Log = {
    val sub = Log(name); lines += sub; sub
  }

  @targetName("append") infix def +=(log: Log): Unit          = lines += log
  @targetName("appendAll") infix def ++=(log: Seq[Log]): Unit = lines ++= log
  def info(message: String, details: String*): Unit           = lines += (message -> details.toVector)

//  def info(message: String, details: String*): Result[Log] = info_(message, details*).success

  def render(nesting: Int = 0): Vector[String] =
    Try {
      val colour = Log.Colours(nesting % Log.Colours.size)
      val attr   = colour ++ fansi.Reversed.On ++ fansi.Bold.On
      val indent = colour("┃ ")

      ((colour("┏━") ++ attr(s" ${name} ") ++ colour("")) +: lines.toVector
        .flatMap {
          case log: Log => log.render(nesting + 1).map(indent ++ _)
          case (line, details) =>
            ((colour ++ fansi.Underlined.On)(s"▓ $line ▓")) +: details.flatMap { l =>
              l.linesIterator.toList match {
                case x :: xs =>
                  ((colour("┃ ╰ ") ++ s"$x") :: xs.map(x => indent ++ s"  $x")).toVector
                case Nil => Vector.empty
              }
            }
        } :+ colour(s"┗━${"━" * (name.length + 2)}"))
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

  def apply(name: String): Log = Log(name, ArrayBuffer.empty)

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

  given SymToExpr: ToExpr[p.Sym] with {
    def apply(x: p.Sym)(using Quotes) = '{ p.Sym(${ Expr(x.fqn) }) }
  }
  given NamedToExpr: ToExpr[p.Named] with {
    def apply(x: p.Named)(using Quotes) = '{ p.Named(${ Expr(x.symbol) }, ${ Expr(x.tpe) }) }
  }
  given StructMemberToExpr: ToExpr[p.StructMember] with {
    def apply(x: p.StructMember)(using Quotes) = '{ p.StructMember(${ Expr(x.named) }, ${ Expr(x.isMutable) }) }
  }
  given StructDefToExpr: ToExpr[p.StructDef] with {
    def apply(x: p.StructDef)(using Quotes) = '{
      p.StructDef(
        ${ Expr(x.name) },
        ${ Expr(x.isReference) },
        ${ Expr(x.tpeVars) },
        ${ Expr(x.members) },
        ${ Expr(x.parents) }
      )
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
      case p.Type.Struct(name, tpeVars, args, parents) =>
        '{ p.Type.Struct(${ Expr(name) }, ${ Expr(tpeVars) }, ${ Expr(args) }, ${ Expr(parents) }) }
      case p.Type.Array(component)         => '{ p.Type.Array(${ Expr(component) }) }
      case p.Type.Exec(tpeVars, args, rtn) => ???
      case p.Type.Nothing                  => ???
    }
  }

}

extension (m: p.StructMember) {
  def repr: String = s"${if (m.isMutable) "var" else "val"} ${m.named.repr}"
}

extension (sd: p.StructDef) {
  def tpe: p.Type.Struct = p.Type.Struct(sd.name, sd.tpeVars, Nil, sd.parents)
  def repr: String =
    s"${sd.name.repr}<${sd.tpeVars.mkString(",")}>${if (sd.isReference) "*" else ""} { ${sd.members
        .map(_.repr)
        .mkString("; ")} } <: ${sd.parents.map(_.repr).mkString("<:")}"
}

extension (e: p.Sym) {
  def repr: String = e.fqn.mkString(".")
}

extension (n: p.Named) {
  def repr: String = s"(${n.symbol}: ${n.tpe.repr})"
}

extension (e: p.Type) {

  @targetName("tpeEquals")
  def =:=(that: p.Type): Boolean =
    (e, that) match {
      case (p.Type.Struct(xSym, xVars, xTpes, xParents), p.Type.Struct(ySym, yVars, yTpes, yParents)) =>
        xSym == ySym && xVars == yVars && xTpes.zip(yTpes).forall(_ =:= _) && xParents.zip(yParents).forall(_ == _)
      case (p.Type.Nothing, p.Type.Nothing)             => true
      case (p.Type.Nothing, _)                          => true
      case (_, p.Type.Nothing)                          => true
      case (p.Type.Array(xs), p.Type.Array(ys))         => xs =:= ys
      case (p.Type.Exec(_, _, _), p.Type.Exec(_, _, _)) => ??? // TODO impl exec
      case (x, y)                                       => x == y
    }

  def mapLeaf(f: p.Type => p.Type): p.Type = e match {
    case p.Type.Struct(name, tpeVars, args, parents) => p.Type.Struct(name, tpeVars, args.map(f), parents)
    case p.Type.Array(component)                     => p.Type.Array(f(component))
    case p.Type.Exec(tpeVars, args, rtn)             => p.Type.Exec(tpeVars, args.map(f), f(rtn))
    case x                                           => f(x)
  }

  def mapNode(f: p.Type => p.Type): p.Type = e match {
    case p.Type.Struct(name, tpeVars, args, parents) => f(p.Type.Struct(name, tpeVars, args.map(f), parents))
    case p.Type.Array(component)                     => f(p.Type.Array(f(component)))
    case p.Type.Exec(tpeVars, args, rtn)             => f(p.Type.Exec(tpeVars, args.map(f), f(rtn)))
    case x                                           => x
  }

  def isNumeric: Boolean = e.kind match {
    case TypeKind.Integral | TypeKind.Fractional => true
    case _                                       => false
  }
  def repr: String = e match {
    case p.Type.Struct(sym, tpeVars, args, parents) =>
      s"@${sym.repr}${tpeVars.zipAll(args, "???", p.Type.Var("???")).map((v, a) => s"$v=${a.repr}").mkString("<", ",", ">")}(${parents.map(_.repr).mkString("<:")})"
    case p.Type.Array(comp) => s"Array[${comp.repr}]"
    case p.Type.Bool        => "Bool"
    case p.Type.Byte        => "Byte"
    case p.Type.Char        => "Char"
    case p.Type.Short       => "Short"
    case p.Type.Int         => "Int"
    case p.Type.Long        => "Long"
    case p.Type.Float       => "Float"
    case p.Type.Double      => "Double"
    case p.Type.Unit        => "Unit"
    case p.Type.Nothing     => "Nothing"
    case p.Type.Var(name)   => s"#$name"
    case p.Type.Exec(tpeArgs, args, rtn) =>
      s"<${tpeArgs.mkString(",")}>(${args.map(_.repr).mkString(",")}) => ${rtn.repr}"
  }

  // TODO remove
  def monomorphicName: String = e match {
    case p.Type.Struct(sym, _, args, parents) =>
      sym.fqn.mkString("_") + args.map(_.monomorphicName).mkString("_", "_", "_")
    case p.Type.Array(comp)              => s"${comp.monomorphicName}[]"
    case p.Type.Bool                     => "Bool"
    case p.Type.Byte                     => "Byte"
    case p.Type.Char                     => "Char"
    case p.Type.Short                    => "Short"
    case p.Type.Int                      => "Int"
    case p.Type.Long                     => "Long"
    case p.Type.Float                    => "Float"
    case p.Type.Double                   => "Double"
    case p.Type.Unit                     => "Unit"
    case p.Type.Nothing                  => "Nothing"
    case p.Type.Var(name)                => s"#$name"
    case p.Type.Exec(tpeArgs, args, rtn) => ???
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

    case p.Expr.Invoke(name, tpeArgs, recv, args, captures, tpe) =>
      s"${recv.map(_.repr).getOrElse("<module>")}.${name.repr}<${tpeArgs.map(_.repr).mkString(",")}>(${args
          .map(_.repr)
          .mkString(",")})[${captures.map(_.repr).mkString(",")}] : ${tpe.repr}"
    case p.Expr.Index(lhs, idx, tpe) => s"${lhs.repr}[${idx.repr}] : ${tpe.repr}"
    case p.Expr.Alloc(tpe, size)     => s"new [${tpe.repr}*${size.repr}]"
  }
}

extension (stmt: p.Stmt) {
  def repr: String = stmt match {
    case p.Stmt.Block(xs) =>
      s"{\n${xs.flatMap(_.repr.linesIterator.map("  " + _)).mkString("\n")}\n}"
    case p.Stmt.Comment(value)          => s" /* $value */"
    case p.Stmt.Var(name, rhs)          => s"var ${name.repr} = ${rhs.fold("_")(_.repr)}"
    case p.Stmt.Mut(name, expr, copy)   => s"${name.repr} ${if (copy) ":=!" else ":="} ${expr.repr}"
    case p.Stmt.Update(lhs, idx, value) => s"${lhs.repr}[${idx.repr}] := ${value.repr}"
    case p.Stmt.While(tests, cond, body) =>
      s"while({${(tests.map(_.repr) :+ cond.repr).mkString(";")}}){\n${body.flatMap(_.repr.linesIterator.map("  " + _)).mkString("\n")}\n}"
    case p.Stmt.Break        => s"break;"
    case p.Stmt.Cont         => s"continue;"
    case p.Stmt.Return(expr) => s"return ${expr.repr}"
    case p.Stmt.Cond(cond, trueBr, falseBr) =>
      s"if(${cond.repr}) {\n${trueBr.flatMap(_.repr.linesIterator.map("  " + _)).mkString("\n")}\n} else {\n${falseBr
          .flatMap(_.repr.linesIterator.map("  " + _))
          .mkString("\n")}\n}"
  }
}

extension (fn: p.Function) {

  def mangledName = fn.receiver.map(_.tpe.repr).getOrElse("") + "!" + fn.name.fqn
    .mkString("_") + "!" + fn.args.map(_.tpe.repr).mkString("_") + "!" + fn.rtn.repr

  def signature = p.Signature(
    fn.name,
    fn.tpeVars,
    fn.receiver.map(_.tpe),
    fn.args.map(_.tpe),
    fn.moduleCaptures.map(_.tpe),
    fn.termCaptures.map(_.tpe),
    fn.rtn
  )

  def signatureRepr = {
    val termCaptures   = fn.termCaptures.map(_.repr).mkString(",")
    val moduleCaptures = fn.moduleCaptures.map(_.repr).mkString(",")
    val tpeVars        = fn.tpeVars.mkString(",")
    val args           = fn.args.map(_.repr).mkString(",")
    s"${fn.receiver.fold("")(r => r.repr + ".")}${fn.name.repr}<$tpeVars>($args)[$moduleCaptures;${termCaptures}] : ${fn.rtn.repr}"
  }

  def repr: String =
    s"""${fn.signatureRepr} = {
       |${fn.body.flatMap(_.repr.linesIterator.map("  " + _)).mkString("\n")}
       |}""".stripMargin
}

extension (f: p.Signature) {
  def repr: String =
    s"<${f.tpeVars.mkString(",")}>${f.receiver
        .fold("")(r => s"(${r.repr}).")}${f.name.repr}(${f.args.map(_.repr).mkString(", ")}) : ${f.rtn.repr}"
}
