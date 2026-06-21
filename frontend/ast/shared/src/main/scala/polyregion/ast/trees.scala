package polyregion.ast

import polyregion.ast.PolyAST as p
import polyregion.ast.PolyAST.Type
import polyregion.ast.Traversal.*

import scala.annotation.{tailrec, targetName}
import scala.util.Success

given Traversal[p.Term, p.Type] = Traversal.derived
given Traversal[p.Expr, p.Type] = Traversal.derived
given Traversal[p.Stmt, p.Type] = Traversal.derived
given Traversal[p.Type, p.Type] = Traversal.derived

given Traversal[p.Type, p.Term] = Traversal.derived
given Traversal[p.Term, p.Term] = Traversal.derived
given Traversal[p.Expr, p.Term] = Traversal.derived
given Traversal[p.Stmt, p.Term] = Traversal.derived

given Traversal[p.Type, p.Expr] = Traversal.derived
given Traversal[p.Term, p.Expr] = Traversal.derived
given Traversal[p.Expr, p.Expr] = Traversal.derived
given Traversal[p.Stmt, p.Expr] = Traversal.derived

given Traversal[p.Type, p.Stmt] = Traversal.derived
given Traversal[p.Term, p.Stmt] = Traversal.derived
given Traversal[p.Expr, p.Stmt] = Traversal.derived
given Traversal[p.Stmt, p.Stmt] = Traversal.derived

given Traversal[p.Signature, p.Type] = Traversal.derived

given Traversal[p.Function, p.Type] = Traversal.derived
given Traversal[p.Function, p.Term] = Traversal.derived
given Traversal[p.Function, p.Expr] = Traversal.derived
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

extension [A](a: Result[A]) {
  def withFilter(p: A => Boolean) = a.flatMap(x => if (p(x)) Right(x) else Left(MatchError(x)))
}

extension [A](a: A) {
  def success: Result[A] = Right(a)
}
extension (message: => String) {
  def fail[A]: Result[A] = Left(CompilerException(message))
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

extension (sd: p.StructDef) {
  def applied(args: List[p.Type]): p.Type.Struct = p.Type.Struct(sd.name, args)
  def erasedTpe: p.Type.Struct =
    p.Type.Struct(sd.name, sd.tpeVars.map(p.Type.Var(_)))
}

extension (e: p.Type) {

  def erased: p.Type = e match {
    case p.Type.Struct(sym, args) =>
      p.Type.Struct(sym, List.tabulate(args.size)(i => p.Type.Var(s"T$i")))
    case x => x
  }

  @targetName("tpeEquals")
  def =:=(that: p.Type): Boolean =
    (e, that) match {
      case (p.Type.Struct(xSym, xArgs), p.Type.Struct(ySym, yArgs)) =>
        xSym == ySym && xArgs.sizeIs == yArgs.size && xArgs.zip(yArgs).forall(_ =:= _)
      case (p.Type.Nothing, p.Type.Nothing)         => true
      case (p.Type.Nothing, _)                      => true
      case (_, p.Type.Nothing)                      => true
      case (p.Type.Ptr(xt, xa), p.Type.Ptr(yt, ya)) => xt =:= yt && xa == ya
      case (p.Type.Arr(xt, xl, xa), p.Type.Arr(yt, yl, ya)) =>
        xt =:= yt && xl == yl && xa == ya
      case (p.Type.Exec(_, _, _), p.Type.Exec(_, _, _)) => ??? // TODO impl exec
      case (x, y)                                       => x == y
    }

  def mapLeaf(f: p.Type => p.Type): p.Type = e match {
    case p.Type.Struct(name, args)            => p.Type.Struct(name, args.map(f))
    case p.Type.Ptr(component, space)         => p.Type.Ptr(f(component), space)
    case p.Type.Arr(component, length, space) => p.Type.Arr(f(component), length, space)
    case p.Type.Exec(tpeVars, args, rtn)      => p.Type.Exec(tpeVars, args.map(f), f(rtn))
    case x                                    => f(x)
  }

  def mapNode(f: p.Type => p.Type): p.Type = e match {
    case p.Type.Struct(name, args)            => f(p.Type.Struct(name, args.map(f)))
    case p.Type.Ptr(component, space)         => f(p.Type.Ptr(f(component), space))
    case p.Type.Arr(component, length, space) => f(p.Type.Arr(f(component), length, space))
    case p.Type.Exec(tpeVars, args, rtn)      => f(p.Type.Exec(tpeVars, args.map(f), f(rtn)))
    case x                                    => x
  }

  def isNumeric: Boolean = e.kind match {
    case Type.Kind.Integral | Type.Kind.Fractional => true
    case _                                         => false
  }

  // TODO remove
  def monomorphicName: String = e match {
    case p.Type.Struct(sym, args) =>
      sym.fqn.mkString("_") + args.map(_.monomorphicName).mkString("_", "_", "_")
    case p.Type.Ptr(comp, space)         => s"${comp.monomorphicName}*^$space"
    case p.Type.Arr(comp, length, space) => s"${comp.monomorphicName}[$length]^$space"
    case p.Type.Bool1                    => "Bool"
    case p.Type.IntU8                    => "U8"
    case p.Type.IntU16                   => "Charc"
    case p.Type.IntU32                   => "U32"
    case p.Type.IntU64                   => "U64"
    case p.Type.IntS8                    => "Byteb"
    case p.Type.IntS16                   => "Shorts"
    case p.Type.IntS32                   => "Inti"
    case p.Type.IntS64                   => "Longl"
    case p.Type.Float16                  => "F16"
    case p.Type.Float32                  => "Floatf"
    case p.Type.Float64                  => "Doubled"
    case p.Type.Unit0                    => "Unitv"
    case p.Type.Nothing                  => "Nothing"
    case p.Type.Var(name)                => s"#$name"
    case p.Type.Exec(tpeArgs, args, rtn) => ???
  }
}

extension (fn: p.Function) {

  def mangledName = fn.receiver.map(_.named.tpe.monomorphicName).getOrElse("") + "!" + fn.name.fqn
    .mkString("_") + "!" + fn.args.map(_.named.tpe.monomorphicName).mkString("_") + "!" + fn.rtn.monomorphicName

  def signature = p.Signature(
    fn.name,
    fn.tpeVars,
    fn.receiver.map(_.named.tpe),
    fn.args.map(_.named.tpe),
    fn.moduleCaptures.map(_.named.tpe),
    fn.termCaptures.map(_.named.tpe),
    fn.rtn
  )

  def signatureRepr = {
    import p.repr as _
    val termCaptures   = fn.termCaptures.map(a => s"${a.named.symbol}: ${typeReprOf(a.named.tpe)}").mkString(",")
    val moduleCaptures = fn.moduleCaptures.map(a => s"${a.named.symbol}: ${typeReprOf(a.named.tpe)}").mkString(",")
    val tpeVars        = fn.tpeVars.mkString(",")
    val args           = fn.args.map(a => s"${a.named.symbol}: ${typeReprOf(a.named.tpe)}").mkString(",")
    val recv           = fn.receiver.map(a => s"${a.named.symbol}: ${typeReprOf(a.named.tpe)}.").getOrElse("")
    s"${recv}${fn.name.fqn.mkString(".")}<$tpeVars>($args)[$moduleCaptures;${termCaptures}] : ${typeReprOf(fn.rtn)}"
  }
}

private def typeReprOf(t: p.Type): String = t match {
  case p.Type.Struct(name, args) => s"${name.fqn.mkString(".")}<${args.map(typeReprOf).mkString(",")}>"
  case p.Type.Ptr(c, s)          => s"${typeReprOf(c)}*$s"
  case p.Type.Arr(c, l, s)       => s"${typeReprOf(c)}[$l]$s"
  case p.Type.Var(name)          => s"#$name"
  case p.Type.Exec(tv, args, rtn) =>
    s"<${tv.mkString(",")}>(${args.map(typeReprOf).mkString(",")}) => ${typeReprOf(rtn)}"
  case p.Type.Float16 => "F16"
  case p.Type.Float32 => "F32"
  case p.Type.Float64 => "F64"
  case p.Type.IntU8   => "U8"
  case p.Type.IntU16  => "U16"
  case p.Type.IntU32  => "U32"
  case p.Type.IntU64  => "U64"
  case p.Type.IntS8   => "I8"
  case p.Type.IntS16  => "I16"
  case p.Type.IntS32  => "I32"
  case p.Type.IntS64  => "I64"
  case p.Type.Nothing => "Nothing"
  case p.Type.Unit0   => "Unit0"
  case p.Type.Bool1   => "Bool1"
}

def selectTerm(prefix: List[p.Named], last: p.Named): p.Term.Select = prefix match {
  case Nil    => p.Term.Select(last, Nil, last.tpe)
  case h :: t => p.Term.Select(h, t.map(n => p.PathStep.Field(n.symbol)) :+ p.PathStep.Field(last.symbol), last.tpe)
}

def selectExpr(prefix: List[p.Named], last: p.Named): p.Expr = p.Expr.Alias(selectTerm(prefix, last))

def asTerm(e: p.Expr): p.Term = e match {
  case p.Expr.Alias(t) => t
  case other =>
    throw IllegalStateException(s"asTerm called on non-atomic Expr: ${other.repr}")
}

object Builder {

  def bind(stmts: scala.collection.mutable.ListBuffer[p.Stmt], hint: String, e: p.Expr): p.Term = e match {
    case p.Expr.Alias(t) => t
    case other =>
      val n = p.Named(s"_${hint}_${stmts.size}", other.tpe)
      stmts += p.Stmt.Var(n, Some(other), isMutable = false)
      p.Term.Select(n, Nil, n.tpe)
  }

  def lift(t: p.Term): p.Expr = p.Expr.Alias(t)
}

val BytePtr: p.Type.Ptr = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)
val U64: p.Type         = p.Type.IntU64
val I64: p.Type         = p.Type.IntS64

def call(name: String, args: List[p.Term], rtn: p.Type): p.Expr = p.Expr.ForeignCall(name, args, rtn)
def sel(n: p.Named): p.Term.Select                              = selectTerm(Nil, n)
def vlet(name: String, tpe: p.Type, e: p.Expr): (p.Named, p.Stmt) = {
  val n = p.Named(name, tpe); (n, p.Stmt.Var(n, Some(e), isMutable = false))
}

def typedCapture(capture: p.Named, ptr: p.Type): (p.Named, p.Stmt) = vlet("typed", ptr, p.Expr.Cast(sel(capture), ptr))

def mapStmtsRec(stmts: List[p.Stmt])(leaf: p.Stmt => List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
  case p.Stmt.While(c, b)                => List(p.Stmt.While(c, mapStmtsRec(b)(leaf)))
  case p.Stmt.Cond(c, t, e)              => List(p.Stmt.Cond(c, mapStmtsRec(t)(leaf), mapStmtsRec(e)(leaf)))
  case p.Stmt.ForRange(i, lb, ub, st, b) => List(p.Stmt.ForRange(i, lb, ub, st, mapStmtsRec(b)(leaf)))
  case p.Stmt.Annotated(inner, pos, c)   => mapStmtsRec(List(inner))(leaf).map(p.Stmt.Annotated(_, pos, c))
  case s                                 => leaf(s)
}

def dropAliasDecls(stmts: List[p.Stmt], aliases: Set[String]): List[p.Stmt] = mapStmtsRec(stmts) {
  case p.Stmt.Var(n, _, _) if aliases(n.symbol) => Nil
  case s                                        => List(s)
}

def constIntValue(t: p.Term): Option[Long] = t match {
  case p.Term.IntS64Const(v) => Some(v)
  case p.Term.IntU64Const(v) => Some(v)
  case p.Term.IntS32Const(v) => Some(v.toLong)
  case p.Term.IntU32Const(v) => Some(v.toLong)
  case _                     => None
}
def isZeroConst(t: p.Term): Boolean = constIntValue(t).contains(0L)

def scalarBytes(t: p.Type): Option[Long] = t match {
  case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8                     => Some(1)
  case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16                 => Some(2)
  case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32                 => Some(4)
  case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 | _: p.Type.Ptr => Some(8)
  case _                                                              => None
}
def scalarBytesOr8(t: p.Type): Int = scalarBytes(t).getOrElse(8L).toInt

def captureRoot(entry: p.Function): Option[(p.Named, p.Type.Struct)] =
  (entry.receiver.toList ::: entry.args).map(_.named).collectFirst {
    case n @ p.Named(p.Conventions.ThisReceiver | p.Conventions.CaptureArg, p.Type.Ptr(s: p.Type.Struct, _)) => (n, s)
  }
