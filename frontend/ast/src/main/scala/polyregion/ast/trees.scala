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

given Traversal[p.Signature, p.Type]    = Traversal.derived
given Traversal[p.Arg.Boundary, p.Type] = Traversal.empty
given Traversal[p.FunctionDecl, p.Type] = Traversal.derived
given Traversal[p.FunctionDecl, p.Term] = Traversal.empty
given Traversal[p.FunctionDecl, p.Expr] = Traversal.empty
given Traversal[p.FunctionDecl, p.Stmt] = Traversal.empty

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

extension (t: p.Stmt.Try) {
  def blocks: List[List[p.Stmt]] = t.body :: t.fin :: t.handlers.map(_.body)
  def mapBlocks(f: List[p.Stmt] => List[p.Stmt]): p.Stmt.Try =
    p.Stmt.Try(f(t.body), t.handlers.map(h => h.copy(body = f(h.body))), f(t.fin))
}

extension (sd: p.StructDef) {
  def applied(args: List[p.Type]): p.Type.Struct = p.Type.Struct(sd.name, args)
  def erasedTpe: p.Type.Struct =
    p.Type.Struct(sd.name, sd.tpeVars)
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

  def isFractional: Boolean = e.kind == Type.Kind.Fractional

  def isSigned: Boolean = e match {
    case p.Type.IntS8 | p.Type.IntS16 | p.Type.IntS32 | p.Type.IntS64 => true
    case _                                                            => false
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
    case p.Type.Var(name, size)          => size.fold(s"#$name")(n => s"#$name:size=$n")
    case p.Type.FnRef(name)              => s"&${name.fqn.mkString("_")}"
    case p.Type.Exec(tpeArgs, args, rtn) => ???
  }
}

private[polyregion] def validateCallableBinders(tpe: p.Type, path: String): List[String] = {
  val errors = List.newBuilder[String]
  def loop(tpe: p.Type, path: String, local: Map[String, p.Type.Var]): Unit = tpe match {
    case variable @ p.Type.Var(name, _) =>
      local.get(name).filter(_ != variable).foreach { binder =>
        errors += s"$path callable type variable `$name` differs from its binder: expected $binder, got $variable"
      }
    case p.Type.Struct(_, args) =>
      args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path type argument $index", local))
    case p.Type.Ptr(comp, _)    => loop(comp, s"$path pointee", local)
    case p.Type.Arr(comp, _, _) => loop(comp, s"$path element", local)
    case p.Type.Exec(tpeVars, args, rtn) =>
      tpeVars
        .groupMapReduce(_.name)(_ => 1)(_ + _)
        .collect { case (name, n) if n > 1 => name }
        .toList
        .sorted
        .foreach(name => errors += s"$path has duplicate callable type variable `$name`")
      tpeVars.zipWithIndex.foreach { case (variable, index) =>
        if (variable.name.trim.isEmpty) errors += s"$path callable type variable $index is empty"
        if (variable.exactSizeInBytes.nonEmpty)
          errors += s"$path callable type variable `${variable.name}` cannot have an exact-size constraint"
      }
      val nested = local ++ tpeVars.map(variable => variable.name -> variable)
      args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path callable argument $index", nested))
      loop(rtn, s"$path callable return", nested)
    case _ => ()
  }
  loop(tpe, path, Map.empty)
  errors.result()
}

extension (decl: p.FunctionDecl) {
  def validate: List[String] = {
    val errors = List.newBuilder[String]

    decl.tpeVars
      .groupMapReduce(_.name)(_ => 1)(_ + _)
      .collect { case (name, n) if n > 1 => name }
      .toList
      .sorted
      .foreach(name => errors += s"duplicate type variable `$name`")
    decl.tpeVars.zipWithIndex.foreach { case (variable, index) =>
      if (variable.name.trim.isEmpty) errors += s"type variable $index is empty"
      variable.exactSizeInBytes
        .filter(_ <= 0)
        .foreach(_ => errors += s"type variable `${variable.name}` exact size must be positive")
    }

    (decl.receiver.toList ::: decl.args ::: decl.moduleCaptures ::: decl.termCaptures)
      .map(_.named.symbol)
      .groupMapReduce(identity)(_ => 1)(_ + _)
      .collect { case (name, n) if n > 1 => name }
      .toList
      .sorted
      .foreach(name => errors += s"duplicate parameter `$name`")

    def freeVars(tpe: p.Type, bound: Set[String] = Set.empty): List[p.Type.Var] = tpe match {
      case variable @ p.Type.Var(name, _) if !bound(name) => List(variable)
      case p.Type.Struct(_, args)                         => args.flatMap(freeVars(_, bound))
      case p.Type.Ptr(comp, _)                            => freeVars(comp, bound)
      case p.Type.Arr(comp, _, _)                         => freeVars(comp, bound)
      case p.Type.Exec(tpeVars, args, rtn) =>
        val inner = bound ++ tpeVars.map(_.name)
        args.flatMap(freeVars(_, inner)) ::: freeVars(rtn, inner)
      case _ => Nil
    }

    val declared = decl.tpeVars.map(v => v.name -> v).toMap
    val allTypes = decl.receiver.toList.map(_.named.tpe) :::
      decl.args.map(_.named.tpe) :::
      decl.moduleCaptures.map(_.named.tpe) :::
      decl.termCaptures.map(_.named.tpe) :::
      List(decl.rtn)
    allTypes.flatMap(freeVars(_)).distinct.sortBy(_.name).foreach { variable =>
      declared.get(variable.name) match {
        case None => errors += s"undeclared type variable `${variable.name}`"
        case Some(binder) if binder != variable =>
          errors += s"type variable `${variable.name}` differs from its binder: expected $binder, got $variable"
        case _ => ()
      }
    }

    decl.receiver.foreach(arg => errors ++= validateCallableBinders(arg.named.tpe, "receiver"))
    decl.args.zipWithIndex.foreach((arg, index) =>
      errors ++= validateCallableBinders(arg.named.tpe, s"argument $index")
    )
    decl.moduleCaptures.zipWithIndex.foreach((arg, index) =>
      errors ++= validateCallableBinders(arg.named.tpe, s"module capture $index")
    )
    decl.termCaptures.zipWithIndex.foreach((arg, index) =>
      errors ++= validateCallableBinders(arg.named.tpe, s"term capture $index")
    )
    errors ++= validateCallableBinders(decl.rtn, "return")

    def validateSize(expr: p.Arg.SizeExpr, owner: String): Unit = expr match {
      case p.Arg.SizeExpr.Param(index) if index < 0 || index >= decl.args.size =>
        errors += s"$owner extent parameter $index is out of range for ${decl.args.size} arguments"
      case p.Arg.SizeExpr.Param(index) =>
        val tpe = decl.args(index).named.tpe
        if (tpe.kind != p.Type.Kind.Integral || tpe == p.Type.Bool1)
          errors += s"$owner extent parameter $index `${decl.args(index).named.symbol}` is not an integral scalar"
      case p.Arg.SizeExpr.Const(value) if value < 0 =>
        errors += s"$owner extent constant is negative: $value"
      case p.Arg.SizeExpr.Const(_) => ()
      case p.Arg.SizeExpr.Add(lhs, rhs) =>
        validateSize(lhs, owner)
        validateSize(rhs, owner)
      case p.Arg.SizeExpr.Mul(lhs, rhs) =>
        validateSize(lhs, owner)
        validateSize(rhs, owner)
      case p.Arg.SizeExpr.Min(lhs, rhs) =>
        validateSize(lhs, owner)
        validateSize(rhs, owner)
    }

    def validateBoundary(arg: p.Arg): Unit = arg.boundary.foreach { boundary =>
      arg.named.tpe match {
        case p.Type.Ptr(_, p.Type.Space.Constant) if boundary.access != p.Arg.Access.Read =>
          errors += s"argument `${arg.named.symbol}` writes through a constant pointer"
        case p.Type.Ptr(_, _) => ()
        case other            => errors += s"argument `${arg.named.symbol}` has a boundary but is not a pointer: $other"
      }
      boundary.extent match {
        case p.Arg.Extent.Elements(size) => validateSize(size, s"argument `${arg.named.symbol}`")
        case p.Arg.Extent.Bytes(size)    => validateSize(size, s"argument `${arg.named.symbol}`")
      }
    }

    decl.receiver.foreach(validateBoundary)
    decl.args.foreach(validateBoundary)
    decl.moduleCaptures.foreach(validateBoundary)
    decl.termCaptures.foreach(validateBoundary)
    errors.result().distinct
  }

  def validateInterfaceDeclaration: List[String] = {
    val errors = List.newBuilder[String]
    errors ++= decl.validate
    decl.args.foreach { arg =>
      arg.named.tpe match {
        case p.Type.Ptr(_, _) if arg.boundary.isEmpty =>
          errors += s"pointer argument `${arg.named.symbol}` has no boundary"
        case _ => ()
      }
    }
    errors.result().distinct
  }

  def signature: p.Signature = p.Signature(
    decl.name,
    decl.tpeVars,
    decl.receiver.map(_.named.tpe),
    decl.args.map(_.named.tpe),
    decl.moduleCaptures.map(_.named.tpe),
    decl.termCaptures.map(_.named.tpe),
    decl.rtn
  )

  def signatureKey: String = decl.signature.signatureKey

  def remapArgs(args: List[p.Arg]): p.FunctionDecl = {
    val newIndices = args.zipWithIndex.map((arg, index) => arg.named.symbol -> index).toMap

    def remapSize(size: p.Arg.SizeExpr): p.Arg.SizeExpr = size match {
      case p.Arg.SizeExpr.Param(index) =>
        val target = for {
          oldArg   <- decl.args.lift(index)
          newIndex <- newIndices.get(oldArg.named.symbol)
        } yield newIndex
        p.Arg.SizeExpr.Param(target.getOrElse(throw IllegalArgumentException(s"removed extent parameter $index")))
      case p.Arg.SizeExpr.Const(_)      => size
      case p.Arg.SizeExpr.Add(lhs, rhs) => p.Arg.SizeExpr.Add(remapSize(lhs), remapSize(rhs))
      case p.Arg.SizeExpr.Mul(lhs, rhs) => p.Arg.SizeExpr.Mul(remapSize(lhs), remapSize(rhs))
      case p.Arg.SizeExpr.Min(lhs, rhs) => p.Arg.SizeExpr.Min(remapSize(lhs), remapSize(rhs))
    }

    def remapArg(arg: p.Arg): p.Arg = arg.copy(boundary = arg.boundary.map { boundary =>
      val extent = boundary.extent match {
        case p.Arg.Extent.Elements(size) => p.Arg.Extent.Elements(remapSize(size))
        case p.Arg.Extent.Bytes(size)    => p.Arg.Extent.Bytes(remapSize(size))
      }
      boundary.copy(extent = extent)
    })

    decl.copy(
      receiver = decl.receiver.map(remapArg),
      args = args.map(remapArg),
      moduleCaptures = decl.moduleCaptures.map(remapArg),
      termCaptures = decl.termCaptures.map(remapArg)
    )
  }
}

extension (definition: p.StructDef) {
  def validate: List[String] = {
    val errors = List.newBuilder[String]
    definition.tpeVars
      .groupMapReduce(_.name)(_ => 1)(_ + _)
      .collect { case (name, n) if n > 1 => name }
      .toList
      .sorted
      .foreach(name => errors += s"duplicate type variable `$name`")
    definition.tpeVars.zipWithIndex.foreach { case (variable, index) =>
      if (variable.name.trim.isEmpty) errors += s"type variable $index is empty"
      variable.exactSizeInBytes
        .filter(_ <= 0)
        .foreach(_ => errors += s"type variable `${variable.name}` exact size must be positive")
    }

    def loop(tpe: p.Type, path: String, bound: Map[String, p.Type.Var]): Unit = tpe match {
      case variable @ p.Type.Var(name, _) =>
        bound.get(name) match {
          case None => errors += s"$path has undeclared type variable `$name`"
          case Some(binder) if binder != variable =>
            errors += s"$path type variable `$name` differs from its binder: expected $binder, got $variable"
          case _ => ()
        }
      case p.Type.Struct(_, args) =>
        args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path type argument $index", bound))
      case p.Type.Ptr(comp, _)    => loop(comp, s"$path pointee", bound)
      case p.Type.Arr(comp, _, _) => loop(comp, s"$path element", bound)
      case p.Type.Exec(tpeVars, args, rtn) =>
        tpeVars
          .groupMapReduce(_.name)(_ => 1)(_ + _)
          .collect { case (name, n) if n > 1 => name }
          .toList
          .sorted
          .foreach(name => errors += s"$path has duplicate callable type variable `$name`")
        tpeVars.zipWithIndex.foreach { case (variable, index) =>
          if (variable.name.trim.isEmpty) errors += s"$path callable type variable $index is empty"
          if (variable.exactSizeInBytes.nonEmpty)
            errors += s"$path callable type variable `${variable.name}` cannot have an exact-size constraint"
        }
        val nested = bound ++ tpeVars.map(variable => variable.name -> variable)
        args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path callable argument $index", nested))
        loop(rtn, s"$path callable return", nested)
      case _ => ()
    }

    val declared = definition.tpeVars.map(variable => variable.name -> variable).toMap
    definition.members.foreach(member => loop(member.tpe, s"member `${member.symbol}`", declared))
    definition.parents.zipWithIndex.foreach((parent, index) => loop(parent, s"parent $index", declared))
    errors.result().distinct
  }
}

extension (fn: p.Function) {

  def modifyDecl(f: p.FunctionDecl => p.FunctionDecl): p.Function =
    fn.copy(decl = f(fn.decl))

  def mangledName = fn.receiver.map(_.named.tpe.monomorphicName).getOrElse("") + "!" + fn.name.fqn
    .mkString("_") + "!" + fn.args.map(_.named.tpe.monomorphicName).mkString("_") + "!" + fn.rtn.monomorphicName

  def signature: p.Signature = fn.decl.signature

  def signatureKey: String = fn.signature.signatureKey
}

extension (ivk: p.Expr.Invoke) {
  def calleeSym: Option[p.Sym] = ivk.callee match {
    case p.Type.FnRef(s) => Some(s)
    case _               => None
  }
  def calleeName: p.Sym =
    calleeSym.getOrElse(throw IllegalStateException(s"callee is not a concrete function: ${ivk.callee}"))
}

extension (space: p.Type.Space) {
  def canonicalName: String = space match {
    case p.Type.Space.Global   => ""
    case p.Type.Space.Local    => "^Local"
    case p.Type.Space.Private  => "^Private"
    case p.Type.Space.Constant => "^Constant"
  }
}

extension (t: p.Type) {
  def canonicalName: String = t match {
    case p.Type.Struct(name, args) => s"${name.fqcn}<${args.map(_.canonicalName).mkString(",")}>"
    case p.Type.Ptr(c, s)          => s"${c.canonicalName}*${s.canonicalName}"
    case p.Type.Arr(c, l, s)       => s"${c.canonicalName}[$l]${s.canonicalName}"
    case p.Type.Var(name, size)    => size.fold(s"#$name")(n => s"#$name:size=$n")
    case p.Type.FnRef(name)        => s"&${name.fqcn}"
    case p.Type.Exec(tv, args, rtn) =>
      s"<${tv.map(_.canonicalName).mkString(",")}>(${args.map(_.canonicalName).mkString(",")}) => ${rtn.canonicalName}"
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
}

extension (signature: p.Signature) {
  def signatureKey: String = {
    def types(xs: List[p.Type]): String = xs.map(_.canonicalName).mkString(",")
    val receiver                        = signature.receiver.map(_.canonicalName + ".").getOrElse("")
    s"$receiver${signature.name.fqcn}<${types(signature.tpeVars)}>(${types(signature.args)})[${types(
        signature.moduleCaptures
      )};${types(signature.termCaptures)}]:${signature.rtn.canonicalName}"
  }
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

def defaultTerm(t: p.Type): p.Term = t match {
  case Type.Float16     => p.Term.Float16Const(0f)
  case Type.Float32     => p.Term.Float32Const(0f)
  case Type.Float64     => p.Term.Float64Const(0d)
  case Type.IntU8       => p.Term.IntU8Const(0)
  case Type.IntU16      => p.Term.IntU16Const(0)
  case Type.IntU32      => p.Term.IntU32Const(0)
  case Type.IntU64      => p.Term.IntU64Const(0)
  case Type.IntS8       => p.Term.IntS8Const(0)
  case Type.IntS16      => p.Term.IntS16Const(0)
  case Type.IntS32      => p.Term.IntS32Const(0)
  case Type.IntS64      => p.Term.IntS64Const(0)
  case Type.Bool1       => p.Term.Bool1Const(false)
  case Type.Unit0       => p.Term.Unit0Const
  case p.Type.Ptr(c, s) => p.Term.NullPtrConst(c, s, p.Region.Opaque)
  case other            => p.Term.Poison(other)
}

def defaultExpr(t: p.Type): p.Expr = p.Expr.Alias(defaultTerm(t))

def typedCapture(capture: p.Named, ptr: p.Type): (p.Named, p.Stmt) = vlet("typed", ptr, p.Expr.Cast(sel(capture), ptr))

def mapStmtsRec(stmts: List[p.Stmt])(leaf: p.Stmt => List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
  case p.Stmt.While(c, b)                => List(p.Stmt.While(c, mapStmtsRec(b)(leaf)))
  case p.Stmt.Cond(c, t, e)              => List(p.Stmt.Cond(c, mapStmtsRec(t)(leaf), mapStmtsRec(e)(leaf)))
  case p.Stmt.ForRange(i, lb, ub, st, b) => List(p.Stmt.ForRange(i, lb, ub, st, mapStmtsRec(b)(leaf)))
  case t: p.Stmt.Try                     => List(t.mapBlocks(mapStmtsRec(_)(leaf)))
  case p.Stmt.Raise(value, exceptionKind, cleanup) =>
    List(p.Stmt.Raise(value, exceptionKind, mapStmtsRec(cleanup)(leaf)))
  case p.Stmt.Annotated(inner, pos, c) => mapStmtsRec(List(inner))(leaf).map(p.Stmt.Annotated(_, pos, c))
  case s                               => leaf(s)
}

def dropAliasDecls(stmts: List[p.Stmt], aliases: Set[String]): List[p.Stmt] = mapStmtsRec(stmts) {
  case p.Stmt.Var(n, _, _) if aliases(n.symbol) => Nil
  case s                                        => List(s)
}

// every variable a tree reads as the root of a Select (the liveness / referenced-names seed used by
// dead-binding and dead-argument elimination)
def selectRoots[A](a: A)(using Traversal[A, p.Term]): Set[p.Named] =
  a.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root }.toSet

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
    case n @ p.Named(p.Conventions.ThisReceiver | p.Conventions.CaptureArg, p.Type.Ptr(s: p.Type.Struct, _), _) =>
      (n, s)
  }
