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
    case p.Type.Var(name)                => s"#$name"
    case p.Type.FnRef(name)              => s"&${name.fqn.mkString("_")}"
    case p.Type.Exec(tpeArgs, args, rtn) => ???
  }
}

private def validateCallableBinders(tpe: p.Type, path: String): List[String] = {
  val errors = List.newBuilder[String]
  def loop(tpe: p.Type, path: String): Unit = tpe match {
    case p.Type.Struct(_, args) => args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path type argument $index"))
    case p.Type.Ptr(comp, _)    => loop(comp, s"$path pointee")
    case p.Type.Arr(comp, _, _) => loop(comp, s"$path element")
    case p.Type.Exec(tpeVars, args, rtn) =>
      tpeVars
        .groupMapReduce(identity)(_ => 1)(_ + _)
        .collect { case (name, n) if n > 1 => name }
        .toList
        .sorted
        .foreach(name => errors += s"$path has duplicate callable type variable `$name`")
      tpeVars.zipWithIndex.foreach { case (name, index) =>
        if (name.trim.isEmpty) errors += s"$path callable type variable $index is empty"
      }
      args.zipWithIndex.foreach((tpe, index) => loop(tpe, s"$path callable argument $index"))
      loop(rtn, s"$path callable return")
    case _ => ()
  }
  loop(tpe, path)
  errors.result()
}

extension (decl: p.FunctionDecl) {
  def validate: List[String] = {
    val errors = List.newBuilder[String]

    decl.tpeVars
      .groupMapReduce(identity)(_ => 1)(_ + _)
      .collect { case (name, n) if n > 1 => name }
      .toList
      .sorted
      .foreach(name => errors += s"duplicate type variable `$name`")
    decl.tpeVars.zipWithIndex.foreach { case (name, index) =>
      if (name.trim.isEmpty) errors += s"type variable $index is empty"
    }

    (decl.receiver.toList ::: decl.args ::: decl.moduleCaptures ::: decl.termCaptures)
      .map(_.named.symbol)
      .groupMapReduce(identity)(_ => 1)(_ + _)
      .collect { case (name, n) if n > 1 => name }
      .toList
      .sorted
      .foreach(name => errors += s"duplicate parameter `$name`")

    def freeVars(tpe: p.Type, bound: Set[String] = Set.empty): List[String] = tpe match {
      case p.Type.Var(name) if !bound(name) => List(name)
      case p.Type.Struct(_, args)           => args.flatMap(freeVars(_, bound))
      case p.Type.Ptr(comp, _)              => freeVars(comp, bound)
      case p.Type.Arr(comp, _, _)           => freeVars(comp, bound)
      case p.Type.Exec(tpeVars, args, rtn) =>
        val inner = bound ++ tpeVars
        args.flatMap(freeVars(_, inner)) ::: freeVars(rtn, inner)
      case _ => Nil
    }

    val declared = decl.tpeVars.toSet
    val allTypes = decl.receiver.toList.map(_.named.tpe) :::
      decl.args.map(_.named.tpe) :::
      decl.moduleCaptures.map(_.named.tpe) :::
      decl.termCaptures.map(_.named.tpe) :::
      List(decl.rtn)
    allTypes.flatMap(freeVars(_)).distinct.sorted.filterNot(declared).foreach { name =>
      errors += s"undeclared type variable `$name`"
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

  def classifyArguments: Either[List[String], List[InterfaceBinding.ArgumentKind]] = {
    def parameters(expr: p.Arg.SizeExpr): Set[Int] = expr match {
      case p.Arg.SizeExpr.Param(index)  => Set(index)
      case p.Arg.SizeExpr.Const(_)      => Set.empty
      case p.Arg.SizeExpr.Add(lhs, rhs) => parameters(lhs) ++ parameters(rhs)
      case p.Arg.SizeExpr.Mul(lhs, rhs) => parameters(lhs) ++ parameters(rhs)
    }
    def extentParameters(extent: p.Arg.Extent): Set[Int] = extent match {
      case p.Arg.Extent.Elements(size) => parameters(size)
      case p.Arg.Extent.Bytes(size)    => parameters(size)
    }

    val errors       = List.newBuilder[String]
    val boundaryArgs = decl.receiver.toList ::: decl.args ::: decl.moduleCaptures ::: decl.termCaptures
    errors ++= decl.validate
    val referenced =
      boundaryArgs.flatMap(_.boundary.toList.flatMap(boundary => extentParameters(boundary.extent))).toSet
    val kinds = decl.args.zipWithIndex.map { case (arg, index) =>
      arg.named.tpe match {
        case signature: p.Type.Exec => InterfaceBinding.ArgumentKind.Callable(signature)
        case p.Type.Ptr(_, _) =>
          arg.boundary match {
            case Some(boundary) => InterfaceBinding.ArgumentKind.Buffer(boundary.access, boundary.extent)
            case None =>
              errors += s"pointer argument `${arg.named.symbol}` has no boundary"
              InterfaceBinding.ArgumentKind.Scalar
          }
        case _ if referenced(index) => InterfaceBinding.ArgumentKind.ExtentScalar
        case _                      => InterfaceBinding.ArgumentKind.Scalar
      }
    }
    val distinct = errors.result().distinct
    if (distinct.isEmpty) Right(kinds) else Left(distinct)
  }

  def bind(
      call: p.InvokeSignature,
      callableDecls: List[p.FunctionDecl]
  ): Either[List[String], InterfaceBinding.Binding] = {
    val errors = List.newBuilder[String]
    errors ++= decl.validate
    call.tpeArgs.zipWithIndex.foreach((tpe, index) =>
      errors ++= validateCallableBinders(tpe, s"call type argument $index")
    )
    call.receiver.foreach(tpe => errors ++= validateCallableBinders(tpe, "call receiver"))
    call.args.zipWithIndex.foreach((tpe, index) => errors ++= validateCallableBinders(tpe, s"call argument $index"))
    errors ++= validateCallableBinders(call.rtn, "call return")

    val matcher       = TypeMatcher(decl.tpeVars.toSet, reconcileAliases = true)
    val deferred      = List.newBuilder[(p.Type.Exec, p.Type, String, Option[Int])]
    var callableBinds = Map.empty[Int, p.Sym]

    def matchCall(expected: p.Type, actual: p.Type, path: String, callableIndex: Option[Int] = None): Unit =
      (expected, actual) match {
        case (expected: p.Type.Exec, actual: (p.Type.Exec | p.Type.FnRef)) =>
          deferred += ((expected, actual, path, callableIndex))
        case _ => matcher.unify(expected, actual, path)
      }

    if (decl.name != call.name)
      errors += s"symbol differs: expected ${decl.name.fqn.mkString(".")}, got ${call.name.fqn.mkString(".")}"

    if (call.tpeArgs.nonEmpty && call.tpeArgs.size != decl.tpeVars.size)
      errors += s"type-argument count differs: expected ${decl.tpeVars.size}, got ${call.tpeArgs.size}"
    decl.tpeVars.zip(call.tpeArgs).foreach { case (name, actual) =>
      matcher.bind(name, actual, s"type argument `$name`")
    }

    if (decl.moduleCaptures.nonEmpty || decl.termCaptures.nonEmpty)
      errors += "public declarations with explicit captures cannot be called directly"

    (decl.receiver, call.receiver) match {
      case (Some(expected), Some(actual)) => matchCall(expected.named.tpe, actual, "receiver")
      case (None, None)                   => ()
      case _                              => errors += "receiver presence differs"
    }

    if (decl.args.size != call.args.size)
      errors += s"argument count differs: expected ${decl.args.size}, got ${call.args.size}"
    decl.args.zip(call.args).zipWithIndex.foreach { case ((expected, actual), index) =>
      matchCall(expected.named.tpe, actual, s"argument $index `${expected.named.symbol}`", Some(index))
    }
    matchCall(decl.rtn, call.rtn, "return")
    errors ++= matcher.errors

    val resolvedTypes = Map.newBuilder[String, p.Type]
    decl.tpeVars.foreach { name =>
      matcher.bindings.get(name) match {
        case None => errors += s"declaration type variable `$name` is not bound by the call"
        case Some(tpe) =>
          substituteType(tpe, matcher.bindings, resolving = Set(name)) match {
            case Left(reason) => errors += s"declaration type variable `$name` is not concrete: $reason"
            case Right(value) => resolvedTypes += name -> value
          }
      }
    }
    val publicTypes   = resolvedTypes.result()
    val callableIndex = callableDecls.groupBy(_.name).view.mapValues(_.sortBy(_.toString)).toMap

    def compareCallable(expected: p.Type.Exec, actual: p.Type, path: String, argumentIndex: Option[Int]): Unit =
      substituteType(expected, publicTypes) match {
        case Left(reason) => errors += s"$path is not concrete: $reason"
        case Right(concreteExpected: p.Type.Exec) =>
          actual match {
            case concreteActual: p.Type.Exec =>
              val exact = TypeMatcher(Set.empty)
              exact.unify(concreteExpected, concreteActual, path)
              errors ++= exact.errors
            case p.Type.FnRef(name) =>
              val candidates = callableIndex.getOrElse(name, Nil)
              if (candidates.isEmpty)
                errors += s"$path references callable `${name.fqn.mkString(".")}` without a declaration"
              else {
                val attempts = candidates.map { candidate =>
                  val candidateErrors = List.newBuilder[String]
                  candidate.validate.foreach(error =>
                    candidateErrors += s"$path callable `${name.fqn.mkString(".")}`: $error"
                  )
                  if (concreteExpected.tpeVars.nonEmpty)
                    candidateErrors +=
                      s"$path generic callable declarations are not supported yet: ${concreteExpected.tpeVars.mkString(", ")}"
                  if (candidate.tpeVars.nonEmpty)
                    candidateErrors += s"$path callable `${name.fqn.mkString(".")}` is still generic"
                  if (
                    candidate.receiver.nonEmpty || candidate.moduleCaptures.nonEmpty || candidate.termCaptures.nonEmpty
                  )
                    candidateErrors +=
                      s"$path callable `${name.fqn.mkString(".")}` has an unsupported receiver or explicit captures"
                  val exact = TypeMatcher(Set.empty)
                  exact.unify(
                    concreteExpected,
                    p.Type.Exec(candidate.tpeVars, candidate.args.map(_.named.tpe), candidate.rtn),
                    path
                  )
                  candidateErrors ++= exact.errors
                  candidate -> candidateErrors.result().distinct
                }
                attempts.filter(_._2.isEmpty) match {
                  case List((_, _)) => argumentIndex.foreach(index => callableBinds += index -> name)
                  case Nil =>
                    errors += s"$path has no matching declaration for callable `${name.fqn.mkString(".")}`"
                    attempts.flatMap(_._2).distinct.foreach(errors += _)
                  case matches =>
                    errors += s"$path has ${matches.size} matching declarations for callable `${name.fqn.mkString(".")}`"
                }
              }
            case other => errors += s"$path callable differs: expected $concreteExpected, got $other"
          }
        case Right(other) => errors += s"$path did not resolve to a callable: $other"
      }

    deferred.result().foreach(compareCallable.tupled)
    val distinct = errors.result().distinct
    if (distinct.isEmpty) Right(InterfaceBinding.Binding(publicTypes, callableBinds)) else Left(distinct)
  }

  def conformsTo(
      publicDecl: p.FunctionDecl
  ): Either[List[String], InterfaceBinding.ImplementationBinding] = {
    val errors  = List.newBuilder[String]
    val matcher = TypeMatcher(decl.tpeVars.toSet)
    errors ++= publicDecl.validate.map(error => s"public declaration: $error")
    errors ++= decl.validate.map(error => s"implementation declaration: $error")

    def compareArg(expected: p.Arg, actual: p.Arg, path: String): Unit = {
      matcher.unify(expected.named.tpe, actual.named.tpe, path)
      if (expected.boundary != actual.boundary)
        errors += s"$path boundary differs: expected ${expected.boundary}, got ${actual.boundary}"
    }

    def compareArgs(expected: List[p.Arg], actual: List[p.Arg], path: String): Unit = {
      if (expected.size != actual.size)
        errors += s"$path count differs: expected ${expected.size}, got ${actual.size}"
      expected.zip(actual).zipWithIndex.foreach { case ((e, a), index) => compareArg(e, a, s"$path $index") }
    }

    if (publicDecl.affinity != decl.affinity)
      errors += s"affinity differs: expected ${publicDecl.affinity}, got ${decl.affinity}"

    (decl.receiver, publicDecl.receiver) match {
      case (Some(expected), Some(actual)) => compareArg(expected, actual, "receiver")
      case (None, None)                   => ()
      case _                              => errors += "receiver presence differs"
    }
    compareArgs(decl.moduleCaptures, publicDecl.moduleCaptures, "module capture")
    compareArgs(decl.termCaptures, publicDecl.termCaptures, "term capture")

    val result =
      if (decl.args.size == publicDecl.args.size) {
        compareArgs(decl.args, publicDecl.args, "argument")
        matcher.unify(decl.rtn, publicDecl.rtn, "return")
        InterfaceBinding.ResultBinding.Direct
      } else if (
        decl.args.size == publicDecl.args.size + 1 &&
        decl.rtn == p.Type.Unit0 &&
        publicDecl.rtn != p.Type.Unit0
      ) {
        compareArgs(decl.args.init, publicDecl.args, "argument")
        val resultIndex = publicDecl.args.size
        val resultArg   = decl.args.last
        resultArg.named.tpe match {
          case p.Type.Ptr(comp, p.Type.Space.Global) => matcher.unify(comp, publicDecl.rtn, "trailing result pointee")
          case p.Type.Ptr(_, space)                  => errors += s"trailing result pointer is not global: $space"
          case other                                 => errors += s"trailing result is not a pointer: $other"
        }
        val expectedBoundary = p.Arg.Boundary(
          p.Arg.Access.Write,
          p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))
        )
        if (!resultArg.boundary.contains(expectedBoundary))
          errors += s"trailing result boundary differs: expected $expectedBoundary, got ${resultArg.boundary}"
        InterfaceBinding.ResultBinding.TrailingOutput(resultIndex)
      } else {
        errors +=
          s"argument/result shape differs: public has ${publicDecl.args.size} arguments and returns ${publicDecl.rtn}; " +
            s"implementation has ${decl.args.size} arguments and returns ${decl.rtn}"
        InterfaceBinding.ResultBinding.Direct
      }

    errors ++= matcher.errors
    decl.tpeVars
      .filterNot(matcher.bindings.contains)
      .foreach(name => errors += s"implementation type variable `$name` is not bound by the public declaration")

    val callables = decl.args
      .zip(publicDecl.args)
      .zipWithIndex
      .collect {
        case ((p.Arg(p.Named(_, p.Type.Var(name), _), _, _), p.Arg(p.Named(_, _: p.Type.Exec, _), _, _)), index) =>
          name -> index
      }
      .toMap
    val distinct = errors.result().distinct
    if (distinct.isEmpty) Right(InterfaceBinding.ImplementationBinding(matcher.bindings, callables, result))
    else Left(distinct)
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

extension (index: p.PackageIndex) {
  def resolve(
      call: p.InvokeSignature,
      callableDecls: List[p.FunctionDecl],
      capabilities: Set[String],
      typeSizes: Map[p.Type, Int]
  ): Either[List[String], InterfaceBinding.Resolution] = {
    val decls = index.interface.decls.filter(_.name == call.name)
    if (decls.isEmpty)
      return Left(List(s"no public declaration `${call.name.fqn.mkString(".")}`"))
    val boundDecls = decls.map(decl => decl -> decl.bind(call, callableDecls))
    val matchingDecls = boundDecls.collect { case (decl, Right(binding)) =>
      decl -> binding
    }
    val (decl, callBinding) = matchingDecls match {
      case List(result) => result
      case Nil =>
        return Left(
          "no matching public declaration" :: boundDecls.flatMap { case (decl, result) =>
            result.left.toOption.toList.flatten.map(error => s"`${decl.toString}`: $error")
          }
        )
      case matches =>
        return Left(List(s"ambiguous public declaration `${call.name.fqn.mkString(".")}`: ${matches.size} matches"))
    }
    def candidateKey(candidate: p.ImplementationCandidate) = (
      candidate.implementation.name.fqn.mkString("."),
      candidate.implementation.toString,
      candidate.requiredCapabilities.sorted.mkString("\u0000"),
      candidate.typeSizes.sortBy(c => (c.typeVariable, c.sizeInBytes)).mkString("\u0000")
    )

    val candidates = index.candidates.filter(_.publicName == decl.name).sortBy(candidateKey)
    if (candidates.isEmpty)
      return Left(List(s"no implementations for `${decl.name.fqn.mkString(".")}`"))

    val compatible = List.newBuilder[(p.ImplementationCandidate, InterfaceBinding.ImplementationBinding)]
    val rejected   = List.newBuilder[String]
    candidates.foreach { candidate =>
      val label  = candidate.implementation.name.fqn.mkString(".")
      val errors = List.newBuilder[String]
      candidate.requiredCapabilities.distinct.sorted.filterNot(capabilities).foreach { capability =>
        errors += s"requires capability `$capability`"
      }
      val implementationBinding = candidate.implementation.conformsTo(decl)
      implementationBinding.left.foreach(_.foreach(errors += _))
      implementationBinding.foreach { binding =>
        candidate.typeSizes.sortBy(c => (c.typeVariable, c.sizeInBytes)).foreach { constraint =>
          binding.types.get(constraint.typeVariable) match {
            case None => errors += s"type-size constraint references unbound variable `${constraint.typeVariable}`"
            case Some(bound) =>
              substituteType(bound, callBinding.types) match {
                case Left(reason) => errors += s"cannot resolve `${constraint.typeVariable}`: $reason"
                case Right(tpe) =>
                  typeSizes.get(tpe) match {
                    case None => errors += s"has no layout for `${tpe.repr}`"
                    case Some(actual) if actual != constraint.sizeInBytes =>
                      errors +=
                        s"requires `${constraint.typeVariable}` size ${constraint.sizeInBytes}, got $actual for `${tpe.repr}`"
                    case _ => ()
                  }
              }
          }
        }
      }
      val distinct = errors.result().distinct
      implementationBinding match {
        case Right(binding) if distinct.isEmpty => compatible += candidate -> binding
        case _                                  => distinct.foreach(error => rejected += s"`$label`: $error")
      }
    }

    compatible.result() match {
      case List((candidate, implementation)) =>
        Right(InterfaceBinding.Resolution(decl, callBinding, candidate, implementation))
      case Nil =>
        Left(s"no compatible implementation for `${decl.name.fqn.mkString(".")}`" :: rejected.result())
      case matches =>
        Left(
          List(
            s"ambiguous implementations for `${decl.name.fqn.mkString(".")}`: ${matches
                .map(_._1.implementation.name.fqn.mkString("."))
                .mkString(", ")}"
          )
        )
    }
  }
}

extension (fn: p.Function) {

  def modifyDecl(f: p.FunctionDecl => p.FunctionDecl): p.Function =
    fn.copy(decl = f(fn.decl))

  def mangledName = fn.receiver.map(_.named.tpe.monomorphicName).getOrElse("") + "!" + fn.name.fqn
    .mkString("_") + "!" + fn.args.map(_.named.tpe.monomorphicName).mkString("_") + "!" + fn.rtn.monomorphicName

  def signature: p.Signature = fn.decl.signature

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

extension (ivk: p.Expr.Invoke) {
  def calleeSym: Option[p.Sym] = ivk.callee match {
    case p.Type.FnRef(s) => Some(s)
    case _               => None
  }
  def calleeName: p.Sym =
    calleeSym.getOrElse(throw IllegalStateException(s"callee is not a concrete function: ${ivk.callee}"))
}

private def typeReprOf(t: p.Type): String = t match {
  case p.Type.Struct(name, args) => s"${name.fqn.mkString(".")}<${args.map(typeReprOf).mkString(",")}>"
  case p.Type.Ptr(c, s)          => s"${typeReprOf(c)}*$s"
  case p.Type.Arr(c, l, s)       => s"${typeReprOf(c)}[$l]$s"
  case p.Type.Var(name)          => s"#$name"
  case p.Type.FnRef(name)        => s"&${name.fqn.mkString(".")}"
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

private def containsType(tpe: p.Type, variables: Set[String]): Boolean = tpe match {
  case p.Type.Var(name)            => variables(name)
  case p.Type.Struct(_, args)      => args.exists(containsType(_, variables))
  case p.Type.Ptr(component, _)    => containsType(component, variables)
  case p.Type.Arr(component, _, _) => containsType(component, variables)
  case p.Type.Exec(tpeVars, args, rtn) =>
    val unshadowed = variables -- tpeVars
    args.exists(containsType(_, unshadowed)) || containsType(rtn, unshadowed)
  case _ => false
}

private final class TypeMatcher(bindable: Set[String], reconcileAliases: Boolean = false) {
  private var env                           = Map.empty[String, p.Type]
  private var binding                       = Set.empty[String]
  private val diagnostics                   = List.newBuilder[String]
  private val defaultReport: String => Unit = diagnostics += _

  def bindings: Map[String, p.Type] = env
  def errors: List[String]          = diagnostics.result().distinct

  def bind(name: String, actual: p.Type, path: String, report: String => Unit = defaultReport): Unit =
    env.get(name) match {
      case None                                 => env += name -> actual
      case Some(existing) if existing == actual => ()
      case Some(existing) if binding(name) =>
        report(s"$path conflicts for `$name`: cyclic binding through $existing and $actual")
      case Some(existing) if !reconcileAliases =>
        TypeMatcher(Set.empty).unify(existing, actual, path, report = report)
      case Some(existing) =>
        binding += name
        try unify(existing, actual, path, report = report)
        finally binding -= name
    }

  def unify(
      expected: p.Type,
      actual: p.Type,
      path: String,
      localVariables: Map[String, String] = Map.empty,
      actualBoundVariables: Set[String] = Set.empty,
      report: String => Unit = defaultReport
  ): Unit = (expected, actual) match {
    case (p.Type.Var(name), p.Type.Var(actualName)) if localVariables.contains(name) =>
      val expectedName = localVariables(name)
      if (expectedName != actualName)
        report(s"$path callable type variable differs: expected `$expectedName`, got `$actualName`")
    case (p.Type.Var(name), _) if localVariables.contains(name) =>
      report(s"$path callable type variable `${localVariables(name)}` is not preserved by $actual")
    case (p.Type.Var(name), actual) if bindable(name) =>
      if (containsType(actual, actualBoundVariables))
        report(s"$path cannot bind `$name` to a type containing a callable-local type variable")
      else bind(name, actual, path, report)
    case (p.Type.Ptr(ec, es), p.Type.Ptr(ac, as)) =>
      if (es != as) report(s"$path pointer space differs: expected $es, got $as")
      unify(ec, ac, s"$path pointee", localVariables, actualBoundVariables, report)
    case (p.Type.Arr(ec, en, es), p.Type.Arr(ac, an, as)) =>
      if (en != an) report(s"$path array length differs: expected $en, got $an")
      if (es != as) report(s"$path array space differs: expected $es, got $as")
      unify(ec, ac, s"$path element", localVariables, actualBoundVariables, report)
    case (p.Type.Struct(en, ea), p.Type.Struct(an, aa)) =>
      if (en != an) report(s"$path struct differs: expected $en, got $an")
      if (ea.size != aa.size)
        report(s"$path struct type-argument count differs: expected ${ea.size}, got ${aa.size}")
      ea.zip(aa).zipWithIndex.foreach { case ((e, a), index) =>
        unify(e, a, s"$path type argument $index", localVariables, actualBoundVariables, report)
      }
    case (p.Type.Exec(etv, eargs, eret), p.Type.Exec(atv, aargs, aret)) =>
      if (etv.size != atv.size)
        report(s"$path callable type-parameter count differs: expected ${etv.size}, got ${atv.size}")
      val nestedVariables = localVariables ++ etv.zip(atv)
      val nestedActual    = actualBoundVariables ++ atv
      if (eargs.size != aargs.size)
        report(s"$path callable argument count differs: expected ${eargs.size}, got ${aargs.size}")
      eargs.zip(aargs).zipWithIndex.foreach { case ((e, a), index) =>
        unify(e, a, s"$path callable argument $index", nestedVariables, nestedActual, report)
      }
      unify(eret, aret, s"$path callable return", nestedVariables, nestedActual, report)
    case _ =>
      if (expected != actual) report(s"$path type differs: expected $expected, got $actual")
  }
}

private def substituteType(
    tpe: p.Type,
    bindings: Map[String, p.Type],
    boundVariables: Set[String] = Set.empty,
    resolving: Set[String] = Set.empty
): Either[String, p.Type] = tpe match {
  case p.Type.Var(name) if boundVariables(name) => Right(tpe)
  case p.Type.Var(name) if resolving(name)      => Left(s"cyclic reference to `$name`")
  case p.Type.Var(name) =>
    bindings.get(name) match {
      case Some(value) => substituteType(value, bindings, boundVariables, resolving + name)
      case None        => Left(s"unresolved type variable `$name`")
    }
  case p.Type.Struct(name, args) =>
    substituteAll(args, bindings, boundVariables, resolving).map(p.Type.Struct(name, _))
  case p.Type.Ptr(component, space) =>
    substituteType(component, bindings, boundVariables, resolving).map(p.Type.Ptr(_, space))
  case p.Type.Arr(component, length, space) =>
    substituteType(component, bindings, boundVariables, resolving).map(p.Type.Arr(_, length, space))
  case p.Type.Exec(typeVariables, args, rtn) =>
    val nested = boundVariables ++ typeVariables
    for {
      resolvedArgs <- substituteAll(args, bindings, nested, resolving)
      resolvedRtn  <- substituteType(rtn, bindings, nested, resolving)
    } yield p.Type.Exec(typeVariables, resolvedArgs, resolvedRtn)
  case other => Right(other)
}

private def substituteAll(
    values: List[p.Type],
    bindings: Map[String, p.Type],
    boundVariables: Set[String],
    resolving: Set[String]
): Either[String, List[p.Type]] =
  values.foldRight[Either[String, List[p.Type]]](Right(Nil)) { (value, result) =>
    for {
      head <- substituteType(value, bindings, boundVariables, resolving)
      tail <- result
    } yield head :: tail
  }
