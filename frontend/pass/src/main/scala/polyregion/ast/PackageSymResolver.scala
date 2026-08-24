package polyregion.ast

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, given}
import polyregion.ast.pass.*

import scala.collection.mutable

private[polyregion] object PackageSymResolver {

  final case class ResolvedImplementation(functions: List[p.Function], definitions: Set[p.StructDef])
  private final case class ResolvedEntry(entry: p.Function, entryArgs: List[p.Package.EntryArgBinding])

  private def containsType(tpe: p.Type, variables: Set[String]): Boolean = tpe match {
    case p.Type.Var(name, _)         => variables(name)
    case p.Type.Struct(_, args)      => args.exists(containsType(_, variables))
    case p.Type.Ptr(component, _)    => containsType(component, variables)
    case p.Type.Arr(component, _, _) => containsType(component, variables)
    case p.Type.Exec(tpeVars, args, rtn) =>
      val unshadowed = variables -- tpeVars.map(_.name)
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
      case (p.Type.Var(name, _), p.Type.Var(actualName, _)) if localVariables.contains(name) =>
        val expectedName = localVariables(name)
        if (expectedName != actualName)
          report(s"$path callable type variable differs: expected `$expectedName`, got `$actualName`")
      case (p.Type.Var(name, _), _) if localVariables.contains(name) =>
        report(s"$path callable type variable `${localVariables(name)}` is not preserved by $actual")
      case (p.Type.Var(name, _), actual) if bindable(name) =>
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
        val nestedVariables = localVariables ++ etv.map(_.name).zip(atv.map(_.name))
        val nestedActual    = actualBoundVariables ++ atv.map(_.name)
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
    case p.Type.Var(name, _) if boundVariables(name) => Right(tpe)
    case p.Type.Var(name, _) if resolving(name)      => Left(s"cyclic reference to `$name`")
    case p.Type.Var(name, _) =>
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
      val nested = boundVariables ++ typeVariables.map(_.name)
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

  def bindCall(
      decl: p.FunctionDecl,
      signature: p.InvokeSignature,
      callerDecls: List[p.FunctionDecl],
      matchResult: Boolean = true
  ): Either[List[String], InterfaceBinding.BoundCall] = {
    val errors = List.newBuilder[String]
    errors ++= decl.validate
    signature.tpeArgs.zipWithIndex.foreach((tpe, index) =>
      errors ++= validateCallableBinders(tpe, s"signature type argument $index")
    )
    signature.receiver.foreach(tpe => errors ++= validateCallableBinders(tpe, "signature receiver"))
    signature.args.zipWithIndex.foreach((tpe, index) =>
      errors ++= validateCallableBinders(tpe, s"signature argument $index")
    )
    errors ++= validateCallableBinders(signature.rtn, "signature return")

    val matcher       = TypeMatcher(decl.tpeVars.map(_.name).toSet, reconcileAliases = true)
    val deferred      = List.newBuilder[(p.Type.Exec, p.Type.FnRef, String, Option[Int])]
    var callableBinds = Map.empty[Int, p.Sym]

    def matchCall(expected: p.Type, actual: p.Type, path: String, callableIndex: Option[Int] = None): Unit =
      (expected, actual) match {
        case (expected: p.Type.Exec, actual: p.Type.FnRef) =>
          deferred += ((expected, actual, path, callableIndex))
        case _ => matcher.unify(expected, actual, path)
      }

    if (decl.name != signature.name)
      errors += s"symbol differs: expected ${decl.name.fqn.mkString(".")}, got ${signature.name.fqn.mkString(".")}"
    if (signature.tpeArgs.nonEmpty && signature.tpeArgs.size != decl.tpeVars.size)
      errors += s"type-argument count differs: expected ${decl.tpeVars.size}, got ${signature.tpeArgs.size}"
    decl.tpeVars.zip(signature.tpeArgs).foreach { case (variable, actual) =>
      matcher.bind(variable.name, actual, s"type argument `${variable.name}`")
    }
    if (decl.moduleCaptures.nonEmpty || decl.termCaptures.nonEmpty)
      errors += "public declarations with explicit captures cannot be called directly"
    (decl.receiver, signature.receiver) match {
      case (Some(expected), Some(actual)) => matchCall(expected.named.tpe, actual, "receiver")
      case (None, None)                   => ()
      case _                              => errors += "receiver presence differs"
    }
    if (decl.args.size != signature.args.size)
      errors += s"argument count differs: expected ${decl.args.size}, got ${signature.args.size}"
    decl.args.zip(signature.args).zipWithIndex.foreach { case ((expected, actual), index) =>
      matchCall(expected.named.tpe, actual, s"argument $index `${expected.named.symbol}`", Some(index))
    }
    if (matchResult) matchCall(decl.rtn, signature.rtn, "return")
    val callableIndex = callerDecls.groupBy(_.name).view.mapValues(_.sortBy(_.signatureKey)).toMap
    deferred.result().foreach { case (expected, p.Type.FnRef(name), path, argumentIndex) =>
      val candidates = callableIndex.getOrElse(name, Nil)
      if (candidates.isEmpty)
        errors += s"$path references callable `${name.fqn.mkString(".")}` without a declaration"
      else {
        val attempts = candidates.map { candidate =>
          val candidateErrors = List.newBuilder[String]
          candidate.validate.foreach(error => candidateErrors += s"$path callable `${name.fqn.mkString(".")}`: $error")
          if (expected.tpeVars.nonEmpty)
            candidateErrors += s"$path generic callable declarations are not supported yet: ${expected.tpeVars.mkString(", ")}"
          if (candidate.tpeVars.nonEmpty)
            candidateErrors += s"$path callable `${name.fqn.mkString(".")}` is still generic"
          if (candidate.receiver.nonEmpty || candidate.moduleCaptures.nonEmpty || candidate.termCaptures.nonEmpty)
            candidateErrors += s"$path callable `${name.fqn.mkString(".")}` has an unsupported receiver or explicit captures"
          val candidateMatcher = TypeMatcher(decl.tpeVars.map(_.name).toSet, reconcileAliases = true)
          matcher.bindings.foreach { case (variable, value) =>
            candidateMatcher.bind(variable, value, path, candidateErrors += _)
          }
          candidateMatcher.unify(
            expected,
            p.Type.Exec(candidate.tpeVars, candidate.args.map(_.named.tpe), candidate.rtn),
            path,
            report = candidateErrors += _
          )
          candidate -> (candidateMatcher.bindings, candidateErrors.result().distinct)
        }
        attempts.filter(_._2._2.isEmpty) match {
          case List((_, (bindings, _))) =>
            bindings.foreach { case (variable, value) => matcher.bind(variable, value, path, errors += _) }
            argumentIndex.foreach(index => callableBinds += index -> name)
          case Nil =>
            errors += s"$path has no matching declaration for callable `${name.fqn.mkString(".")}`"
            attempts.flatMap(_._2._2).distinct.foreach(errors += _)
          case matches =>
            errors += s"$path has ${matches.size} matching declarations for callable `${name.fqn.mkString(".")}`"
        }
      }
    }
    errors ++= matcher.errors

    val resolvedTypes = Map.newBuilder[String, p.Type]
    decl.tpeVars.foreach { variable =>
      val name = variable.name
      matcher.bindings.get(name) match {
        case None => errors += s"declaration type variable `$name` is not bound by the signature"
        case Some(tpe) =>
          substituteType(tpe, matcher.bindings, resolving = Set(name)) match {
            case Left(reason) => errors += s"declaration type variable `$name` is not concrete: $reason"
            case Right(value) => resolvedTypes += name -> value
          }
      }
    }
    val publicTypes = resolvedTypes.result()
    val distinct    = errors.result().distinct
    if (distinct.isEmpty) Right(InterfaceBinding.BoundCall(publicTypes, callableBinds)) else Left(distinct)
  }

  private def shiftSize(size: p.Arg.SizeExpr, by: Int): p.Arg.SizeExpr = size match {
    case p.Arg.SizeExpr.Param(index)  => p.Arg.SizeExpr.Param(index + by)
    case value: p.Arg.SizeExpr.Const  => value
    case p.Arg.SizeExpr.Add(lhs, rhs) => p.Arg.SizeExpr.Add(shiftSize(lhs, by), shiftSize(rhs, by))
    case p.Arg.SizeExpr.Mul(lhs, rhs) => p.Arg.SizeExpr.Mul(shiftSize(lhs, by), shiftSize(rhs, by))
    case p.Arg.SizeExpr.Min(lhs, rhs) => p.Arg.SizeExpr.Min(shiftSize(lhs, by), shiftSize(rhs, by))
  }

  private def contextualBoundary(boundary: Option[p.Arg.Boundary], by: Int): Option[p.Arg.Boundary] =
    boundary.map { value =>
      val extent = value.extent match {
        case p.Arg.Extent.Elements(size) => p.Arg.Extent.Elements(shiftSize(size, by))
        case p.Arg.Extent.Bytes(size)    => p.Arg.Extent.Bytes(shiftSize(size, by))
      }
      value.copy(extent = extent)
    }

  def bindImplementation(
      implementation: p.FunctionDecl,
      publicDecl: p.FunctionDecl
  ): Either[List[String], InterfaceBinding.BoundImplementation] = {
    val errors  = List.newBuilder[String]
    val matcher = TypeMatcher(implementation.tpeVars.map(_.name).toSet)
    errors ++= publicDecl.validate.map(error => s"public declaration: $error")
    errors ++= implementation.validate.map(error => s"implementation declaration: $error")

    def compareArg(expected: p.Arg, actual: p.Arg, path: String): Unit = {
      matcher.unify(expected.named.tpe, actual.named.tpe, path)
      if (expected.boundary != actual.boundary)
        errors += s"$path boundary differs: expected ${actual.boundary}, got ${expected.boundary}"
    }

    def compareArgs(expected: List[p.Arg], actual: List[p.Arg], path: String): Unit = {
      if (expected.size != actual.size)
        errors += s"$path count differs: expected ${expected.size}, got ${actual.size}"
      expected.zip(actual).zipWithIndex.foreach { case ((e, a), index) => compareArg(e, a, s"$path $index") }
    }

    if (publicDecl.affinity != implementation.affinity)
      errors += s"affinity differs: expected ${publicDecl.affinity}, got ${implementation.affinity}"
    (implementation.receiver, publicDecl.receiver) match {
      case (Some(expected), Some(actual)) => compareArg(expected, actual, "receiver")
      case (None, None)                   => ()
      case _                              => errors += "receiver presence differs"
    }
    compareArgs(implementation.moduleCaptures, publicDecl.moduleCaptures, "module capture")
    compareArgs(implementation.termCaptures, publicDecl.termCaptures, "term capture")

    val systemArguments = if (implementation.args.headOption.exists(_.named.symbol == "#context")) 1 else 0
    implementation.args.headOption.filter(_.named.symbol == "#context").foreach { argument =>
      if (argument.named.tpe != p.Spec.ContextType) errors += "context system argument has the wrong type"
    }
    val comparable = implementation.args.drop(systemArguments)
    val (ordinary, result) =
      if (comparable.size == publicDecl.args.size) {
        matcher.unify(implementation.rtn, publicDecl.rtn, "return")
        comparable -> InterfaceBinding.ReturnConvention.Direct
      } else if (
        comparable.size == publicDecl.args.size + 1 &&
        implementation.rtn == p.Type.Unit0 &&
        publicDecl.rtn != p.Type.Unit0
      ) {
        val resultIndex = systemArguments + publicDecl.args.size
        val resultArg   = comparable.last
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
        comparable.init -> InterfaceBinding.ReturnConvention.TrailingOutput(resultIndex)
      } else {
        errors +=
          s"argument/result shape differs: public has ${publicDecl.args.size} arguments and returns ${publicDecl.rtn}; " +
            s"implementation has ${comparable.size} public arguments and returns ${implementation.rtn}"
        comparable.take(publicDecl.args.size) -> InterfaceBinding.ReturnConvention.Direct
      }
    if (ordinary.size != publicDecl.args.size)
      errors += s"argument count differs: expected ${publicDecl.args.size}, got ${ordinary.size}"
    ordinary.zip(publicDecl.args).zipWithIndex.foreach { case ((implementationArg, publicArg), index) =>
      matcher.unify(implementationArg.named.tpe, publicArg.named.tpe, s"argument $index")
      val expectedBoundary = contextualBoundary(publicArg.boundary, systemArguments)
      if (implementationArg.boundary != expectedBoundary)
        errors += s"argument $index boundary differs: expected $expectedBoundary, got ${implementationArg.boundary}"
    }

    errors ++= matcher.errors
    implementation.tpeVars
      .filterNot(variable => matcher.bindings.contains(variable.name))
      .foreach(variable =>
        errors += s"implementation type variable `${variable.name}` is not bound by the public declaration"
      )
    val callables = ordinary
      .zip(publicDecl.args)
      .zipWithIndex
      .collect {
        case ((p.Arg(p.Named(_, p.Type.Var(name, _), _), _, _), p.Arg(p.Named(_, _: p.Type.Exec, _), _, _)), index) =>
          name -> index
      }
      .toMap
    val distinct = errors.result().distinct
    if (distinct.isEmpty)
      Right(InterfaceBinding.BoundImplementation(matcher.bindings, callables, result, systemArguments))
    else Left(distinct)
  }

  def resolve(
      pkg: p.Package,
      signature: p.InvokeSignature,
      callerDecls: List[p.FunctionDecl],
      capabilities: Set[String],
      typeSizes: Map[p.Type, Int],
      matchResult: Boolean = true
  ): Either[List[String], InterfaceBinding.ResolvedCall] = {
    val decls = pkg.interface.declarations.filter(_.name == signature.name)
    if (decls.isEmpty)
      return Left(List(s"no public declaration `${signature.name.fqn.mkString(".")}`"))
    val boundDecls    = decls.map(decl => decl -> bindCall(decl, signature, callerDecls, matchResult))
    val matchingDecls = boundDecls.collect { case (decl, Right(binding)) => decl -> binding }
    val (decl, signatureBinding) = matchingDecls match {
      case List(result) => result
      case Nil =>
        return Left(
          "no matching public declaration" :: boundDecls.flatMap { case (candidate, result) =>
            result.left.toOption.toList.flatten.map(error => s"`${candidate.toString}`: $error")
          }
        )
      case matches =>
        return Left(
          List(s"ambiguous public declaration `${signature.name.fqn.mkString(".")}`: ${matches.size} matches")
        )
    }
    def implementationKey(function: p.Function) = (
      function.signatureKey,
      function.requiredCapabilities.sorted.mkString("\u0000")
    )
    val implementations = pkg.program.functions.filter(_.implements.contains(decl.name)).sortBy(implementationKey)
    if (implementations.isEmpty)
      return Left(List(s"no implementations for `${decl.name.fqn.mkString(".")}`"))

    val compatible = List.newBuilder[(p.Function, InterfaceBinding.BoundImplementation)]
    val rejected   = List.newBuilder[String]
    implementations.foreach { implementation =>
      val label  = implementation.name.fqn.mkString(".")
      val errors = List.newBuilder[String]
      implementation.requiredCapabilities.distinct.sorted.filterNot(capabilities).foreach { capability =>
        errors += s"requires capability `$capability`"
      }
      val implementationBinding = bindImplementation(implementation.decl, decl)
      implementationBinding.left.foreach(_.foreach(errors += _))
      implementationBinding.foreach { binding =>
        val constraints = implementation.tpeVars.flatMap(variable => variable.exactSizeInBytes.map(variable -> _))
        val constrained = constraints.map(_._1.name)
        val sizeable    = binding.types.keySet -- binding.callables.keySet
        if (constrained.distinct.size != constrained.size)
          errors += "type-size constraints must be distinct"
        if (constrained.nonEmpty && constrained.toSet != sizeable)
          errors += s"type-size constraints must cover `${sizeable.toList.sorted.mkString(", ")}`"
        constraints.sortBy(_._1.name).foreach { case (variable, requiredSize) =>
          if (requiredSize <= 0)
            errors += s"type-size constraint for `${variable.name}` must be positive"
          binding.types.get(variable.name) match {
            case None => errors += s"type-size constraint references unbound variable `${variable.name}`"
            case Some(bound) =>
              substituteType(bound, signatureBinding.types) match {
                case Left(reason) => errors += s"cannot resolve `${variable.name}`: $reason"
                case Right(tpe) =>
                  typeSizes.get(tpe) match {
                    case None => errors += s"has no layout for `${tpe.repr}`"
                    case Some(actual) if actual != requiredSize =>
                      errors += s"requires `${variable.name}` size $requiredSize, got $actual for `${tpe.repr}`"
                    case _ => ()
                  }
              }
          }
        }
      }
      val distinct = errors.result().distinct
      implementationBinding match {
        case Right(binding) if distinct.isEmpty => compatible += implementation -> binding
        case _                                  => distinct.foreach(error => rejected += s"`$label`: $error")
      }
    }

    val matches     = compatible.result()
    val specialised = matches.filter(_._1.tpeVars.exists(_.exactSizeInBytes.nonEmpty))
    val selected    = if (specialised.nonEmpty) specialised else matches
    selected match {
      case List((implementation, binding)) =>
        Right(InterfaceBinding.ResolvedCall(decl, signatureBinding, implementation, binding))
      case Nil => Left(s"no compatible implementation for `${decl.name.fqn.mkString(".")}`" :: rejected.result())
      case values =>
        Left(
          List(
            s"ambiguous implementations for `${decl.name.fqn.mkString(".")}`: ${values
                .map(_._1.name.fqn.mkString("."))
                .mkString(", ")}"
          )
        )
    }
  }

  private def substitute(tpe: p.Type, bindings: Map[String, p.Type], bound: Set[String] = Set.empty): p.Type =
    tpe match {
      case p.Type.Var(name, _) if !bound(name)  => bindings.get(name).map(substitute(_, bindings, bound)).getOrElse(tpe)
      case p.Type.Struct(name, args)            => p.Type.Struct(name, args.map(substitute(_, bindings, bound)))
      case p.Type.Ptr(component, space)         => p.Type.Ptr(substitute(component, bindings, bound), space)
      case p.Type.Arr(component, length, space) => p.Type.Arr(substitute(component, bindings, bound), length, space)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = bound ++ vars.map(_.name)
        p.Type.Exec(vars, args.map(substitute(_, bindings, nested)), substitute(rtn, bindings, nested))
      case _ => tpe
    }

  private def functionClosure(program: p.Program, root: p.Function): Either[List[String], List[p.Function]] = {
    @annotation.tailrec
    def loop(
        frontier: List[p.Function],
        reached: Set[p.Sym],
        out: List[p.Function]
    ): Either[List[String], List[p.Function]] = frontier match {
      case Nil                                        => Right(out.reverse)
      case function :: rest if reached(function.name) => loop(rest, reached, out)
      case function :: rest =>
        val names        = function.collectAll[p.Type].collect { case ref: p.Type.FnRef => ref.name }.distinct
        val dependencies = names.map(name => name -> program.functions.filter(_.name == name))
        val ambiguous = dependencies.collect {
          case (name, matches) if matches.size != 1 =>
            s"function `${name.repr}` has ${matches.size} package definitions"
        }
        if (ambiguous.nonEmpty) Left(ambiguous)
        else loop(dependencies.flatMap(_._2) ::: rest, reached + function.name, function :: out)
    }
    loop(List(root), Set.empty, Nil)
  }

  private def structClosure(program: p.Program, functions: List[p.Function]): Set[p.StructDef] = {
    @annotation.tailrec
    def loop(frontier: List[p.Sym], reached: Set[p.Sym], out: Set[p.StructDef]): Set[p.StructDef] = frontier match {
      case Nil                           => out
      case name :: rest if reached(name) => loop(rest, reached, out)
      case name :: rest =>
        val definitions = program.defs.filter(_.name == name)
        val next        = definitions.flatMap(_.collectAll[p.Type].collect { case p.Type.Struct(struct, _) => struct })
        loop(next ::: rest, reached + name, out ++ definitions)
    }
    val roots = functions.flatMap(_.collectAll[p.Type].collect { case p.Type.Struct(name, _) => name })
    loop(roots, Set.empty, Set.empty)
  }

  private def implementationClosure(
      pkg: p.Package,
      resolution: InterfaceBinding.ResolvedCall,
      callerFns: List[p.Function]
  ): Either[List[String], List[p.Function]] = {
    val errors = List.newBuilder[String]
    val callableVariables = resolution.implementationBinding.callables.flatMap { case (name, index) =>
      resolution.signature.callables.get(index).map(name -> _)
    }
    val implementationVariables = resolution.implementationBinding.types.map { case (name, tpe) =>
      val bound = callableVariables.get(name).fold(substitute(tpe, resolution.signature.types))(p.Type.FnRef(_))
      name -> bound
    }
    val candidates = pkg.program.functions ::: callerFns
    val byName     = candidates.groupBy(_.name)
    byName.foreach { case (name, functions) =>
      if (!functions.forall(_ == functions.head))
        errors += s"function `${name.repr}` conflicts between package and caller"
    }
    val selectedCount = pkg.program.functions.count(_.decl == resolution.implementation.decl)
    if (selectedCount != 1)
      errors += (if (selectedCount == 0) "selected implementation is absent"
                 else "selected implementation is ambiguous")
    callableVariables.values.foreach { name =>
      if (!byName.contains(name)) errors += s"bound callable `${name.repr}` is absent"
    }
    val roots = callableVariables.values.toList :+ resolution.implementation.name
    @annotation.tailrec
    def loop(frontier: List[p.Sym], reached: Set[p.Sym], out: List[p.Function]): List[p.Function] = frontier match {
      case Nil                           => out.reverse
      case name :: rest if reached(name) => loop(rest, reached, out)
      case name :: rest =>
        byName.get(name).flatMap(_.headOption) match {
          case None =>
            errors += s"function `${name.repr}` is absent"
            loop(rest, reached + name, out)
          case Some(original) =>
            val selected = original.decl == resolution.implementation.decl
            val withoutCallables =
              if (!selected) original
              else {
                val removed = original.args.zipWithIndex.collect {
                  case (argument, index)
                      if index >= resolution.implementationBinding.systemArguments &&
                        resolution.signature.callables.contains(
                          index - resolution.implementationBinding.systemArguments
                        ) =>
                    argument.named.symbol
                }.toSet
                original
                  .collectAll[p.Term]
                  .collect { case select: p.Term.Select => select }
                  .filter { select =>
                    removed(select.root.symbol)
                  }
                  .foreach(select =>
                    errors += s"callable placeholder `${select.root.symbol}` is used as a runtime value"
                  )
                original.copy(
                  decl = original.decl.copy(args = original.args.zipWithIndex.collect {
                    case (argument, index)
                        if index < resolution.implementationBinding.systemArguments ||
                          !resolution.signature.callables
                            .contains(index - resolution.implementationBinding.systemArguments) =>
                      argument
                  }),
                  implements = None,
                  requiredCapabilities = Nil
                )
              }
            val bindings =
              if (selected) implementationVariables else implementationVariables -- withoutCallables.tpeVars.map(_.name)
            val function     = withoutCallables.modifyAll[p.Type](substitute(_, bindings))
            val dependencies = function.collectAll[p.Type].collect { case p.Type.FnRef(target) => target }
            loop(dependencies ::: rest, reached + name, function :: out)
        }
    }
    val closed = loop(roots, Set.empty, Nil)
    callableVariables.values.foreach { name =>
      if (!closed.exists(_.name == name)) errors += s"bound callable `${name.repr}` is unreachable"
    }
    val distinct = errors.result().distinct
    Either.cond(distinct.isEmpty, closed, distinct)
  }

  private def boundStructClosure(
      pkg: p.Package,
      functions: List[p.Function],
      callerDefs: List[p.StructDef]
  ): Either[List[String], List[p.StructDef]] = {
    val errors = List.newBuilder[String]
    val out    = List.newBuilder[p.StructDef]
    @annotation.tailrec
    def loop(frontier: List[p.Type.Struct], reached: Set[p.Sym]): Unit = frontier match {
      case Nil => ()
      case applied :: rest =>
        val matches = (pkg.program.defs ::: callerDefs).filter(_.name == applied.name)
        matches.headOption match {
          case None =>
            errors += s"struct definition `${applied.name.repr}` is absent"
            loop(rest, reached)
          case Some(selected) =>
            if (matches.exists(_ != selected))
              errors += s"struct definition `${applied.name.repr}` conflicts between package and caller"
            if (selected.tpeVars.size != applied.args.size)
              errors +=
                s"struct `${applied.name.repr}` type-argument count differs: expected ${selected.tpeVars.size}, got ${applied.args.size}"
            if (reached(applied.name)) loop(rest, reached)
            else {
              selected.validate.foreach(error => errors += s"struct definition `${applied.name.repr}`: $error")
              out += selected
              val nested = selected.collectAll[p.Type].collect { case value: p.Type.Struct => value }
              loop(nested ::: rest, reached + applied.name)
            }
        }
    }
    val roots = functions.flatMap(_.collectAll[p.Type].collect { case value: p.Type.Struct => value })
    loop(roots, Set.empty)
    val distinct = errors.result().distinct
    Either.cond(distinct.isEmpty, out.result(), distinct)
  }

  private def materializeEntry(
      name: String,
      resolution: InterfaceBinding.ResolvedCall,
      typeSizes: Map[p.Type, Int],
      returnConvention: p.Package.ReturnConvention
  ): Either[List[String], ResolvedEntry] = {
    val errors          = List.newBuilder[String]
    val publicDecl      = resolution.publicDeclaration
    val implementation  = resolution.implementation.decl
    val concreteTypes   = publicDecl.args.map(arg => substitute(arg.named.tpe, resolution.signature.types))
    val entryArgs       = mutable.ListBuffer.empty[p.Arg]
    val argumentSources = mutable.ListBuffer[p.Package.EntryArgBinding](p.Package.EntryArgBinding.Context)
    val body            = mutable.ListBuffer.empty[p.Stmt]
    val downloads       = mutable.ListBuffer.empty[p.Stmt]
    val frees           = mutable.ListBuffer.empty[p.Stmt]
    val invokeArgs      = mutable.ListBuffer.empty[p.Term]
    val sourceArgs      = mutable.Map.empty[Int, p.Named]
    val scalarValues    = mutable.Map.empty[Int, p.Named]

    val outParamIndex = returnConvention match {
      case p.Package.ReturnConvention.Return          => None
      case p.Package.ReturnConvention.OutParam(index) => Some(index)
    }
    def sourceIndex(index: Int): Int = outParamIndex.fold(index)(out => if (index < out) index else index + 1)

    def select(named: p.Named): p.Term.Select           = p.Term.Select(named, Nil, named.tpe)
    def spec(op: p.Spec): p.Expr                        = p.Expr.SpecOp(op)
    def intr(op: p.Intr): p.Expr                        = p.Expr.IntrOp(op)
    def immutable(named: p.Named, expr: p.Expr): Unit   = body += p.Stmt.Var(named, Some(expr), isMutable = false)
    def temporary(prefix: String, tpe: p.Type): p.Named = p.Named(s"$prefix${body.size}", tpe)

    val contextType = p.Spec.ContextType
    val context     = p.Named("#context", contextType)
    entryArgs += p.Arg(context)
    if (resolution.implementationBinding.systemArguments != 0) invokeArgs += select(context)
    publicDecl.args.indices.foreach { index =>
      if (!resolution.signature.callables.contains(index)) {
        val concrete = concreteTypes(index)
        val abiType = concrete match {
          case _: p.Type.Ptr => concrete
          case _             => p.Type.Ptr(concrete, p.Type.Space.Global)
        }
        val named = p.Named(s"a$index", abiType)
        sourceArgs(index) = named
        entryArgs += p.Arg(named)
        argumentSources += (concrete match {
          case _: p.Type.Ptr => p.Package.EntryArgBinding.CallValue(sourceIndex(index))
          case _             => p.Package.EntryArgBinding.CallAddress(sourceIndex(index))
        })
      }
    }
    publicDecl.args.indices.foreach { index =>
      if (!resolution.signature.callables.contains(index) && !concreteTypes(index).isInstanceOf[p.Type.Ptr]) {
        val concrete = concreteTypes(index)
        val named    = p.Named(s"v$index", concrete)
        immutable(named, p.Expr.Index(select(sourceArgs(index)), p.Term.IntS32Const(0), concrete))
        scalarValues(index) = named
      }
    }

    def extent(expr: p.Arg.SizeExpr): p.Term = expr match {
      case p.Arg.SizeExpr.Param(index) =>
        scalarValues.get(index) match {
          case None =>
            errors += s"extent references unavailable argument $index"
            p.Term.IntU64Const(0)
          case Some(value) =>
            val named = temporary(s"extentParam$index", p.Type.IntU64)
            immutable(named, p.Expr.Cast(select(value), p.Type.IntU64))
            select(named)
        }
      case p.Arg.SizeExpr.Const(value) => p.Term.IntU64Const(value)
      case p.Arg.SizeExpr.Add(lhs, rhs) =>
        val lhsValue = extent(lhs)
        val rhsValue = extent(rhs)
        val named    = temporary("extent", p.Type.IntU64)
        immutable(named, intr(p.Intr.Add(lhsValue, rhsValue, p.Type.IntU64)))
        select(named)
      case p.Arg.SizeExpr.Mul(lhs, rhs) =>
        val lhsValue = extent(lhs)
        val rhsValue = extent(rhs)
        val named    = temporary("extent", p.Type.IntU64)
        immutable(named, intr(p.Intr.Mul(lhsValue, rhsValue, p.Type.IntU64)))
        select(named)
      case p.Arg.SizeExpr.Min(lhs, rhs) =>
        val lhsValue = extent(lhs)
        val rhsValue = extent(rhs)
        val named    = temporary("extent", p.Type.IntU64)
        immutable(named, intr(p.Intr.Min(lhsValue, rhsValue, p.Type.IntU64)))
        select(named)
    }

    publicDecl.args.indices.foreach { index =>
      if (!resolution.signature.callables.contains(index)) {
        concreteTypes(index) match {
          case pointer: p.Type.Ptr =>
            publicDecl.args(index).boundary match {
              case None => errors += s"pointer argument `${publicDecl.args(index).named.symbol}` has no boundary"
              case Some(boundary) =>
                val count = boundary.extent match {
                  case p.Arg.Extent.Bytes(size) => extent(size)
                  case p.Arg.Extent.Elements(size) =>
                    val elements = extent(size)
                    typeSizes.get(pointer.comp) match {
                      case None =>
                        errors += s"has no layout for `${pointer.comp.repr}`"
                        p.Term.IntU64Const(0)
                      case Some(width) =>
                        val named = temporary(s"bytes$index", p.Type.IntU64)
                        immutable(
                          named,
                          intr(p.Intr.Mul(elements, p.Term.IntU64Const(width.toLong), p.Type.IntU64))
                        )
                        select(named)
                    }
                }
                val remote = p.Named(s"remote$index", p.Type.Ptr(p.Type.IntU8, p.Type.Space.Global))
                body += p.Stmt.Var(remote, Some(spec(p.Spec.RemoteAlloc(select(context), count))), isMutable = true)
                val typed = p.Named(s"p$index", pointer)
                immutable(typed, p.Expr.Cast(select(remote), pointer))
                invokeArgs += select(typed)
                if (boundary.access == p.Arg.Access.Read || boundary.access == p.Arg.Access.ReadWrite)
                  body += p.Stmt.Var(
                    p.Named(s"upload$index", p.Type.Unit0),
                    Some(
                      spec(
                        p.Spec.RemoteMemcpy(
                          select(context),
                          select(remote),
                          select(sourceArgs(index)),
                          count,
                          p.Direction.LocalToRemote
                        )
                      )
                    ),
                    isMutable = true
                  )
                if (boundary.access == p.Arg.Access.Write || boundary.access == p.Arg.Access.ReadWrite)
                  downloads += p.Stmt.Var(
                    p.Named(s"download$index", p.Type.Unit0),
                    Some(
                      spec(
                        p.Spec.RemoteMemcpy(
                          select(context),
                          select(sourceArgs(index)),
                          select(remote),
                          count,
                          p.Direction.RemoteToLocal
                        )
                      )
                    ),
                    isMutable = true
                  )
                frees += p.Stmt.Var(
                  p.Named(s"free$index", p.Type.Unit0),
                  Some(spec(p.Spec.RemoteFree(select(context), select(remote)))),
                  isMutable = true
                )
            }
          case _ => invokeArgs += select(scalarValues(index))
        }
      }
    }

    val tpeArgs = implementation.tpeVars.flatMap { variable =>
      resolution.implementationBinding.types.get(variable.name) match {
        case None =>
          errors += s"implementation type variable `${variable.name}` is not bound"
          Nil
        case Some(bound) =>
          val publicType = substitute(bound, resolution.signature.types)
          publicType match {
            case _: p.Type.Exec =>
              val callable = for {
                index  <- resolution.implementationBinding.callables.get(variable.name)
                symbol <- resolution.signature.callables.get(index)
              } yield symbol
              callable match {
                case Some(symbol) => List(p.Type.FnRef(symbol))
                case None =>
                  errors += s"callable type variable `${variable.name}` has no bound function"
                  Nil
              }
            case value => List(value)
          }
      }
    }
    val concreteResult = substitute(publicDecl.rtn, resolution.signature.types)
    val returnsValue   = concreteResult != p.Type.Unit0
    val result         = Option.when(returnsValue)(p.Named("result", p.Type.Ptr(concreteResult, p.Type.Space.Global)))
    result.foreach { named =>
      entryArgs += p.Arg(named)
      argumentSources += (returnConvention match {
        case p.Package.ReturnConvention.Return          => p.Package.EntryArgBinding.ResultAddress
        case p.Package.ReturnConvention.OutParam(index) => p.Package.EntryArgBinding.CallValue(index)
      })
    }
    resolution.implementationBinding.result match {
      case InterfaceBinding.ReturnConvention.TrailingOutput(_) => result.foreach(named => invokeArgs += select(named))
      case InterfaceBinding.ReturnConvention.Direct            => ()
    }
    val invokeResult = resolution.implementationBinding.result match {
      case InterfaceBinding.ReturnConvention.TrailingOutput(_) => p.Type.Unit0
      case InterfaceBinding.ReturnConvention.Direct            => concreteResult
    }
    val invoke = p.Expr.Invoke(p.Type.FnRef(implementation.name), tpeArgs, None, invokeArgs.toList, invokeResult)
    (returnsValue, resolution.implementationBinding.result) match {
      case (true, InterfaceBinding.ReturnConvention.Direct) =>
        val callResult = p.Named("callResult", concreteResult)
        immutable(callResult, invoke)
        body += p.Stmt.Update(select(result.get), p.Term.IntS32Const(0), select(callResult))
      case _ => body += p.Stmt.Var(p.Named("invoke", invokeResult), Some(invoke), isMutable = true)
    }
    body ++= downloads
    body ++= frees
    body += p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))

    val distinct = errors.result().distinct
    val declaration = p.FunctionDecl(
      p.Sym(name),
      Nil,
      None,
      entryArgs.toList,
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val entry = p.Function(
      declaration,
      body.toList,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall
    )
    Either.cond(distinct.isEmpty, ResolvedEntry(entry, argumentSources.toList), distinct)
  }

  def resolveSym(request: p.Package.SymRequest): Either[List[String], p.Package.SymResolvedProgram] = {
    val groupedTypeSizes = request.typeSizes.groupBy(_.tpe)
    val layoutErrors = groupedTypeSizes.toList
      .flatMap { case (tpe, values) =>
        val sizes = values.map(_.sizeInBytes).distinct
        Option.when(sizes.exists(_ <= 0))(s"layout for `${tpe.repr}` must be positive") ::
          Option.when(sizes.size != 1)(s"layout for `${tpe.repr}` conflicts: ${sizes.sorted.mkString(", ")}") :: Nil
      }
      .flatten
      .distinct
    val typeSizes = groupedTypeSizes.view.mapValues(_.head.sizeInBytes).toMap
    val returnConventionErrors = request.returnConvention match {
      case p.Package.ReturnConvention.Return => Nil
      case p.Package.ReturnConvention.OutParam(index) =>
        Option.when(index < 0 || index > request.signature.args.size)(
          s"result output parameter index $index is outside source argument range 0..${request.signature.args.size}"
        ) ::
          Option.when(request.signature.rtn != p.Type.Nothing)(
            s"result output parameter requires an erased ${p.Type.Nothing.repr} signature return, got ${request.signature.rtn.repr}"
          ) :: Nil
    }
    for {
      _ <- Either.cond(returnConventionErrors.flatten.isEmpty, (), returnConventionErrors.flatten)
      _ <- Either.cond(layoutErrors.isEmpty, (), layoutErrors)
      resolution <- resolve(
        request.pkg,
        request.signature,
        request.callerDecls,
        request.capabilities.toSet,
        typeSizes,
        matchResult = request.returnConvention == p.Package.ReturnConvention.Return
      )
      closure     <- implementationClosure(request.pkg, resolution, request.callerFns)
      definitions <- boundStructClosure(request.pkg, closure, request.callerDefs)
      entry       <- materializeEntry(request.entryName, resolution, typeSizes, request.returnConvention)
    } yield {
      val initial     = p.Program(Some(entry.entry), closure, definitions)
      val specialised = Specialisation(initial, PluginEntry.defaultLog)
      val monomorphic = MonoStruct(specialised, PluginEntry.defaultLog)._2
      val prepared = DeadStructElimination(
        DeadFunctionElimination(
          KernelCaptureFlatten(OffloadEntryInline(monomorphic, PluginEntry.defaultLog), PluginEntry.defaultLog),
          PluginEntry.defaultLog
        ),
        PluginEntry.defaultLog
      )
      p.Package.SymResolvedProgram(prepared, entry.entryArgs)
    }
  }

  def resolveImplementation(
      pkg: p.Package,
      declaration: String,
      target: p.Expr.Invoke,
      callerDecls: List[p.FunctionDecl] = Nil,
      capabilities: Set[String] = Set.empty,
      typeSizes: Map[p.Type, Int] = Map.empty
  ): Either[List[String], ResolvedImplementation] = {
    val signature = p.InvokeSignature(
      p.Sym(declaration),
      target.tpeArgs,
      None,
      target.args.map(_.tpe),
      target.rtn
    )
    for {
      resolution <- resolve(pkg, signature, callerDecls, capabilities, typeSizes)
      selected = resolution.implementation
      _ <- Either.cond(target.tpeArgs.isEmpty, (), List("generic Scala interface calls are not yet supported"))
      _ <- Either.cond(
        resolution.implementationBinding.result == InterfaceBinding.ReturnConvention.Direct,
        (),
        List("Scala interface resolution does not support trailing-output implementations")
      )
      callableBindings <- resolution.implementationBinding.callables.toList
        .traverse { case (name, index) =>
          resolution.signature.callables
            .get(index)
            .toRight(List(s"callable binding `$name` has no signature argument at index $index"))
            .map(name -> _)
        }
        .map(_.toMap)
      bindings = resolution.implementationBinding.types.view
        .mapValues(substitute(_, resolution.signature.types))
        .toMap ++ callableBindings.view.mapValues(p.Type.FnRef(_))
      closed <- functionClosure(pkg.program, selected)
      functions = closed
        .map(_.modifyAll[p.Type](substitute(_, bindings)))
        .map(_.modifyAll[p.Expr] {
          case invoke: p.Expr.Invoke =>
            invoke.callee match {
              case p.Type.Var(name, _) =>
                callableBindings.get(name).fold(invoke)(symbol => invoke.copy(callee = p.Type.FnRef(symbol)))
              case _ => invoke
            }
          case expression => expression
        })
      renamed = functions.map(_.modifyAll[p.Type] {
        case p.Type.FnRef(name) if name == selected.name => p.Type.FnRef(target.calleeName)
        case tpe                                         => tpe
      })
      resolved = renamed.map { function =>
        if (function.name != selected.name) function
        else
          function.copy(decl =
            function.decl.copy(
              name = target.calleeName,
              tpeVars = Nil,
              receiver = target.receiver.map(value => p.Arg(p.Named("this", value.tpe), None))
            )
          )
      }
    } yield ResolvedImplementation(resolved, structClosure(pkg.program, resolved))
  }
}
