package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.pass.TypeOps.sameType
import polyregion.ast.{PolyAST as p, *, given}

// monomorphises generic functions: one specialised copy per distinct tpeArg set reached from entry,
// rewrites each Invoke to the monomorphic name + drops tpeArgs, then removes generic templates
// examples:
//   id[Int](3); id[Float](1.0)  ->  Int_id(3); Float_id(1.0)   (+ two specialised copies, template dropped)
//   id[Int](3); id[Int](7)      ->  Int_id(3); Int_id(7)       (+ one specialised copy, deduped)
//   pair[Int,Float](3, 1.0)     ->  Int_Float_pair(3, 1.0)     (+ one specialised copy, template dropped)
// edge cases:
//   specialisation that invokes further generics -> recursiveSpecialise recurses into the new body
//   tpeArg set already specialised               -> deduped by monomorphic name (no second copy)
//   Invoke of an unknown / non-generic name      -> left untouched (no tpeArgs -> not rewritten)
object Specialisation extends ProgramPass {

  private case class Callsite(
      calleeName: p.Sym,
      tpeArgs: List[p.Type],
      receiver: Option[p.Type],
      args: List[p.Type],
      rtn: Option[p.Type]
  ) {
    def remote: Boolean = rtn.isEmpty
  }

  private case class Candidate(
      tpeArgs: List[p.Type],
      templateKey: String,
      identityKey: String,
      function: p.Function
  )

  private def appliedTypeArgs(receiver: Option[p.Term], tpeArgs: List[p.Type]): List[p.Type] =
    if (tpeArgs.nonEmpty) tpeArgs
    else
      receiver.toList.flatMap(_.tpe match {
        case p.Type.Struct(_, args)                => args
        case p.Type.Ptr(p.Type.Struct(_, args), _) => args
        case _                                     => Nil
      })

  private def containsFreeVariable(tpe: p.Type, variables: Set[String], bound: Set[String] = Set.empty): Boolean =
    tpe match {
      case p.Type.Var(name, _)    => variables(name) && !bound(name)
      case p.Type.Struct(_, args) => args.exists(containsFreeVariable(_, variables, bound))
      case p.Type.Ptr(comp, _)    => containsFreeVariable(comp, variables, bound)
      case p.Type.Arr(comp, _, _) => containsFreeVariable(comp, variables, bound)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = bound ++ vars.map(_.name)
        args.exists(containsFreeVariable(_, variables, nested)) || containsFreeVariable(rtn, variables, nested)
      case _ => false
    }

  private def bindAll(
      patterns: List[p.Type],
      actuals: List[p.Type],
      bindings: Map[String, p.Type],
      variables: Set[String]
  ): Option[Map[String, p.Type]] =
    patterns.zip(actuals).foldLeft(Option(bindings)) { case (acc, (expected, found)) =>
      acc.flatMap(bind(expected, found, _, variables))
    }

  private def bind(
      pattern: p.Type,
      actual: p.Type,
      bindings: Map[String, p.Type],
      variables: Set[String]
  ): Option[Map[String, p.Type]] =
    pattern match {
      case p.Type.Var(name, _) if variables(name) =>
        bindings.get(name) match {
          case Some(bound) if !sameType(bound, actual) => None
          case Some(_)                                 => Some(bindings)
          case None                                    => Some(bindings.updated(name, actual))
        }
      case p.Type.Struct(name, args) =>
        actual match {
          case p.Type.Struct(`name`, actualArgs) if args.size == actualArgs.size =>
            bindAll(args, actualArgs, bindings, variables)
          case _ => None
        }
      case p.Type.Ptr(comp, space) =>
        actual match {
          case p.Type.Ptr(actualComp, `space`) => bind(comp, actualComp, bindings, variables)
          case _                               => None
        }
      case p.Type.Arr(comp, length, space) =>
        actual match {
          case p.Type.Arr(actualComp, `length`, `space`) => bind(comp, actualComp, bindings, variables)
          case _                                         => None
        }
      case p.Type.Exec(patternVars, args, rtn) =>
        actual match {
          case p.Type.Exec(actualVars, actualArgs, actualRtn)
              if patternVars.size == actualVars.size && args.size == actualArgs.size =>
            val env             = actualVars.map(_.name).zip(patternVars.map(_.name)).toMap
            val alignedArgs     = actualArgs.map(Binders.renameVars(_, env))
            val alignedRtn      = Binders.renameVars(actualRtn, env)
            val nestedVariables = variables -- patternVars.map(_.name)
            bindAll(args, alignedArgs, bindings, nestedVariables).flatMap(bind(rtn, alignedRtn, _, nestedVariables))
          case _ => None
        }
      case _ => Option.when(sameType(pattern, actual))(bindings)
    }

  private def launchParams(kernel: p.Function): List[p.Arg] =
    (kernel.moduleCaptures ::: kernel.termCaptures ::: kernel.args)
      .filterNot(arg => arg.named.tpe == p.Type.Unit0 || arg.named.tpe == p.Type.Nothing)

  private def callsiteParams(fn: p.Function, remote: Boolean): List[p.Arg] =
    if (remote) launchParams(fn) else fn.args

  private def normalizedLaunchType(expected: p.Type, actual: p.Type): p.Type = (expected, actual) match {
    case (p.Type.Struct(expectedName, expectedArgs), p.Type.Ptr(struct: p.Type.Struct, _))
        if expectedName == struct.name && expectedArgs.size == struct.args.size =>
      struct
    case (p.Type.Ptr(struct: p.Type.Struct, space), actualStruct: p.Type.Struct)
        if struct.name == actualStruct.name && struct.args.size == actualStruct.args.size =>
      p.Type.Ptr(actualStruct, space)
    case _ => actual
  }

  private def normalizeLaunchArgument(expected: p.Type, actual: p.Term): p.Term =
    (expected, actual.tpe, actual) match {
      case (
            p.Type.Struct(expectedName, expectedArgs),
            p.Type.Ptr(struct: p.Type.Struct, _),
            select: p.Term.Select
          ) if expectedName == struct.name && expectedArgs.size == struct.args.size =>
        p.Term.Select(select.root, select.steps :+ p.PathStep.Deref, struct)
      case _ => actual
    }

  private def isScalar(tpe: p.Type): Boolean = tpe match {
    case p.Type.Float16 | p.Type.Float32 | p.Type.Float64 | p.Type.IntU8 | p.Type.IntU16 | p.Type.IntU32 |
        p.Type.IntU64 | p.Type.IntS8 | p.Type.IntS16 | p.Type.IntS32 | p.Type.IntS64 | p.Type.Bool1 =>
      true
    case _ => false
  }

  private def compatibleArgumentType(expected: p.Type, actual: p.Type, allowScalarConversion: Boolean): Boolean = {
    val normalized = normalizedLaunchType(expected, actual)
    sameType(expected, normalized) ||
    ((expected, normalized) match {
      case (p.Type.Ptr(p.Type.Nothing, _), p.Type.FnRef(_)) => true
      case _                                                => false
    }) ||
    (allowScalarConversion && isScalar(expected) && isScalar(normalized))
  }

  private def inferLaunchArguments(
      kernel: p.Function,
      args: List[p.Type],
      compatible: (p.Type, p.Type) => Boolean
  ): Option[List[p.Type]] = {
    val params = launchParams(kernel)
    if (
      params.size != args.size || kernel.tpeVars.isEmpty || kernel.affinity != p.Function.Affinity.Offload ||
      kernel.receiver.nonEmpty || kernel.rtn != p.Type.Unit0
    ) None
    else {
      val variables = kernel.tpeVars.map(_.name).toSet
      params
        .zip(args)
        .foldLeft(Option(Map.empty[String, p.Type])) { case (acc, (param, arg)) =>
          val normalized = normalizedLaunchType(param.named.tpe, arg)
          if (!containsFreeVariable(param.named.tpe, variables))
            acc.filter(_ => compatible(param.named.tpe, normalized))
          else acc.flatMap(bind(param.named.tpe, normalized, _, variables))
        }
        .flatMap(bindings =>
          Option.when(kernel.tpeVars.forall(variable => bindings.contains(variable.name)))(
            kernel.tpeVars.map(variable => bindings(variable.name))
          )
        )
    }
  }

  private def inferExactLaunchArguments(kernel: p.Function, args: List[p.Type]): Option[List[p.Type]] =
    inferLaunchArguments(kernel, args, compatibleArgumentType(_, _, allowScalarConversion = false))

  private def inferScalarLaunchArguments(kernel: p.Function, args: List[p.Type]): Option[List[p.Type]] =
    inferLaunchArguments(kernel, args, compatibleArgumentType(_, _, allowScalarConversion = true))

  private def inferLaunchArguments(fn: p.Function, overloads: Map[p.Sym, List[p.Function]]): p.Function = {
    val hasRemoteLaunch = fn
      .collectFirst_[p.Expr] { case p.Expr.SpecOp(p.Spec.RemoteLaunch(_, _, _, _, _, _, _, _, _, _, _)) => true }
      .isDefined
    if (!hasRemoteLaunch) return fn

    val inferred = fn.modifyAll[p.Expr] {
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.isEmpty =>
        val inferred = launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            val kernels = overloads
              .getOrElse(name, Nil)
            val argTypes = launch.args.map(_.tpe)
            val exact = kernels
              .flatMap(kernel => inferExactLaunchArguments(kernel, argTypes))
              .distinct
            val matches =
              if (exact.nonEmpty) exact
              else
                kernels
                  .flatMap(kernel => inferScalarLaunchArguments(kernel, argTypes))
                  .distinct
            matches match {
              case arguments :: Nil => Some(arguments)
              case _                => None
            }
          case _ => None
        }
        p.Expr.SpecOp(inferred.fold(launch)(arguments => launch.copy(tpeArgs = arguments)))
      case expr => expr
    }

    var temporaryIndex = 0
    val usedSymbols = scala.collection.mutable.HashSet.from(
      (inferred.receiver.toList ::: inferred.moduleCaptures ::: inferred.termCaptures ::: inferred.args)
        .map(_.named.symbol)
    )
    def recordDeclarations(statement: p.Stmt): Unit = statement match {
      case p.Stmt.Var(name, _, _) => usedSymbols += name.symbol
      case p.Stmt.While(_, body)  => body.foreach(recordDeclarations)
      case p.Stmt.ForRange(induction = induction, body = body) =>
        usedSymbols += induction.symbol
        body.foreach(recordDeclarations)
      case p.Stmt.Cond(_, trueBr, falseBr) =>
        trueBr.foreach(recordDeclarations)
        falseBr.foreach(recordDeclarations)
      case p.Stmt.Annotated(inner, _, _) => recordDeclarations(inner)
      case p.Stmt.Try(body, handlers, fin) =>
        body.foreach(recordDeclarations)
        handlers.foreach { handler =>
          handler.binder.foreach(named => usedSymbols += named.symbol)
          handler.body.foreach(recordDeclarations)
        }
        fin.foreach(recordDeclarations)
      case p.Stmt.Raise(_, _, cleanup) => cleanup.foreach(recordDeclarations)
      case _                           => ()
    }
    inferred.body.foreach(recordDeclarations)
    def temporary(tpe: p.Type): p.Named = {
      while (usedSymbols(s"#launch_cast_${temporaryIndex}")) temporaryIndex += 1
      val named = p.Named(s"#launch_cast_${temporaryIndex}", tpe)
      temporaryIndex += 1
      usedSymbols += named.symbol
      named
    }
    def normalize(launch: p.Spec.RemoteLaunch): (List[p.Stmt], p.Spec.RemoteLaunch) =
      launch.kernel.tpe match {
        case p.Type.FnRef(name) =>
          val callsite    = Callsite(name, launch.tpeArgs, None, launch.args.map(_.tpe), None)
          val specialised = candidates(callsite, overloads).map(_.function)
          val concrete = overloads
            .getOrElse(name, Nil)
            .filter(function =>
              function.tpeVars.isEmpty && launch.tpeArgs.isEmpty && erasedCallableBindings(callsite, function).isEmpty
            )
          val available = (specialised ::: concrete).distinct
          val exact     = available.filter(matches(callsite, _, allowScalarConversion = false))
          val resolved =
            if (exact.nonEmpty) exact
            else available.filter(matches(callsite, _, allowScalarConversion = true))
          resolved match {
            case kernel :: Nil =>
              val parameters = launchParams(kernel).map(_.named.tpe)
              val (preludeReversed, argumentsReversed) =
                parameters.zip(launch.args).foldLeft(List.empty[p.Stmt] -> List.empty[p.Term]) {
                  case ((statements, terms), (expected, actual)) =>
                    val normalized = normalizeLaunchArgument(expected, actual)
                    if (sameType(expected, normalized.tpe)) statements -> (normalized :: terms)
                    else if (isScalar(expected) && isScalar(normalized.tpe)) {
                      val named = temporary(expected)
                      (p.Stmt.Var(named, Some(p.Expr.Cast(normalized, expected))) :: statements) ->
                        (p.Term.Select(named, Nil, expected) :: terms)
                    } else statements -> (normalized :: terms)
                }
              preludeReversed.reverse -> launch.copy(args = argumentsReversed.reverse)
            case _ => Nil -> launch
          }
        case _ => Nil -> launch
      }

    def normalizeExpr(expr: p.Expr): (List[p.Stmt], p.Expr) = expr match {
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
        val (prelude, normalized) = normalize(launch)
        prelude -> p.Expr.SpecOp(normalized)
      case _ => Nil -> expr
    }

    def rewrite(body: List[p.Stmt]): List[p.Stmt] = body.flatMap {
      case p.Stmt.Var(name, Some(expr), isMutable) =>
        val (prelude, normalized) = normalizeExpr(expr)
        prelude :+ p.Stmt.Var(name, Some(normalized), isMutable)
      case p.Stmt.Mut(name, expr) =>
        val (prelude, normalized) = normalizeExpr(expr)
        prelude :+ p.Stmt.Mut(name, normalized)
      case p.Stmt.Return(value) =>
        val (prelude, normalized) = normalizeExpr(value)
        prelude :+ p.Stmt.Return(normalized)
      case p.Stmt.While(cond, body) => List(p.Stmt.While(cond, rewrite(body)))
      case p.Stmt.ForRange(induction, lbIncl, ubExcl, step, body) =>
        List(p.Stmt.ForRange(induction, lbIncl, ubExcl, step, rewrite(body)))
      case p.Stmt.Cond(cond, trueBr, falseBr) => List(p.Stmt.Cond(cond, rewrite(trueBr), rewrite(falseBr)))
      case p.Stmt.Annotated(inner, pos, comment) =>
        val statements = rewrite(List(inner))
        statements.lastOption.fold(Nil)(last => statements.dropRight(1) :+ p.Stmt.Annotated(last, pos, comment))
      case p.Stmt.Try(body, handlers, fin) =>
        List(
          p.Stmt
            .Try(body = rewrite(body), handlers = handlers.map(h => h.copy(body = rewrite(h.body))), fin = rewrite(fin))
        )
      case p.Stmt.Raise(value, exceptionKind, cleanup) =>
        List(p.Stmt.Raise(value, exceptionKind, rewrite(cleanup)))
      case statement => List(statement)
    }

    inferred.copy(body = rewrite(inferred.body))
  }

  private def callsites(fn: p.Function): List[Callsite] =
    fn.collectWhere[p.Expr] {
      case p.Expr.Invoke(p.Type.FnRef(name), tpeArgs, receiver, args, rtn) =>
        Callsite(
          name,
          appliedTypeArgs(receiver, tpeArgs),
          receiver.map(_.tpe),
          args.map(_.tpe),
          Some(rtn)
        ) :: Nil
      case p.Expr.Invoke(_, _, _, _, _) => Nil
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            Callsite(name, launch.tpeArgs, None, launch.args.map(_.tpe), None) :: Nil
          case _ => Nil
        }
    }.flatten

  private def insertSegment(sym: p.Sym, segment: String): p.Sym = sym.fqn match {
    case xs :+ x => p.Sym(xs :+ segment :+ x)
    case xs      => p.Sym(segment :: xs)
  }

  def monomorphicName(calleeName: p.Sym, tpeArgs: List[p.Type]): p.Sym = {
    if (tpeArgs.isEmpty) return calleeName
    insertSegment(calleeName, tpeArgs.map(_.monomorphicName).mkString("_"))
  }

  def monomorphicName(ivk: p.Expr.Invoke): p.Sym =
    monomorphicName(ivk.calleeName, appliedTypeArgs(ivk.receiver, ivk.tpeArgs))

  private def monomorphicName(
      fn: p.Function,
      tpeArgs: List[p.Type],
      callableNames: List[p.Sym],
      overloads: Map[p.Sym, List[p.Function]]
  ): p.Sym = {
    val typedBase          = monomorphicName(fn.name, tpeArgs)
    val tpeArgSet          = tpeArgs.toSet
    val uncoveredCallables = callableNames.filterNot(name => tpeArgSet(p.Type.FnRef(name)))
    val base =
      if (uncoveredCallables.isEmpty) typedBase
      else {
        // Kernel function pointers are ordinary value parameters in CUB/rocPRIM dispatch helpers, so their identity
        // is not necessarily present in the helper's template arguments.  Keep callable-specialised names compact:
        // vendor kernel symbols can be thousands of characters long after template instantiation.
        val identity = uncoveredCallables.map(_.fqn.mkString(".")).mkString("\u0000")
        var hash     = -3750763034362895579L // FNV-1a offset basis as a signed Long
        identity.foreach { ch =>
          hash ^= ch.toLong
          hash *= 1099511628211L
        }
        val token = s"callable_${java.lang.Long.toUnsignedString(hash, 16)}"
        insertSegment(typedBase, token)
      }
    val siblings = overloads.getOrElse(fn.name, Nil).sortBy(_.signatureKey)
    if (siblings.size <= 1) base
    else {
      val ordinal = siblings.indexWhere(_.signatureKey == fn.signatureKey)
      insertSegment(base, s"overload${ordinal.max(0)}")
    }
  }

  private val ExecBinderPrefix = "#specialisation_exec#"

  private def bindClosureCallees(fn: p.Function, overloads: Map[p.Sym, List[p.Function]]): p.Function = {
    lazy val operators = overloads.values.flatten.toList
      .flatMap { candidate =>
        candidate.receiver.map(_.named.tpe).collect {
          case p.Type.Ptr(closure: p.Type.Struct, _) => closure.name -> candidate
          case closure: p.Type.Struct                => closure.name -> candidate
        }
      }
      .groupMap(_._1)(_._2)
    fn.modifyAll[p.Expr] {
      case invoke @ p.Expr.Invoke(callee = closure: p.Type.Struct, args = args) =>
        val arity = args.size
        operators.getOrElse(closure.name, Nil).filter(_.args.size == arity).distinct match {
          case operator :: Nil => invoke.copy(callee = p.Type.FnRef(operator.name))
          case _               => invoke
        }
      case invoke @ p.Expr.Invoke(p.Type.FnRef(_), _, Some(receiver), _, _) =>
        receiver.tpe match {
          case p.Type.FnRef(_) | p.Type.Ptr(p.Type.FnRef(_), _) => invoke.copy(receiver = None)
          case _                                                => invoke
        }
      case expression => expression
    }
  }

  private def instantiate(
      fn: p.Function,
      tpeArgs: List[p.Type],
      newName: p.Sym,
      overloads: Map[p.Sym, List[p.Function]]
  ): Option[p.Function] =
    if (fn.tpeVars.isEmpty) Option.when(tpeArgs.isEmpty)(fn.copy(decl = fn.decl.copy(name = newName)))
    else
      Option.when(fn.tpeVars.size == tpeArgs.size) {
        val tpeLut = fn.tpeVars.map(_.name).zip(tpeArgs).toMap
        val substituted = fn
          .copy(decl = fn.decl.copy(name = newName, tpeVars = Nil))
          .modifyAll[p.Type](Binders.protectExec(ExecBinderPrefix))
          .modifyAll[p.Type] {
            case v @ p.Type.Var(name, _) => tpeLut.getOrElse(name, v)
            case x                       => x
          }
          .modifyAll[p.Type](Binders.unprotectExec(ExecBinderPrefix))
        bindClosureCallees(substituted, overloads)
      }

  private def erasedCallableBindings(callsite: Callsite, fn: p.Function): List[(p.Named, p.Type.FnRef)] =
    callsiteParams(fn, callsite.remote)
      .zip(callsite.args)
      .flatMap {
        case (formal, callable: p.Type.FnRef) =>
          formal.named.tpe match {
            case p.Type.Ptr(p.Type.Nothing, _) => Some(formal.named -> callable)
            case _                             => None
          }
        case _ => None
      }

  private def matches(callsite: Callsite, fn: p.Function, allowScalarConversion: Boolean): Boolean =
    if (callsite.remote) {
      val params = launchParams(fn).map(_.named.tpe)
      fn.affinity == p.Function.Affinity.Offload && fn.receiver.isEmpty && fn.rtn == p.Type.Unit0 &&
      params.size == callsite.args.size &&
      params
        .zip(callsite.args)
        .forall((expected, actual) => compatibleArgumentType(expected, actual, allowScalarConversion))
    } else {
      val params = fn.args.map(_.named.tpe)
      fn.receiver.size == callsite.receiver.size &&
      fn.receiver.map(_.named.tpe).zip(callsite.receiver).forall(sameType) &&
      params.size == callsite.args.size &&
      params
        .zip(callsite.args)
        .forall((expected, actual) => compatibleArgumentType(expected, actual, allowScalarConversion = false)) &&
      callsite.rtn.exists(sameType(_, fn.rtn))
    }

  private def receiverTypeArgs(receiver: Option[p.Type]): List[p.Type] = receiver.toList.flatMap {
    case p.Type.Struct(_, args)                => args
    case p.Type.Ptr(p.Type.Struct(_, args), _) => args
    case _                                     => Nil
  }

  private def typeArgsFor(
      callsite: Callsite,
      fn: p.Function,
      allowScalarConversion: Boolean
  ): Option[List[p.Type]] = {
    val supplied     = callsite.tpeArgs
    val receiverArgs = receiverTypeArgs(callsite.receiver)
    val inferred     = receiverArgs ++ supplied
    if (supplied.size == fn.tpeVars.size) Some(supplied)
    else if (inferred.size == fn.tpeVars.size) Some(inferred)
    else if (inferred.size < fn.tpeVars.size) {
      val variables = fn.tpeVars.map(_.name).toSet
      val initial   = fn.tpeVars.iterator.map(_.name).zip(inferred).toMap
      val receiverPatterns = fn.receiver.map(_.named.tpe).zip(callsite.receiver).toList.map { case (expected, actual) =>
        (expected, actual, false)
      }
      val argumentPatterns =
        callsiteParams(fn, callsite.remote).map(_.named.tpe).zip(callsite.args).map { case (expected, actual) =>
          (expected, actual, true)
        }
      val returnPatterns = callsite.rtn.map(fn.rtn -> _).toList.map { case (expected, actual) =>
        (expected, actual, false)
      }
      val patterns = receiverPatterns ++ argumentPatterns ++ returnPatterns
      patterns
        .foldLeft(Option(initial)) { case (acc, (pattern, actual, isArgument)) =>
          if (!containsFreeVariable(pattern, variables))
            acc.filter(_ =>
              if (isArgument) compatibleArgumentType(pattern, actual, callsite.remote && allowScalarConversion)
              else sameType(pattern, actual)
            )
          else acc.flatMap(bind(pattern, actual, _, variables))
        }
        .flatMap(bindings =>
          Option.when(fn.tpeVars.forall(variable => bindings.contains(variable.name)))(
            fn.tpeVars.map(variable => bindings(variable.name))
          )
        )
    } else None
  }

  private def candidates(
      callsite: Callsite,
      overloads: Map[p.Sym, List[p.Function]]
  ): List[Candidate] = {
    val functions = overloads.getOrElse(callsite.calleeName, Nil)
    def select(allowScalarConversion: Boolean): List[Candidate] = functions
      .flatMap { fn =>
        val callables = erasedCallableBindings(callsite, fn)
        typeArgsFor(callsite, fn, allowScalarConversion).flatMap { args =>
          val identity =
            (fn.signatureKey :: args.map(_.canonicalName)) ::: callables.map(_._2.name.fqcn)
          Option
            .when(fn.tpeVars.nonEmpty || callables.nonEmpty)(())
            .flatMap(_ =>
              instantiate(fn, args, monomorphicName(fn, args, callables.map(_._2.name), overloads), overloads)
            )
            .map(materialiseErasedCallables(callsite, _, overloads))
            .map(Candidate(args, fn.signatureKey, identity.mkString("\u0000"), _))
        }
      }
      .filter(candidate => matches(callsite, candidate.function, allowScalarConversion))
    val exact = select(allowScalarConversion = false)
    if (!callsite.remote || exact.nonEmpty) exact else select(allowScalarConversion = true)
  }

  private def materialiseErasedCallables(
      callsite: Callsite,
      fn: p.Function,
      overloads: Map[p.Sym, List[p.Function]]
  ): p.Function = {
    val replacements = erasedCallableBindings(callsite, fn).toMap
    if (replacements.isEmpty) fn
    else {
      val materialised = fn.modifyAll[p.Term] {
        case p.Term.Select(root, Nil, _) if replacements.contains(root) => p.Term.Poison(replacements(root))
        case term                                                       => term
      }
      val callableAliases = materialised
        .collectWhere[p.Stmt] { case p.Stmt.Var(name, Some(p.Expr.Alias(value)), false) =>
          name -> value
        }
        .toMap
      def resolveCallableAlias(term: p.Term, visited: Set[p.Named] = Set.empty): p.Term = term match {
        case select @ p.Term.Select(root, Nil, _) if callableAliases.contains(root) && !visited(root) =>
          resolveCallableAlias(callableAliases(root), visited + root) match {
            case callable @ p.Term.Poison(p.Type.FnRef(_)) => callable
            case _                                         => select
          }
        case other => other
      }
      val resolved = materialised.modifyAll[p.Expr] {
        case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
          p.Expr.SpecOp(launch.copy(kernel = resolveCallableAlias(launch.kernel)))
        case expr => expr
      }
      val callsiteArity = callsite.tpeArgs.size
      resolved.modifyAll[p.Expr] {
        case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.isEmpty =>
          launch.kernel.tpe match {
            case p.Type.FnRef(name)
                if replacements.values.exists(_.name == name) &&
                  callsite.tpeArgs.nonEmpty &&
                  overloads.get(name).fold(true)(_.exists(_.tpeVars.size == callsiteArity)) =>
              p.Expr.SpecOp(launch.copy(tpeArgs = callsite.tpeArgs))
            case _ => p.Expr.SpecOp(launch)
          }
        case expr => expr
      }
    }
  }

  private def typeComplexity(tpe: p.Type): Int = tpe match {
    case p.Type.Struct(_, args)    => 1 + args.map(typeComplexity).sum
    case p.Type.Ptr(comp, _)       => 1 + typeComplexity(comp)
    case p.Type.Arr(comp, _, _)    => 1 + typeComplexity(comp)
    case p.Type.Exec(_, args, rtn) => 1 + args.map(typeComplexity).sum + typeComplexity(rtn)
    case _                         => 1
  }

  private case class SpecialisationState(
      functions: Map[String, p.Function],
      visited: Set[String]
  )

  private def recursiveSpecialise(
      fnOverloads: Map[p.Sym, List[p.Function]],
      entry: p.Function,
      state: SpecialisationState,
      ancestry: Map[String, List[List[p.Type]]] = Map.empty
  ): SpecialisationState = {
    val prepared = inferLaunchArguments(entry, fnOverloads)
    val entryKey = prepared.signatureKey
    if (state.visited(entryKey)) state
    else
      callsites(prepared).distinct
        .foldLeft(state.copy(visited = state.visited + entryKey)) { case (callsiteState, callsite) =>
          candidates(callsite, fnOverloads).foldLeft(callsiteState) { case (overloadState, candidate) =>
            val specialisedFnImpl = inferLaunchArguments(candidate.function, fnOverloads)
            val key               = candidate.identityKey
            if (overloadState.functions.contains(key)) overloadState
            else {
              val history = ancestry.getOrElse(candidate.templateKey, Nil)
              history.takeRight(2) match {
                case previousPrevious :: previous :: Nil
                    if candidate.tpeArgs.map(typeComplexity).sum > previous.map(typeComplexity).sum &&
                      previous.map(typeComplexity).sum > previousPrevious.map(typeComplexity).sum =>
                  throw IllegalStateException(
                    s"Specialisation detected expanding polymorphic recursion in ${callsite.calleeName.repr}: " +
                      s"${previousPrevious.map(_.repr).mkString("[", ", ", "]")} -> " +
                      s"${previous.map(_.repr).mkString("[", ", ", "]")} -> " +
                      candidate.tpeArgs.map(_.repr).mkString("[", ", ", "]")
                  )
                case _ => ()
              }
              recursiveSpecialise(
                fnOverloads,
                specialisedFnImpl,
                overloadState.copy(functions = overloadState.functions + (key -> specialisedFnImpl)),
                ancestry.updated(candidate.templateKey, history :+ candidate.tpeArgs)
              )
            }
          }
        }
  }

  override def apply(program: p.Program, log: Log): p.Program = {

    val fnOverloads = program.functions.distinct.groupBy(_.name)

    if (log.enabled) {
      val allCallsites =
        (program.entry.toList ::: program.functions)
          .map(inferLaunchArguments(_, fnOverloads))
          .flatMap(callsites)
          .distinct

      log.info("functions", program.functions.map(_.signatureKey)*)
      log.info(
        "callsites",
        allCallsites.map(x =>
          s"${x.calleeName.repr}[${x.tpeArgs.map(_.repr).mkString(", ")}]" +
            x.args.map(_.repr).mkString("(", ", ", ")")
        )*
      )
    }

    val roots = program.entry.toList ::: program.functions.filter(_.tpeVars.isEmpty)
    val specialisationsByIdentity = roots
      .foldLeft(SpecialisationState(Map.empty, Set.empty)) { case (state, root) =>
        recursiveSpecialise(fnOverloads, root, state)
      }
      .functions
    specialisationsByIdentity.toList.groupBy(_._2.signatureKey).foreach {
      case (signature, identities) if identities.map(_._1).distinct.size > 1 =>
        throw IllegalStateException(
          s"Specialisation generated colliding symbol `$signature` for distinct callable identities"
        )
      case _ =>
    }
    val specialisations = specialisationsByIdentity.values

    if (log.enabled) log.info("Specialisations", specialisations.map(_.signatureKey).toList.sorted*)
    def doReplace(f: p.Function) = inferLaunchArguments(f, fnOverloads).modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke =>
        val callsite = Callsite(
          ivk.calleeName,
          ivk.tpeArgs,
          ivk.receiver.map(_.tpe),
          ivk.args.map(_.tpe),
          Some(ivk.rtn)
        )
        candidates(callsite, fnOverloads).map(_.function.name).distinct match {
          case name :: Nil => ivk.copy(callee = p.Type.FnRef(name), tpeArgs = Nil)
          case _           => ivk
        }
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.nonEmpty =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            val callsite = Callsite(name, launch.tpeArgs, None, launch.args.map(_.tpe), None)
            candidates(callsite, fnOverloads).map(_.function).distinct match {
              case candidate :: Nil =>
                val kernel = launch.kernel.modifyAll[p.Type] {
                  case p.Type.FnRef(`name`) => p.Type.FnRef(candidate.name)
                  case x                    => x
                }
                val parameters = launchParams(candidate).map(_.named.tpe)
                val arguments  = parameters.zip(launch.args).map(normalizeLaunchArgument)
                p.Expr.SpecOp(launch.copy(kernel = kernel, tpeArgs = Nil, args = arguments))
              case _ => p.Expr.SpecOp(launch)
            }
          case _ => p.Expr.SpecOp(launch)
        }
      case x => x
    }

    program.copy(
      entry = program.entry.map(doReplace),
      functions = (program.functions.filter(_.tpeVars.isEmpty) ++ specialisations).map(doReplace(_))
    )

  }

}
