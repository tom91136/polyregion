package polyregion.ast.pass

import polyregion.ast.Traversal.*
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

  private def sameType(left: p.Type, right: p.Type): Boolean = {
    def loop(
        x: p.Type,
        y: p.Type,
        xBound: Map[String, Int],
        yBound: Map[String, Int],
        depth: Int
    ): Boolean = (x, y) match {
      case (p.Type.Nothing, _) | (_, p.Type.Nothing) => true
      case (p.Type.Var(xName), p.Type.Var(yName)) =>
        (xBound.get(xName), yBound.get(yName)) match {
          case (Some(xId), Some(yId)) => xId == yId
          case (None, None)           => xName == yName
          case _                      => false
        }
      case (p.Type.Struct(xName, xArgs), p.Type.Struct(yName, yArgs)) =>
        xName == yName && xArgs.size == yArgs.size &&
        xArgs.zip(yArgs).forall((a, b) => loop(a, b, xBound, yBound, depth))
      case (p.Type.Ptr(xComp, xSpace), p.Type.Ptr(yComp, ySpace)) =>
        xSpace == ySpace && loop(xComp, yComp, xBound, yBound, depth)
      case (p.Type.Arr(xComp, xLength, xSpace), p.Type.Arr(yComp, yLength, ySpace)) =>
        xLength == yLength && xSpace == ySpace && loop(xComp, yComp, xBound, yBound, depth)
      case (p.Type.Exec(xVars, xArgs, xRtn), p.Type.Exec(yVars, yArgs, yRtn)) =>
        val ids       = xVars.indices.map(depth + _)
        val nestedX   = xBound ++ xVars.zip(ids)
        val nestedY   = yBound ++ yVars.zip(ids)
        val nestedEnd = depth + xVars.size
        xVars.size == yVars.size && xArgs.size == yArgs.size &&
        xArgs.zip(yArgs).forall((a, b) => loop(a, b, nestedX, nestedY, nestedEnd)) &&
        loop(xRtn, yRtn, nestedX, nestedY, nestedEnd)
      case _ => x == y
    }
    loop(left, right, Map.empty, Map.empty, 0)
  }

  private case class Callsite(
      calleeName: p.Sym,
      tpeArgs: List[p.Type],
      receiver: Option[p.Type],
      args: List[p.Type],
      rtn: Option[p.Type],
      remote: Boolean
  )

  private case class Candidate(tpeArgs: List[p.Type], templateKey: String, function: p.Function)

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
      case p.Type.Var(name)       => variables(name) && !bound(name)
      case p.Type.Struct(_, args) => args.exists(containsFreeVariable(_, variables, bound))
      case p.Type.Ptr(comp, _)    => containsFreeVariable(comp, variables, bound)
      case p.Type.Arr(comp, _, _) => containsFreeVariable(comp, variables, bound)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = bound ++ vars
        args.exists(containsFreeVariable(_, variables, nested)) || containsFreeVariable(rtn, variables, nested)
      case _ => false
    }

  private def bind(
      pattern: p.Type,
      actual: p.Type,
      bindings: Map[String, p.Type],
      variables: Set[String]
  ): Option[Map[String, p.Type]] =
    pattern match {
      case p.Type.Var(name) if variables(name) =>
        bindings.get(name) match {
          case Some(bound) if !sameType(bound, actual) => None
          case Some(_)                                 => Some(bindings)
          case None                                    => Some(bindings.updated(name, actual))
        }
      case p.Type.Struct(name, args) =>
        actual match {
          case p.Type.Struct(`name`, actualArgs) if args.size == actualArgs.size =>
            args.zip(actualArgs).foldLeft(Option(bindings)) { case (acc, (expected, found)) =>
              acc.flatMap(bind(expected, found, _, variables))
            }
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
            def align(t: p.Type, env: Map[String, String]): p.Type = t match {
              case p.Type.Var(name)                => p.Type.Var(env.getOrElse(name, name))
              case p.Type.Struct(name, args)       => p.Type.Struct(name, args.map(align(_, env)))
              case p.Type.Ptr(comp, space)         => p.Type.Ptr(align(comp, env), space)
              case p.Type.Arr(comp, length, space) => p.Type.Arr(align(comp, env), length, space)
              case p.Type.Exec(vars, args, rtn) =>
                val nested = env -- vars
                p.Type.Exec(vars, args.map(align(_, nested)), align(rtn, nested))
              case other => other
            }
            val env             = actualVars.zip(patternVars).toMap
            val alignedArgs     = actualArgs.map(align(_, env))
            val alignedRtn      = align(actualRtn, env)
            val nestedVariables = variables -- patternVars
            args
              .zip(alignedArgs)
              .foldLeft(Option(bindings)) { case (acc, (expected, found)) =>
                acc.flatMap(bind(expected, found, _, nestedVariables))
              }
              .flatMap(bind(rtn, alignedRtn, _, nestedVariables))
          case _ => None
        }
      case _ => Option.when(sameType(pattern, actual))(bindings)
    }

  private def launchParams(kernel: p.Function): List[p.Arg] =
    (kernel.moduleCaptures ::: kernel.termCaptures ::: kernel.args)
      .filterNot(arg => arg.named.tpe == p.Type.Unit0 || arg.named.tpe == p.Type.Nothing)

  private def inferLaunchArguments(kernel: p.Function, args: List[p.Type]): Option[List[p.Type]] = {
    val params = launchParams(kernel)
    if (
      params.size != args.size || kernel.tpeVars.isEmpty || kernel.affinity != p.Function.Affinity.Offload ||
      kernel.receiver.nonEmpty || kernel.rtn != p.Type.Unit0
    ) None
    else {
      val variables = kernel.tpeVars.toSet
      params
        .zip(args)
        .foldLeft(Option(Map.empty[String, p.Type])) { case (acc, (param, arg)) =>
          if (!containsFreeVariable(param.named.tpe, variables)) acc.filter(_ => sameType(param.named.tpe, arg))
          else acc.flatMap(bind(param.named.tpe, arg, _, variables))
        }
        .flatMap(bindings => Option.when(kernel.tpeVars.forall(bindings.contains))(kernel.tpeVars.map(bindings)))
    }
  }

  private def inferLaunchArguments(fn: p.Function, overloads: Map[p.Sym, List[p.Function]]): p.Function =
    fn.modifyAll[p.Expr] {
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.isEmpty =>
        val inferred = launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            overloads
              .getOrElse(name, Nil)
              .flatMap(kernel => inferLaunchArguments(kernel, launch.args.map(_.tpe)))
              .distinct match {
              case arguments :: Nil => Some(arguments)
              case _                => None
            }
          case _ => None
        }
        p.Expr.SpecOp(inferred.fold(launch)(arguments => launch.copy(tpeArgs = arguments)))
      case expr => expr
    }

  private def callsites(fn: p.Function): List[Callsite] =
    fn.collectWhere[p.Expr] {
      case p.Expr.Invoke(p.Type.FnRef(name), tpeArgs, receiver, args, rtn) =>
        Callsite(
          name,
          appliedTypeArgs(receiver, tpeArgs),
          receiver.map(_.tpe),
          args.map(_.tpe),
          Some(rtn),
          false
        ) :: Nil
      case _: p.Expr.Invoke => Nil
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            Callsite(name, launch.tpeArgs, None, launch.args.map(_.tpe), None, true) :: Nil
          case _ => Nil
        }
    }.flatten

  def monomorphicName(calleeName: p.Sym, tpeArgs: List[p.Type]): p.Sym = {
    val monomorphicToken = tpeArgs.map(_.monomorphicName).mkString("_")
    calleeName.fqn match {
      case xs :+ x => p.Sym(xs :+ monomorphicToken :+ x)
      case xs      => p.Sym(monomorphicToken :: xs)
    }
  }

  def monomorphicName(ivk: p.Expr.Invoke): p.Sym =
    monomorphicName(ivk.calleeName, appliedTypeArgs(ivk.receiver, ivk.tpeArgs))

  private def monomorphicName(
      fn: p.Function,
      tpeArgs: List[p.Type],
      overloads: Map[p.Sym, List[p.Function]]
  ): p.Sym = {
    val base     = monomorphicName(fn.name, tpeArgs)
    val siblings = overloads.getOrElse(fn.name, Nil).sortBy(_.signatureRepr)
    if (siblings.size <= 1) base
    else {
      val ordinal = siblings.indexWhere(_.signatureRepr == fn.signatureRepr)
      base.fqn match {
        case xs :+ x => p.Sym(xs :+ s"overload${ordinal.max(0)}" :+ x)
        case xs      => p.Sym(s"overload${ordinal.max(0)}" :: xs)
      }
    }
  }

  private val ExecBinderPrefix = "#specialisation_exec#"

  private def protectExecBinders(tpe: p.Type): p.Type = {
    def rename(t: p.Type, env: Map[String, String]): p.Type = t match {
      case p.Type.Var(name)                => p.Type.Var(env.getOrElse(name, name))
      case p.Type.Struct(name, args)       => p.Type.Struct(name, args.map(rename(_, env)))
      case p.Type.Ptr(comp, space)         => p.Type.Ptr(rename(comp, env), space)
      case p.Type.Arr(comp, length, space) => p.Type.Arr(rename(comp, env), length, space)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = env -- vars
        p.Type.Exec(vars, args.map(rename(_, nested)), rename(rtn, nested))
      case other => other
    }
    tpe match {
      case p.Type.Exec(vars, args, rtn) if !vars.forall(_.startsWith(ExecBinderPrefix)) =>
        val renamed = vars.map(ExecBinderPrefix + _)
        val env     = vars.zip(renamed).toMap
        p.Type.Exec(renamed, args.map(rename(_, env)), rename(rtn, env))
      case other => other
    }
  }

  private def unprotectExecBinders(tpe: p.Type): p.Type = tpe match {
    case p.Type.Var(name) if name.startsWith(ExecBinderPrefix) => p.Type.Var(name.stripPrefix(ExecBinderPrefix))
    case p.Type.Exec(vars, args, rtn) =>
      p.Type.Exec(vars.map(_.stripPrefix(ExecBinderPrefix)), args, rtn)
    case other => other
  }

  private def instantiate(fn: p.Function, tpeArgs: List[p.Type], newName: p.Sym): Option[p.Function] =
    Option.when(fn.tpeVars.nonEmpty && fn.tpeVars.size == tpeArgs.size) {
      val tpeLut = fn.tpeVars.zip(tpeArgs).toMap
      fn
        .copy(decl = fn.decl.copy(name = newName, tpeVars = Nil))
        .modifyAll[p.Type](protectExecBinders)
        .modifyAll[p.Type] {
          case v @ p.Type.Var(name) => tpeLut.getOrElse(name, v)
          case x                    => x
        }
        .modifyAll[p.Type](unprotectExecBinders)
    }

  private def matches(callsite: Callsite, fn: p.Function): Boolean =
    if (callsite.remote) {
      val params = launchParams(fn).map(_.named.tpe)
      fn.affinity == p.Function.Affinity.Offload && fn.receiver.isEmpty && fn.rtn == p.Type.Unit0 &&
      params.size == callsite.args.size && params.zip(callsite.args).forall(sameType)
    } else {
      val params = fn.args.map(_.named.tpe)
      fn.receiver.size == callsite.receiver.size &&
      fn.receiver.map(_.named.tpe).zip(callsite.receiver).forall(sameType) &&
      params.size == callsite.args.size && params.zip(callsite.args).forall(sameType) &&
      callsite.rtn.exists(sameType(_, fn.rtn))
    }

  private def receiverTypeArgs(receiver: Option[p.Type]): List[p.Type] = receiver.toList.flatMap {
    case p.Type.Struct(_, args)                => args
    case p.Type.Ptr(p.Type.Struct(_, args), _) => args
    case _                                     => Nil
  }

  private def typeArgsFor(callsite: Callsite, fn: p.Function): Option[List[p.Type]] = {
    val supplied = callsite.tpeArgs
    val inferred = receiverTypeArgs(callsite.receiver) ++ supplied
    if (supplied.size == fn.tpeVars.size) Some(supplied)
    else Option.when(inferred.size == fn.tpeVars.size)(inferred)
  }

  private def candidates(
      callsite: Callsite,
      overloads: Map[p.Sym, List[p.Function]]
  ): List[Candidate] =
    overloads
      .getOrElse(callsite.calleeName, Nil)
      .flatMap(fn =>
        typeArgsFor(callsite, fn).flatMap(args =>
          instantiate(fn, args, monomorphicName(fn, args, overloads))
            .map(Candidate(args, fn.signatureRepr, _))
        )
      )
      .filter(candidate => matches(callsite, candidate.function))

  private def typeComplexity(tpe: p.Type): Int = tpe match {
    case p.Type.Struct(_, args)    => 1 + args.map(typeComplexity).sum
    case p.Type.Ptr(comp, _)       => 1 + typeComplexity(comp)
    case p.Type.Arr(comp, _, _)    => 1 + typeComplexity(comp)
    case p.Type.Exec(_, args, rtn) => 1 + args.map(typeComplexity).sum + typeComplexity(rtn)
    case _                         => 1
  }

  def recursiveSpecialise(
      fnOverloads: Map[p.Sym, List[p.Function]],
      entry: p.Function,
      done: Map[String, p.Function] = Map.empty,
      ancestry: Map[String, List[List[p.Type]]] = Map.empty
  ): Map[String, p.Function] = {
    val prepared = inferLaunchArguments(entry, fnOverloads)
    callsites(prepared).distinct
      .foldLeft(done) { case (acc, callsite) =>
        candidates(callsite, fnOverloads).foldLeft(acc) { case (overloadAcc, candidate) =>
          val specialisedFnImpl = inferLaunchArguments(candidate.function, fnOverloads)
          val key               = specialisedFnImpl.signatureRepr
          if (overloadAcc.contains(key)) overloadAcc
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
              overloadAcc + (key -> specialisedFnImpl),
              ancestry.updated(candidate.templateKey, history :+ candidate.tpeArgs)
            )
          }
        }
      }
  }

  override def apply(program: p.Program, log: Log): p.Program = {

    val fnOverloads = program.functions.distinct.groupBy(_.name)

    val allCallsites =
      (program.entry :: program.functions).map(inferLaunchArguments(_, fnOverloads)).flatMap(callsites).distinct

    log.info("functions", fnOverloads.keys.toSeq.map(_.repr)*)
    log.info(
      "callsites",
      allCallsites.map(x => s"${x.calleeName.repr}[${x.tpeArgs.map(_.repr).mkString(", ")}]")*
    )

    val specialisations =
      (program.entry :: program.functions.filter(_.tpeVars.isEmpty)).foldLeft(Map.empty[String, p.Function]) {
        case (done, root) => recursiveSpecialise(fnOverloads, root, done)
      }

    log.info("Specialisations", specialisations.values.map(_.signatureRepr).toList.sorted*)

    def doReplace(f: p.Function) = inferLaunchArguments(f, fnOverloads).modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke =>
        val callsite = Callsite(
          ivk.calleeName,
          ivk.tpeArgs,
          ivk.receiver.map(_.tpe),
          ivk.args.map(_.tpe),
          Some(ivk.rtn),
          remote = false
        )
        candidates(callsite, fnOverloads).map(_.function.name).distinct match {
          case name :: Nil => ivk.copy(callee = p.Type.FnRef(name), tpeArgs = Nil)
          case _           => ivk
        }
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.nonEmpty =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            val callsite = Callsite(name, launch.tpeArgs, None, launch.args.map(_.tpe), None, remote = true)
            candidates(callsite, fnOverloads).map(_.function.name).distinct match {
              case newName :: Nil =>
                val kernel = launch.kernel.modifyAll[p.Type] {
                  case p.Type.FnRef(`name`) => p.Type.FnRef(newName)
                  case x                    => x
                }
                p.Expr.SpecOp(launch.copy(kernel = kernel, tpeArgs = Nil))
              case _ => p.Expr.SpecOp(launch)
            }
          case _ => p.Expr.SpecOp(launch)
        }
      case x => x
    }

    program.copy(
      entry = doReplace(program.entry),
      functions = (program.functions.filter(_.tpeVars.isEmpty) ++ specialisations.values).map(doReplace(_))
    )

  }

}
