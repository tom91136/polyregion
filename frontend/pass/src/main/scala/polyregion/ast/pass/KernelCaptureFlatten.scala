package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// makes global pointer fields embedded in an offload capture explicit kernel arguments, rewriting both the
// kernel body and every direct or remote launch site to keep the constrained buffer ABI in sync. scalar
// capture fields stay in the aggregate; an escaped by-value subobject is copied locally and its pointer
// fields patched from the explicit arguments before use
// examples:
//   kernel(cap){ cap.data[i] }          ->  kernel(cap, #ptr){ #ptr[i] }
//   launch(kernel, cap)                 ->  launch(kernel, cap, cap.data)
//   kernel(cap){ copy = cap.view }      ->  copy = cap.view; copy.data = #ptr  (then use copy)
// edge cases:
//   scalar-only capture                ->  unchanged
//   module/term captures               ->  retain their flattened call/launch order ahead of ordinary args
//   overloaded remote kernel           ->  selected by substituted effective parameter types
//   nested pointer graph / union / slot mutation / unsupported escape  ->  rejected
object KernelCaptureFlatten extends ProgramPass {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private final case class Leaf(path: List[p.PathStep], tpe: p.Type.Ptr)
  private final case class Binding(leaf: Leaf, arg: p.Arg)
  private final case class Repair(path: List[p.PathStep], tpe: p.Type.Struct, local: p.Named)
  private final case class Plan(capture: p.Named, bindings: List[Binding], repairs: List[Repair])

  private def fail(kernel: p.Function, message: String): Nothing =
    throw RuntimeException(s"KernelCaptureFlatten: kernel ${kernel.name.repr}: $message")

  private def pathRepr(path: List[p.PathStep]): String =
    if (path.isEmpty) "<capture>"
    else
      path
        .map {
          case p.PathStep.Field(n)    => n
          case p.PathStep.Deref       => "*"
          case p.PathStep.Index(i)    => s"[$i]"
          case p.PathStep.IndexDyn(_) => "[?]"
        }
        .mkString(".")

  override def apply(program: p.Program, log: Log): p.Program = {
    val defs = program.defs.groupBy(_.name)

    def structDef(kernel: p.Function, tpe: p.Type.Struct): p.StructDef = defs.getOrElse(tpe.name, Nil) match {
      case definition :: Nil => definition
      case Nil               => fail(kernel, s"missing definition for capture member type ${tpe.repr}")
      case _                 => fail(kernel, s"ambiguous definitions for capture member type ${tpe.repr}")
    }

    def subst(tpe: p.Type, env: Map[String, p.Type]): p.Type =
      tpe.modifyAll[p.Type] { case variable @ p.Type.Var(name) => env.getOrElse(name, variable); case other => other }

    def hasPointers(tpe: p.Type, seen: Set[p.Sym]): Boolean = tpe match {
      case _: p.Type.Ptr          => true
      case p.Type.Arr(comp, _, _) => hasPointers(comp, seen)
      case struct: p.Type.Struct if !seen(struct.name) =>
        defs.get(struct.name).flatMap(_.headOption).exists { definition =>
          val env = definition.tpeVars.zip(struct.args).toMap
          definition.members.exists(member => hasPointers(subst(member.tpe, env), seen + struct.name))
        }
      case _ => false
    }

    def leavesOf(kernel: p.Function, root: p.Type.Struct): List[Leaf] = {
      def walk(tpe: p.Type, path: List[p.PathStep], seen: Set[p.Sym]): List[Leaf] = tpe match {
        case ptr @ p.Type.Ptr(comp, p.Type.Space.Global) =>
          if (hasPointers(comp, Set.empty))
            fail(kernel, s"pointer graph at capture field ${pathRepr(path)} (${ptr.repr}) is unsupported")
          List(Leaf(path, ptr))
        case ptr: p.Type.Ptr =>
          fail(kernel, s"non-global capture pointer at field ${pathRepr(path)} (${ptr.repr}) is unsupported")
        case array @ p.Type.Arr(comp, _, _) =>
          if (hasPointers(comp, Set.empty))
            fail(kernel, s"pointer-bearing capture array at field ${pathRepr(path)} (${array.repr}) is unsupported")
          Nil
        case struct: p.Type.Struct =>
          if (seen(struct.name))
            fail(kernel, s"recursive by-value capture type ${struct.repr} at field ${pathRepr(path)}")
          val definition = structDef(kernel, struct)
          if (definition.tpeVars.size != struct.args.size)
            fail(kernel, s"type argument mismatch for capture member type ${struct.repr}")
          val env = definition.tpeVars.zip(struct.args).toMap
          if (
            definition.isUnion && definition.members
              .exists(member => hasPointers(subst(member.tpe, env), Set(struct.name)))
          )
            fail(kernel, s"pointer-bearing capture union ${struct.repr} at field ${pathRepr(path)} is unsupported")
          definition.members.flatMap(member =>
            walk(subst(member.tpe, env), path :+ p.PathStep.Field(member.symbol), seen + struct.name)
          )
        case variable: p.Type.Var =>
          fail(kernel, s"unresolved type variable ${variable.repr} at capture field ${pathRepr(path)}")
        case _ => Nil
      }
      walk(root, Nil, Set.empty)
    }

    val allFunctions = program.entry :: program.functions
    val calls        = allFunctions.flatMap(_.collectWhere[p.Expr] { case call: p.Expr.Invoke => call })
    val launches = allFunctions.flatMap(
      _.collectWhere[p.Expr] { case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) => launch }
    )
    val byName = allFunctions.groupBy(_.name)

    def launchParameters(kernel: p.Function, tpeArgs: List[p.Type]): List[p.Arg] = {
      val env = kernel.tpeVars.zip(tpeArgs).toMap
      (kernel.moduleCaptures ::: kernel.termCaptures ::: kernel.args)
        .map(arg => arg.copy(named = arg.named.copy(tpe = subst(arg.named.tpe, env))))
        .filterNot(arg => arg.named.tpe == p.Type.Unit0 || arg.named.tpe == p.Type.Nothing)
    }

    def launchTarget(launch: p.Spec.RemoteLaunch): Option[p.Function] = launch.kernel.tpe match {
      case p.Type.FnRef(name) =>
        val matches = byName.getOrElse(name, Nil).filter { candidate =>
          val parameters = launchParameters(candidate, launch.tpeArgs)
          candidate.affinity == p.Function.Affinity.Offload &&
          candidate.receiver.isEmpty &&
          candidate.rtn == p.Type.Unit0 &&
          candidate.tpeVars.size == launch.tpeArgs.size &&
          parameters.map(_.named.tpe) == launch.args.map(_.tpe)
        }
        matches match {
          case kernel :: Nil => Some(kernel)
          case Nil           => None
          case _ =>
            throw RuntimeException(s"KernelCaptureFlatten: launch of ${name.repr} matches multiple function bodies")
        }
      case _ => None
    }

    def buildPlan(kernel: p.Function): Option[Plan] = captureRoot(kernel) match {
      case None => None
      case Some((capture, captureStruct)) =>
        val leaves = leavesOf(kernel, captureStruct)
        if (leaves.isEmpty) None
        else {
          val captureSelects = kernel.body.flatMap(_.collectWhere[p.Term] {
            case select @ p.Term.Select(root, _, _) if root == capture => select
          })
          val escaped = captureSelects.filter(select =>
            leaves.exists(leaf => select.steps.length < leaf.path.length && leaf.path.startsWith(select.steps))
          )
          escaped
            .find(_.steps.isEmpty)
            .foreach(_ =>
              fail(kernel, "the entire capture pointer escapes before a pointer leaf; capture identity is unsupported")
            )
          val minimalEscapes = escaped.sortBy(_.steps.length).foldLeft(List.empty[p.Term.Select]) { (roots, select) =>
            if (roots.exists(root => select.steps.startsWith(root.steps))) roots else roots :+ select
          }
          minimalEscapes.foreach { select =>
            if (!select.tpe.isInstanceOf[p.Type.Struct])
              fail(
                kernel,
                s"pointer-bearing capture subobject ${pathRepr(select.steps)} has non-struct type ${select.tpe.repr}"
              )
          }
          kernel.body
            .flatMap(_.collectWhere[p.Expr] {
              case p.Expr.RefTo(p.Term.Select(root, steps, _), None, _, _, _) if root == capture => steps
            })
            .foreach(steps =>
              leaves
                .find(_.path == steps)
                .foreach(leaf =>
                  fail(kernel, s"address-taking of capture pointer slot ${pathRepr(leaf.path)} is unsupported")
                )
            )
          kernel.body
            .flatMap(_.collectWhere[p.Stmt] {
              case p.Stmt.Mut(p.Term.Select(root, steps, _), _) if root == capture => steps
            })
            .foreach(steps =>
              leaves
                .find(_.path == steps)
                .foreach(leaf =>
                  fail(kernel, s"mutation of capture pointer slot ${pathRepr(leaf.path)} is unsupported")
                )
            )
          val used = leaves.filter(leaf =>
            captureSelects.exists(select => select.steps.startsWith(leaf.path)) ||
              minimalEscapes.exists(select => leaf.path.startsWith(select.steps))
          )
          val bindings = used.zipWithIndex.map { case (leaf, index) =>
            Binding(leaf, p.Arg(p.Named(s"#capture_ptr_$index", leaf.tpe)))
          }
          val repairs = minimalEscapes.zipWithIndex.map { case (select, index) =>
            val tpe = select.tpe.asInstanceOf[p.Type.Struct]
            Repair(select.steps, tpe, p.Named(s"#capture_copy_$index", tpe))
          }
          Option.when(bindings.nonEmpty)(Plan(capture, bindings, repairs))
        }
    }

    val directlyCalled = calls.flatMap(_.calleeSym).toSet.flatMap { name =>
      byName.getOrElse(name, Nil) match {
        case kernel :: Nil if kernel.affinity == p.Function.Affinity.Offload => List(kernel)
        case Nil                                                             => Nil
        case _ :: Nil                                                        => Nil
        case _ =>
          throw RuntimeException(s"KernelCaptureFlatten: called kernel ${name.repr} has multiple function bodies")
      }
    }
    val launchTargets = launches.flatMap(launchTarget)
    val plans: Map[p.FunctionDecl, Plan] =
      (directlyCalled ++ launchTargets).flatMap(kernel => buildPlan(kernel).map(kernel.decl -> _)).toMap
    val directPlans: Map[p.Sym, (p.Function, Plan)] =
      directlyCalled.flatMap(kernel => plans.get(kernel.decl).map(plan => kernel.name -> (kernel, plan))).toMap

    def rewriteKernel(kernel: p.Function, plan: Plan): p.Function = {
      def rewriteTerm(term: p.Term): p.Term = term match {
        case p.Term.Select(root, steps, resultTpe) if root == plan.capture =>
          plan.repairs.find(repair => steps.startsWith(repair.path)) match {
            case Some(repair) => p.Term.Select(repair.local, steps.drop(repair.path.length), resultTpe)
            case None =>
              plan.bindings.find(binding => steps.startsWith(binding.leaf.path)) match {
                case Some(binding) => p.Term.Select(binding.arg.named, steps.drop(binding.leaf.path.length), resultTpe)
                case None          => term
              }
          }
        case _ => term
      }
      val rewritten = kernel.body
        .modifyAll[p.Term](rewriteTerm)
        .modifyAll[p.Stmt] {
          case p.Stmt.Mut(select, expr)      => p.Stmt.Mut(rewriteTerm(select).asInstanceOf[p.Term.Select], expr)
          case p.Stmt.Update(select, idx, v) => p.Stmt.Update(rewriteTerm(select).asInstanceOf[p.Term.Select], idx, v)
          case stmt                          => stmt
        }
      val stale = rewritten.flatMap(_.collectWhere[p.Term] {
        case select @ p.Term.Select(root, steps, _)
            if root == plan.capture &&
              (plan.bindings.exists(binding => steps.startsWith(binding.leaf.path)) ||
                plan.repairs.exists(repair => steps.startsWith(repair.path))) =>
          select
      })
      if (stale.nonEmpty) fail(kernel, s"internal rewrite left pointer-bearing capture access ${stale.head.repr}")
      val prelude = plan.repairs.flatMap { repair =>
        val copy = p.Stmt.Var(
          repair.local,
          Some(p.Expr.Alias(p.Term.Select(plan.capture, repair.path, repair.tpe))),
          isMutable = true
        )
        val patches = plan.bindings.collect {
          case binding if binding.leaf.path.startsWith(repair.path) =>
            p.Stmt.Mut(
              p.Term.Select(repair.local, binding.leaf.path.drop(repair.path.length), binding.leaf.tpe),
              p.Expr.Alias(p.Term.Select(binding.arg.named, Nil, binding.leaf.tpe))
            )
        }
        copy :: patches
      }
      kernel.copy(
        decl = kernel.decl.remapArgs(kernel.args ::: plan.bindings.map(_.arg)),
        body = prelude ::: rewritten
      )
    }

    def extractedArgs(
        kernel: p.Function,
        plan: Plan,
        parameters: List[p.Arg],
        arguments: List[p.Term],
        transport: String
    ): List[p.Term] = {
      val expected = parameters.size
      if (arguments.size != expected)
        fail(kernel, s"$transport has ${arguments.size} arguments but kernel has $expected effective parameters")
      val captureIndex = parameters.indexWhere(_.named == plan.capture)
      if (captureIndex < 0)
        fail(kernel, s"capture ${plan.capture.symbol} is absent from the effective kernel parameters")
      val captureArg = arguments(captureIndex) match {
        case select: p.Term.Select => select
        case other                 => fail(kernel, s"$transport capture argument ${other.repr} is not a Select")
      }
      plan.bindings.map(binding =>
        p.Term.Select(captureArg.root, captureArg.steps ::: binding.leaf.path, binding.leaf.tpe): p.Term
      )
    }

    def rewriteCalls(owner: p.Function): p.Function = owner.modifyAll[p.Expr] {
      case call: p.Expr.Invoke if call.calleeSym.exists(directPlans.contains) =>
        val (kernel, plan) = directPlans(call.calleeSym.get)
        val params         = call.receiver.toList ::: call.args
        val parameters     = kernel.receiver.toList ::: kernel.moduleCaptures ::: kernel.termCaptures ::: kernel.args
        call.copy(args = call.args ::: extractedArgs(kernel, plan, parameters, params, "call"))
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
        launchTarget(launch).flatMap(kernel => plans.get(kernel.decl).map(kernel -> _)) match {
          case Some((kernel, plan)) =>
            val parameters = launchParameters(kernel, launch.tpeArgs)
            p.Expr.SpecOp(
              launch.copy(args = launch.args ::: extractedArgs(kernel, plan, parameters, launch.args, "launch"))
            )
          case None => p.Expr.SpecOp(launch)
        }
      case expr => expr
    }

    val entry = rewriteCalls(plans.get(program.entry.decl).fold(program.entry)(rewriteKernel(program.entry, _)))
    val functions = program.functions.map(function =>
      rewriteCalls(plans.get(function.decl).fold(function)(rewriteKernel(function, _)))
    )
    if (plans.nonEmpty)
      log.info(
        s"flattened ${plans.valuesIterator.map(_.bindings.size).sum} capture pointer field(s) across ${plans.size} kernel(s)"
      )
    program.copy(entry = entry, functions = functions)
  }
}
