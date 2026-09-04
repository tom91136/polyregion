package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// inlines every Invoke reachable from entry to a fixed point (the result Program has no functions):
// the callee body is type-substituted, alpha-renamed, its params bound to the call args, and spliced
// in at the callsite scope
// examples:
//   f(x){ ret x+1 }; y=f(2)            ->  y=2+1   // body spliced, x bound to 2, return becomes the value
//   f[T](x:T){ ret x }; y=f[Int](a)    ->  y=a     // T -> Int before splicing
//   f(x){ if c {ret x} else {ret 0} }  ->  var _phi; if c {_phi=x} else {_phi=0}; ..; _phi  (multi-return phi)
//   g(){ ret f(1) }; z=g()             ->  z=1+1   // nested Invoke inlined first, to a fixed point
// edge cases:
//   single Return        -> the return Expr becomes the callsite value; Return stmts stripped
//   multiple Returns     -> a mutable phi var, each Return rebound to a Mut of the phi
//   moduleCaptures       -> NOT alpha-renamed (shared across inlinings); only locals get the per-call id
//   nested Invoke in body -> recursed before splicing so inlining reaches a fixed point
// inlined stmts stay in the original Invoke's enclosing scope - hoisting to the body root would let
// references to loop-/branch-local vars escape their scope
object FnInline extends ProgramPass {

  private type OverloadLut = Map[(p.Sym, Int), List[p.Function]]

  private def sameType(left: p.Type, right: p.Type): Boolean = {
    def loop(
        x: p.Type,
        y: p.Type,
        xBound: Map[String, Int],
        yBound: Map[String, Int],
        depth: Int
    ): Boolean = (x, y) match {
      case (p.Type.Nothing, _) | (_, p.Type.Nothing) => true
      case (p.Type.Var(xName, _), p.Type.Var(yName, _)) =>
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
        val nestedX   = xBound ++ xVars.map(_.name).zip(ids)
        val nestedY   = yBound ++ yVars.map(_.name).zip(ids)
        val nestedEnd = depth + xVars.size
        xVars.size == yVars.size && xArgs.size == yArgs.size &&
        xArgs.zip(yArgs).forall((a, b) => loop(a, b, nestedX, nestedY, nestedEnd)) &&
        loop(xRtn, yRtn, nestedX, nestedY, nestedEnd)
      case _ => x == y
    }
    loop(left, right, Map.empty, Map.empty, 0)
  }

  private def flatParams(f: p.Function): List[p.Type] =
    f.moduleCaptures.map(_.named.tpe) ++ f.termCaptures.map(_.named.tpe) ++ f.args.map(_.named.tpe)

  private def receiverTypeArgs(ivk: p.Expr.Invoke): List[p.Type] = ivk.receiver.toList.flatMap(_.tpe match {
    case p.Type.Struct(_, args)                => args
    case p.Type.Ptr(p.Type.Struct(_, args), _) => args
    case _                                     => Nil
  })

  private def appliedTypeArgs(ivk: p.Expr.Invoke, expected: Int): List[p.Type] = {
    val combined = receiverTypeArgs(ivk) ++ ivk.tpeArgs
    if (ivk.tpeArgs.size == expected) ivk.tpeArgs
    else if (combined.size == expected) combined
    else ivk.tpeArgs
  }

  private val ExecBinderPrefix = "#fn_inline_exec#"

  private def protectExecBinders(tpe: p.Type): p.Type = {
    def rename(t: p.Type, env: Map[String, String]): p.Type = t match {
      case variable @ p.Type.Var(name, _)  => variable.copy(name = env.getOrElse(name, name))
      case p.Type.Struct(name, args)       => p.Type.Struct(name, args.map(rename(_, env)))
      case p.Type.Ptr(comp, space)         => p.Type.Ptr(rename(comp, env), space)
      case p.Type.Arr(comp, length, space) => p.Type.Arr(rename(comp, env), length, space)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = env -- vars.map(_.name)
        p.Type.Exec(vars, args.map(rename(_, nested)), rename(rtn, nested))
      case other => other
    }
    tpe match {
      case p.Type.Exec(vars, args, rtn) if !vars.forall(_.name.startsWith(ExecBinderPrefix)) =>
        val renamed: List[p.Type.Var] =
          vars.map(variable => p.Type.Var(ExecBinderPrefix + variable.name, variable.exactSizeInBytes))
        val env = vars.map(_.name).zip(renamed.map(_.name)).toMap
        p.Type.Exec(renamed, args.map(rename(_, env)), rename(rtn, env))
      case other => other
    }
  }

  private def unprotectExecBinders(tpe: p.Type): p.Type = tpe match {
    case variable @ p.Type.Var(name, _) if name.startsWith(ExecBinderPrefix) =>
      variable.copy(name = name.stripPrefix(ExecBinderPrefix))
    case p.Type.Exec(vars, args, rtn) =>
      p.Type.Exec(vars.map(variable => variable.copy(name = variable.name.stripPrefix(ExecBinderPrefix))), args, rtn)
    case other => other
  }

  private def substTerms(tree: p.Function, table: Map[p.Named, p.Term]): p.Function =
    tree.modifyAll[p.Term] {
      case p.Term.Select(root, steps, tpe) =>
        table.get(root) match {
          case Some(p.Term.Select(rRoot, rSteps, _)) => p.Term.Select(rRoot, rSteps ::: steps, tpe)
          case Some(other) if steps.isEmpty          => other
          case Some(_)                               => p.Term.Select(root, steps, tpe)
          case None                                  => p.Term.Select(root, steps, tpe)
        }
      case x => x
    }

  private def renameAll(f: p.Function, ctr: java.util.concurrent.atomic.AtomicLong): p.Function = {
    val id      = ctr.incrementAndGet()
    val renamed = scala.collection.mutable.HashMap.empty[String, String]
    def semantic(n: p.Named) =
      n.symbol == p.Conventions.ExceptionValue || n.symbol == p.Conventions.ExceptionWhat ||
        n.symbol == p.Conventions.ExceptionCode
    def rename(n: p.Named) =
      if (semantic(n)) n
      else n.copy(symbol = renamed.getOrElseUpdate(n.symbol, s"_inline_${id}_${renamed.size}"))
    val captureNames = f.moduleCaptures.map(_.named).toSet
    val body = f.body
      .modifyAll[p.Term] {
        case s @ p.Term.Select(root, _, _) if captureNames.contains(root) => s
        case s @ p.Term.Select(root, _, _) if semantic(root)              => s
        case p.Term.Select(root, steps, tpe)                              => p.Term.Select(rename(root), steps, tpe)
        case x                                                            => x
      }
      .modifyAll[p.Stmt] {
        case p.Stmt.Var(n, expr, isMutable)         => p.Stmt.Var(rename(n), expr, isMutable)
        case p.Stmt.ForRange(i, lb, ub, step, body) => p.Stmt.ForRange(rename(i), lb, ub, step, body)
        case p.Stmt.Try(b, hs, fin) => p.Stmt.Try(b, hs.map(h => h.copy(binder = h.binder.map(rename))), fin)
        case x                      => x
      }
    p.Function(
      f.decl.copy(
        receiver = f.receiver.map(arg => arg.copy(rename(arg.named))),
        args = f.args.map(arg => arg.copy(rename(arg.named))),
        termCaptures = f.termCaptures.map(arg => arg.copy(rename(arg.named)))
      ),
      body,
      f.visibility,
      f.fpMode,
      f.convention
    )
  }

  private def inlineOne(
      ivk: p.Expr.Invoke,
      f: p.Function,
      ctr: java.util.concurrent.atomic.AtomicLong
  ): (p.Expr, List[p.Stmt], List[p.Arg]) = {

    val concreteTpeArgs = appliedTypeArgs(ivk, f.tpeVars.size)

    val table = f.tpeVars.map(_.name).zip(concreteTpeArgs).toMap

    val typed =
      if (table.isEmpty) f
      else
        f
          .modifyAll[p.Type](protectExecBinders)
          .modifyAll[p.Type](_.mapLeaf {
            case p.Type.Var(name, _) if table.contains(name) => table(name)
            case x                                           => x
          })
          .modifyAll[p.Type](unprotectExecBinders)
    val renamed = renameAll(typed, ctr)

    // ivk.args is the flattened (moduleCaptures ::: termCaptures ::: args) per Compiler.patchIvk.
    val targetNames =
      renamed.receiver.map(_.named).toList ++
        renamed.moduleCaptures.map(_.named) ++
        renamed.termCaptures.map(_.named) ++
        renamed.args.map(_.named)
    val replacements = ivk.receiver.toList ++ ivk.args
    val parameters = targetNames.zip(replacements).map { case (name, value) =>
      val conformed = (name.tpe, value) match {
        // A null pointer is valid in every address space. Calls can acquire a
        // more precise formal space during package specialisation, so carry
        // that space into the inlined binding instead of retaining the generic
        // null type from the original call site.
        case (p.Type.Ptr(comp, space), p.Term.NullPtrConst(_, _, region)) =>
          p.Term.NullPtrConst(comp, space, region)
        case _ => value
      }
      name -> conformed
    }
    val mutableParameters = renamed
      .collectWhere[p.Stmt] { case p.Stmt.Mut(p.Term.Select(root, _, _), _) =>
        root
      }
      .toSet
    val moduleCaptureNames = renamed.moduleCaptures.map(_.named).toSet
    val mutableLocals = parameters.collect {
      case (name, value) if mutableParameters.contains(name) =>
        val local =
          if (moduleCaptureNames(name)) p.Named(s"_inline_${ctr.incrementAndGet()}_${name.symbol}", name.tpe)
          else name
        name -> (local, p.Stmt.Var(local, Some(p.Expr.Alias(value)), isMutable = true))
    }
    val bindings = mutableLocals.map(_._2._2)
    val substTable = parameters.filterNot((name, _) => mutableParameters.contains(name)).toMap ++
      mutableLocals.map { case (original, (local, _)) => original -> p.Term.Select(local, Nil, local.tpe) }
    val substituted = substTerms(renamed, substTable)

    // rebindReturn turns each `return` into a phi store but cannot model early exit; sink the
    // fall-through after a returning branch into the sibling branch so the returns become mutually
    // exclusive (else an unconditional tail return clobbers the phi set in a conditional branch)
    val sunk         = substituted.copy(body = sinkAfterReturn(bindings ::: substituted.body))
    val returnExprs  = sunk.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e }
    val directReturn = sunk.body.collectFirst { case p.Stmt.Return(e) => e }

    returnExprs match {
      case Nil =>
        throw AssertionError(s"no return in function ${f.signature}")
      case expr :: Nil if directReturn.contains(expr) && !returnsNeedUnwind(sunk.body) =>
        val noReturnBody = sunk.body.flatMap(stripReturn)
        (expr, noReturnBody, renamed.moduleCaptures)
      case _ =>
        val phiName                  = p.Named(s"_inline_phi_${ctr.incrementAndGet()}", ivk.tpe)
        val phiSelect: p.Term.Select = p.Term.Select(phiName, Nil, ivk.tpe)
        val phiDecl                  = p.Stmt.Var(phiName, None, isMutable = true)
        if (returnsNeedUnwind(sunk.body)) {
          val flagName                = p.Named(s"_inline_exit_${ctr.incrementAndGet()}", p.Type.Bool1)
          val flagTerm: p.Term.Select = p.Term.Select(flagName, Nil, p.Type.Bool1)
          val flagDecl = p.Stmt.Var(flagName, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
          val unwound  = unwindReturn(sunk.body, phiSelect, flagTerm, inLoop = false, ctr)
          (p.Expr.Alias(phiSelect), phiDecl :: flagDecl :: unwound, renamed.moduleCaptures)
        } else {
          val rebound = sunk.body.map(rebindReturn(phiSelect))
          (p.Expr.Alias(phiSelect), phiDecl :: rebound, renamed.moduleCaptures)
        }
    }
  }

  private def containsReturn(s: p.Stmt): Boolean =
    s.collectFirst_[p.Stmt] { case r: p.Stmt.Return => r }.isDefined

  private def returnsNeedUnwind(stmts: List[p.Stmt]): Boolean = stmts.exists {
    case t: p.Stmt.Try      => t.blocks.exists(_.exists(containsReturn)) || t.blocks.exists(returnsNeedUnwind)
    case p.Stmt.While(_, b) => b.exists(containsReturn) || returnsNeedUnwind(b)
    case p.Stmt.ForRange(_, _, _, _, b) => b.exists(containsReturn) || returnsNeedUnwind(b)
    case p.Stmt.Cond(_, t, f)           => returnsNeedUnwind(t) || returnsNeedUnwind(f)
    case p.Stmt.Annotated(inner, _, _)  => returnsNeedUnwind(List(inner))
    case _                              => false
  }

  private def unwindReturn(
      stmts: List[p.Stmt],
      phi: p.Term.Select,
      flag: p.Term.Select,
      inLoop: Boolean,
      ctr: java.util.concurrent.atomic.AtomicLong
  ): List[p.Stmt] = {
    def guard(tail: List[p.Stmt]): List[p.Stmt] =
      if (inLoop) List(p.Stmt.Cond(flag, List(p.Stmt.Break), tail))
      else if (tail.isEmpty) Nil
      else List(p.Stmt.Cond(flag, Nil, tail))

    def finalizer(fin: List[p.Stmt]): List[p.Stmt] = {
      val rewritten = unwindReturn(fin, phi, flag, inLoop = false, ctr)
      if (!fin.exists(containsReturn)) rewritten
      else {
        val savedName            = p.Named(s"_inline_pending_${ctr.incrementAndGet()}", p.Type.Bool1)
        val saved: p.Term.Select = p.Term.Select(savedName, Nil, p.Type.Bool1)
        val save                 = p.Stmt.Var(savedName, Some(p.Expr.Alias(flag)), isMutable = false)
        val clear                = p.Stmt.Mut(flag, p.Expr.Alias(p.Term.Bool1Const(false)))
        val restore              = p.Stmt.Mut(flag, p.Expr.Alias(saved))
        save :: clear :: rewritten ::: List(p.Stmt.Cond(flag, Nil, List(restore)))
      }
    }

    stmts match {
      case Nil => Nil
      case p.Stmt.Return(e) :: _ =>
        List(p.Stmt.Mut(phi, e), p.Stmt.Mut(flag, p.Expr.Alias(p.Term.Bool1Const(true)))) :::
          Option.when(inLoop)(p.Stmt.Break).toList
      case p.Stmt.Cond(c, t, f) :: rest =>
        val head = p.Stmt.Cond(c, unwindReturn(t, phi, flag, inLoop, ctr), unwindReturn(f, phi, flag, inLoop, ctr))
        val tail = unwindReturn(rest, phi, flag, inLoop, ctr)
        if (t.exists(containsReturn) || f.exists(containsReturn)) head :: guard(tail) else head :: tail
      case p.Stmt.While(c, b) :: rest =>
        val head = p.Stmt.While(c, unwindReturn(b, phi, flag, inLoop = true, ctr))
        val tail = unwindReturn(rest, phi, flag, inLoop, ctr)
        if (b.exists(containsReturn)) head :: guard(tail) else head :: tail
      case p.Stmt.ForRange(i, lb, ub, step, b) :: rest =>
        val head = p.Stmt.ForRange(i, lb, ub, step, unwindReturn(b, phi, flag, inLoop = true, ctr))
        val tail = unwindReturn(rest, phi, flag, inLoop, ctr)
        if (b.exists(containsReturn)) head :: guard(tail) else head :: tail
      case (t: p.Stmt.Try) :: rest =>
        val head = p.Stmt.Try(
          unwindReturn(t.body, phi, flag, inLoop = false, ctr),
          t.handlers.map(h => h.copy(body = unwindReturn(h.body, phi, flag, inLoop = false, ctr))),
          finalizer(t.fin)
        )
        val tail = unwindReturn(rest, phi, flag, inLoop, ctr)
        if (containsReturn(t)) head :: guard(tail) else head :: tail
      case p.Stmt.Annotated(inner, pos, comment) :: rest =>
        val head = unwindReturn(List(inner), phi, flag, inLoop, ctr).map(p.Stmt.Annotated(_, pos, comment))
        val tail = unwindReturn(rest, phi, flag, inLoop, ctr)
        if (containsReturn(inner)) head ::: guard(tail) else head ::: tail
      case other :: rest => other :: unwindReturn(rest, phi, flag, inLoop, ctr)
    }
  }

  private def alwaysReturns(stmts: List[p.Stmt]): Boolean = stmts.lastOption match {
    case Some(p.Stmt.Return(_))                            => true
    case Some(p.Stmt.Cond(p.Term.Bool1Const(true), t, _))  => alwaysReturns(t)
    case Some(p.Stmt.Cond(p.Term.Bool1Const(false), _, f)) => alwaysReturns(f)
    case Some(p.Stmt.Cond(_, t, f))                        => alwaysReturns(t) && alwaysReturns(f)
    case Some(p.Stmt.Annotated(s, _, _))                   => alwaysReturns(List(s))
    case _                                                 => false
  }

  private def sinkAfterReturn(stmts: List[p.Stmt]): List[p.Stmt] = stmts match {
    case Nil                     => Nil
    case (r: p.Stmt.Return) :: _ => List(r)
    case p.Stmt.Cond(c @ p.Term.Bool1Const(true), t, _) :: rest =>
      List(p.Stmt.Cond(c, sinkAfterReturn(t ::: rest), Nil))
    case p.Stmt.Cond(c @ p.Term.Bool1Const(false), _, f) :: rest =>
      List(p.Stmt.Cond(c, Nil, sinkAfterReturn(f ::: rest)))
    case p.Stmt.Cond(c, t, f) :: rest =>
      val t2 = sinkAfterReturn(t)
      val f2 = sinkAfterReturn(f)
      (alwaysReturns(t2), alwaysReturns(f2)) match {
        case (true, true)                   => List(p.Stmt.Cond(c, t2, f2)) // rest unreachable
        case (true, false) if rest.nonEmpty => List(p.Stmt.Cond(c, t2, sinkAfterReturn(f2 ::: rest)))
        case (false, true) if rest.nonEmpty => List(p.Stmt.Cond(c, sinkAfterReturn(t2 ::: rest), f2))
        case _                              => p.Stmt.Cond(c, t2, f2) :: sinkAfterReturn(rest)
      }
    case p.Stmt.Annotated(inner, pos, comment) :: rest =>
      val head = sinkAfterReturn(List(inner)).map(p.Stmt.Annotated(_, pos, comment))
      if (alwaysReturns(List(inner))) head else head ::: sinkAfterReturn(rest)
    case (t: p.Stmt.Try) :: rest => t.mapBlocks(sinkAfterReturn) :: sinkAfterReturn(rest)
    case other :: rest           => other :: sinkAfterReturn(rest)
  }

  private def stripReturn(s: p.Stmt): List[p.Stmt] = s match {
    case p.Stmt.Return(_)                  => Nil
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, t.flatMap(stripReturn), f.flatMap(stripReturn)) :: Nil
    case p.Stmt.While(c, b)                => p.Stmt.While(c, b.flatMap(stripReturn)) :: Nil
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, b.flatMap(stripReturn)) :: Nil
    case t: p.Stmt.Try                     => t.mapBlocks(_.flatMap(stripReturn)) :: Nil
    case p.Stmt.Annotated(inner, pos, c)   => stripReturn(inner).map(p.Stmt.Annotated(_, pos, c))
    case other                             => other :: Nil
  }

  private def rebindReturn(phi: p.Term.Select)(s: p.Stmt): p.Stmt = s match {
    case p.Stmt.Return(e)                  => p.Stmt.Mut(phi, e)
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, t.map(rebindReturn(phi)), f.map(rebindReturn(phi)))
    case p.Stmt.While(c, b)                => p.Stmt.While(c, b.map(rebindReturn(phi)))
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, b.map(rebindReturn(phi)))
    case t: p.Stmt.Try                     => t.mapBlocks(_.map(rebindReturn(phi)))
    case p.Stmt.Annotated(inner, pos, c)   => p.Stmt.Annotated(rebindReturn(phi)(inner), pos, c)
    case other                             => other
  }

  private def resolveOverload(ivk: p.Expr.Invoke, overloads: OverloadLut): p.Function = {
    def argumentMatches(expected: p.Type, actual: p.Term, allowErasedCallable: Boolean): Boolean = actual match {
      case p.Term.NullPtrConst(comp, _, _) =>
        expected match {
          case p.Type.Ptr(expectedComp, _) => sameType(expectedComp, comp)
          case _                           => false
        }
      case _ =>
        sameType(expected, actual.tpe) ||
        (allowErasedCallable && expected.isInstanceOf[p.Type.Ptr] && (expected match {
          case p.Type.Ptr(p.Type.Nothing, _) => actual.tpe.isInstanceOf[p.Type.FnRef]
          case _                             => false
        }))
    }
    val candidates = overloads.getOrElse((ivk.calleeName, ivk.args.size), Nil)
    def matching(allowErasedCallable: Boolean) = candidates.filter { f =>
      val varToTpeLut = f.tpeVars.map(_.name).zip(appliedTypeArgs(ivk, f.tpeVars.size)).toMap
      val sig =
        if (varToTpeLut.isEmpty) f.signature
        else
          f.signature
            .modifyAll[p.Type](protectExecBinders)
            .modifyAll[p.Type](_.mapLeaf {
              case v @ p.Type.Var(n, _) => varToTpeLut.getOrElse(n, v)
              case x                    => x
            })
            .modifyAll[p.Type](unprotectExecBinders)
      val flatSigParams = sig.moduleCaptures ++ sig.termCaptures ++ sig.args
      sig.receiver.size == ivk.receiver.size &&
      sig.receiver.zip(ivk.receiver).forall((expected, actual) => argumentMatches(expected, actual, false)) &&
      flatSigParams
        .zip(ivk.args)
        .forall((expected, actual) => argumentMatches(expected, actual, allowErasedCallable)) &&
      sameType(sig.rtn, ivk.rtn)
    }
    val exact   = matching(allowErasedCallable = false)
    val matched = if (exact.nonEmpty) exact else matching(allowErasedCallable = true)
    matched match {
      case f :: Nil => f
      case Nil =>
        throw IllegalStateException(
          s"FnInline: no matching overload for ${ivk.repr}; candidates were ${candidates.map(_.repr).mkString("; ")}"
        )
      case xs =>
        throw IllegalStateException(
          s"FnInline: ambiguous overloads for ${ivk.repr}: ${xs.map(_.repr).mkString("; ")}"
        )
    }
  }

  private def inlineExpr(
      expr: p.Expr,
      overloads: OverloadLut,
      ctr: java.util.concurrent.atomic.AtomicLong,
      active: List[String]
  ): (p.Expr, List[p.Stmt], List[p.Arg]) =
    expr match {
      case ivk: p.Expr.Invoke =>
        val callee    = resolveOverload(ivk, overloads)
        val calleeKey = callee.signatureKey
        if (active.contains(calleeKey))
          throw IllegalStateException(
            s"FnInline: recursive call cycle is unsupported: ${(calleeKey :: active).reverse.mkString(" -> ")}"
          )
        val nestedActive                     = calleeKey :: active
        val (resultExpr, inlineStmts, caps)  = inlineOne(ivk, callee, ctr)
        val (rewrittenStmts, nestedCaps)     = inlineBlock(inlineStmts, overloads, ctr, nestedActive)
        val (finalExpr, tailStmts, tailCaps) = inlineExpr(resultExpr, overloads, ctr, nestedActive)
        (finalExpr, rewrittenStmts ::: tailStmts, caps ++ nestedCaps ++ tailCaps)
      case _ => (expr, Nil, Nil)
    }

  private def inlineBlock(
      statements: List[p.Stmt],
      overloads: OverloadLut,
      ctr: java.util.concurrent.atomic.AtomicLong,
      active: List[String]
  ): (List[p.Stmt], List[p.Arg]) = {
    val rewritten = List.newBuilder[p.Stmt]
    val captures  = List.newBuilder[p.Arg]
    statements.foreach { statement =>
      val (nextStatements, nextCaptures) = inlineStmt(statement, overloads, ctr, active)
      rewritten ++= nextStatements
      captures ++= nextCaptures
    }
    (rewritten.result(), captures.result())
  }

  private def inlineStmt(
      stmt: p.Stmt,
      overloads: OverloadLut,
      ctr: java.util.concurrent.atomic.AtomicLong,
      active: List[String]
  ): (List[p.Stmt], List[p.Arg]) = stmt match {
    case p.Stmt.Var(n, Some(e), mut) =>
      val (newE, prepend, caps) = inlineExpr(e, overloads, ctr, active)
      (prepend :+ p.Stmt.Var(n, Some(newE), mut), caps)
    case p.Stmt.Var(_, None, _) => (List(stmt), Nil)
    case p.Stmt.Mut(name, e) =>
      val (newE, prepend, caps) = inlineExpr(e, overloads, ctr, active)
      (prepend :+ p.Stmt.Mut(name, newE), caps)
    case _: p.Stmt.Update => (List(stmt), Nil)
    case p.Stmt.Return(e) =>
      val (newE, prepend, caps) = inlineExpr(e, overloads, ctr, active)
      (prepend :+ p.Stmt.Return(newE), caps)
    case p.Stmt.While(cond, body) =>
      val (newBody, caps) = inlineBlock(body, overloads, ctr, active)
      (List(p.Stmt.While(cond, newBody)), caps)
    case p.Stmt.Cond(cond, t, e) =>
      val (newT, capsT) = inlineBlock(t, overloads, ctr, active)
      val (newE, capsE) = inlineBlock(e, overloads, ctr, active)
      (List(p.Stmt.Cond(cond, newT, newE)), capsT ++ capsE)
    case p.Stmt.ForRange(i, lb, ub, step, body) =>
      val (newBody, caps) = inlineBlock(body, overloads, ctr, active)
      (List(p.Stmt.ForRange(i, lb, ub, step, newBody)), caps)
    case p.Stmt.Try(body, handlers, fin) =>
      val (newBody, capsB) = inlineBlock(body, overloads, ctr, active)
      val newHandlers      = List.newBuilder[p.Handler]
      val handlerCaptures  = List.newBuilder[p.Arg]
      handlers.foreach { handler =>
        val (handlerBody, captures) = inlineBlock(handler.body, overloads, ctr, active)
        newHandlers += handler.copy(body = handlerBody)
        handlerCaptures ++= captures
      }
      val (newFin, capsF) = inlineBlock(fin, overloads, ctr, active)
      (List(p.Stmt.Try(newBody, newHandlers.result(), newFin)), capsB ++ handlerCaptures.result() ++ capsF)
    case p.Stmt.Raise(value, exceptionKind, cleanup) =>
      val (newCleanup, caps) = inlineBlock(cleanup, overloads, ctr, active)
      (List(p.Stmt.Raise(value, exceptionKind, newCleanup)), caps)
    case p.Stmt.Annotated(inner, pos, c) =>
      val (rewritten, caps) = inlineStmt(inner, overloads, ctr, active)
      (rewritten.map(p.Stmt.Annotated(_, pos, c)), caps)
    case _ => (List(stmt), Nil)
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    // per-run counter: names from repeated inlinings stay unique within one program, and the numbering
    // is independent of process compile order (the names embed into emitted kernel images)
    val ctr       = new java.util.concurrent.atomic.AtomicLong(0L)
    val overloads = program.functions.distinct.groupBy(f => (f.name, flatParams(f).size))
    val source    = program.entry.getOrElse(throw IllegalArgumentException("FnInline requires a program entry"))
    val (n, f) = doUntilNotEq(source, limit = 10) { (i, f) =>
      val (stmts, moduleCaptures) = inlineBlock(f.body, overloads, ctr, Nil)
      f.copy(
        decl = f.decl.copy(moduleCaptures = (f.moduleCaptures ++ moduleCaptures).distinct),
        body = stmts
      )
    }

    val remaining = f.collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk }
    if (remaining.nonEmpty)
      throw IllegalStateException(
        s"FnInline did not converge after $n iteration(s); remaining calls: ${remaining.map(_.calleeName.repr).distinct.mkString(", ")}"
      )
    log.info(s"converged in $n iteration(s)")
    program.copy(entry = Some(f), functions = Nil)

  }

}
