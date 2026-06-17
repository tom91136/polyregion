package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// inlines every Invoke reachable from entry to a fixed point (the result Program has no functions):
// the callee body is type-substituted, alpha-renamed, its params bound to the call args, and spliced
// in at the callsite scope
// edge cases:
//   single Return        -> the return Expr becomes the callsite value; Return stmts stripped
//   multiple Returns     -> a mutable phi var, each Return rebound to a Mut of the phi
//   moduleCaptures       -> NOT alpha-renamed (shared across inlinings); only locals get the per-call id
//   nested Invoke in body -> recursed before splicing so inlining reaches a fixed point
// inlined stmts stay in the original Invoke's enclosing scope - hoisting to the body root would let
// references to loop-/branch-local vars escape their scope
object FnInline extends ProgramPass {

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
    val id                 = ctr.incrementAndGet()
    def rename(n: p.Named) = p.Named(s"_inline_${id}_${f.mangledName}_${n.symbol}", n.tpe)
    val captureNames       = f.moduleCaptures.map(_.named).toSet
    val body = f.body
      .modifyAll[p.Term] {
        case s @ p.Term.Select(root, _, _) if captureNames.contains(root) => s
        case p.Term.Select(root, steps, tpe)                              => p.Term.Select(rename(root), steps, tpe)
        case x                                                            => x
      }
      .modifyAll[p.Stmt] {
        case p.Stmt.Var(n, expr, isMutable) => p.Stmt.Var(rename(n), expr, isMutable)
        case x                              => x
      }
    p.Function(
      f.name,
      f.tpeVars,
      f.receiver.map(arg => arg.copy(rename(arg.named))),
      f.args.map(arg => arg.copy(rename(arg.named))),
      f.moduleCaptures,
      f.termCaptures.map(arg => arg.copy(rename(arg.named))),
      f.rtn,
      body,
      f.visibility,
      f.fpMode,
      f.isEntry
    )
  }

  private def inlineOne(
      ivk: p.Expr.Invoke,
      f: p.Function,
      ctr: java.util.concurrent.atomic.AtomicLong
  ): (p.Expr, List[p.Stmt], List[p.Arg]) = {

    val concreteTpeArgs = ivk.receiver
      .map(_.tpe match {
        case p.Type.Struct(_, tpeArgs) => tpeArgs
        case _                         => Nil
      })
      .getOrElse(Nil) ++ ivk.tpeArgs

    val table = f.tpeVars.zip(concreteTpeArgs).toMap

    val renamed = renameAll(
      f.modifyAll[p.Type](_.mapLeaf {
        case p.Type.Var(name) if table.contains(name) => table(name)
        case x                                        => x
      }),
      ctr
    )

    // ivk.args is the flattened (moduleCaptures ::: termCaptures ::: args) per Compiler.patchIvk.
    val targetNames =
      renamed.receiver.map(_.named).toList ++
        renamed.moduleCaptures.map(_.named) ++
        renamed.termCaptures.map(_.named) ++
        renamed.args.map(_.named)
    val replacements = ivk.receiver.toList ++ ivk.args
    val substTable   = targetNames.zip(replacements).toMap
    val substituted  = substTerms(renamed, substTable)

    val returnExprs = substituted.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e }

    returnExprs match {
      case Nil =>
        throw AssertionError(s"no return in function ${f.signature}")
      case expr :: Nil =>
        val noReturnBody = substituted.body.flatMap(stripReturn)
        (expr, noReturnBody, renamed.moduleCaptures)
      case xs =>
        val phiName                  = p.Named("phi", ivk.tpe)
        val phiSelect: p.Term.Select = p.Term.Select(phiName, Nil, ivk.tpe)
        val phiDecl                  = p.Stmt.Var(phiName, None, isMutable = true)
        val rebound                  = substituted.body.map(rebindReturn(phiSelect))
        (p.Expr.Alias(phiSelect), phiDecl :: rebound, renamed.moduleCaptures)
    }
  }

  private def stripReturn(s: p.Stmt): List[p.Stmt] = s match {
    case p.Stmt.Return(_)                  => Nil
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, t.flatMap(stripReturn), f.flatMap(stripReturn)) :: Nil
    case p.Stmt.While(c, b)                => p.Stmt.While(c, b.flatMap(stripReturn)) :: Nil
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, b.flatMap(stripReturn)) :: Nil
    case p.Stmt.Annotated(inner, pos, c)   => stripReturn(inner).map(p.Stmt.Annotated(_, pos, c))
    case other                             => other :: Nil
  }

  private def rebindReturn(phi: p.Term.Select)(s: p.Stmt): p.Stmt = s match {
    case p.Stmt.Return(e)                  => p.Stmt.Mut(phi, e)
    case p.Stmt.Cond(c, t, f)              => p.Stmt.Cond(c, t.map(rebindReturn(phi)), f.map(rebindReturn(phi)))
    case p.Stmt.While(c, b)                => p.Stmt.While(c, b.map(rebindReturn(phi)))
    case p.Stmt.ForRange(i, lb, ub, st, b) => p.Stmt.ForRange(i, lb, ub, st, b.map(rebindReturn(phi)))
    case p.Stmt.Annotated(inner, pos, c)   => p.Stmt.Annotated(rebindReturn(phi)(inner), pos, c)
    case other                             => other
  }

  private def resolveOverload(ivk: p.Expr.Invoke, program: p.Program): p.Function = {
    def flatParams(f: p.Function): List[p.Type] =
      f.moduleCaptures.map(_.named.tpe) ++ f.termCaptures.map(_.named.tpe) ++ f.args.map(_.named.tpe)
    val candidates = program.functions.distinct.filter(f => f.name == ivk.name && flatParams(f).size == ivk.args.size)
    candidates.filter { f =>
      val varToTpeLut = f.tpeVars.zip(ivk.tpeArgs).toMap
      val sig = f.signature.modifyAll[p.Type](_.mapLeaf {
        case v @ p.Type.Var(n) => varToTpeLut.getOrElse(n, v)
        case x                 => x
      })
      val flatSigParams = sig.moduleCaptures ++ sig.termCaptures ++ sig.args
      sig.receiver.size == ivk.receiver.size &&
      flatSigParams.zip(ivk.args.map(_.tpe)).forall(_ =:= _) &&
      sig.rtn =:= ivk.rtn
    } match {
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
      program: p.Program,
      ctr: java.util.concurrent.atomic.AtomicLong
  ): (p.Expr, List[p.Stmt], List[p.Arg]) =
    expr match {
      case ivk: p.Expr.Invoke =>
        val (resultExpr, inlineStmts, caps) = inlineOne(ivk, resolveOverload(ivk, program), ctr)
        val (rewrittenStmts, nestedCaps)    = inlineStmts.foldMap(s => inlineStmt(s, program, ctr))
        (resultExpr, rewrittenStmts, caps ++ nestedCaps)
      case _ => (expr, Nil, Nil)
    }

  private def inlineStmt(
      stmt: p.Stmt,
      program: p.Program,
      ctr: java.util.concurrent.atomic.AtomicLong
  ): (List[p.Stmt], List[p.Arg]) = stmt match {
    case p.Stmt.Var(n, Some(e), mut) =>
      val (newE, prepend, caps) = inlineExpr(e, program, ctr)
      (prepend :+ p.Stmt.Var(n, Some(newE), mut), caps)
    case p.Stmt.Var(_, None, _) => (List(stmt), Nil)
    case p.Stmt.Mut(name, e) =>
      val (newE, prepend, caps) = inlineExpr(e, program, ctr)
      (prepend :+ p.Stmt.Mut(name, newE), caps)
    case _: p.Stmt.Update => (List(stmt), Nil)
    case p.Stmt.Return(e) =>
      val (newE, prepend, caps) = inlineExpr(e, program, ctr)
      (prepend :+ p.Stmt.Return(newE), caps)
    case p.Stmt.While(cond, body) =>
      val (newBody, caps) = body.foldMap(s => inlineStmt(s, program, ctr))
      (List(p.Stmt.While(cond, newBody)), caps)
    case p.Stmt.Cond(cond, t, e) =>
      val (newT, capsT) = t.foldMap(s => inlineStmt(s, program, ctr))
      val (newE, capsE) = e.foldMap(s => inlineStmt(s, program, ctr))
      (List(p.Stmt.Cond(cond, newT, newE)), capsT ++ capsE)
    case p.Stmt.ForRange(i, lb, ub, step, body) =>
      val (newBody, caps) = body.foldMap(s => inlineStmt(s, program, ctr))
      (List(p.Stmt.ForRange(i, lb, ub, step, newBody)), caps)
    case p.Stmt.Annotated(inner, pos, c) =>
      val (rewritten, caps) = inlineStmt(inner, program, ctr)
      (rewritten.map(p.Stmt.Annotated(_, pos, c)), caps)
    case _ => (List(stmt), Nil)
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    // per-run counter: names from repeated inlinings stay unique within one program, and the numbering
    // is independent of process compile order (the names embed into emitted kernel images)
    val ctr = new java.util.concurrent.atomic.AtomicLong(0L)
    val (n, f) = doUntilNotEq(program.entry, limit = 10) { (i, f) =>
      val (stmts, moduleCaptures) = f.body.foldMap(s => inlineStmt(s, program, ctr))
      f.copy(body = stmts, moduleCaptures = (f.moduleCaptures ++ moduleCaptures).distinct)
    }

    log.info(s"converged in $n iteration(s)")
    program.copy(entry = f, functions = Nil)

  }

}
