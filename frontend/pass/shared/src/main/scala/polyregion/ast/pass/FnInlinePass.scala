package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// inline all calls originating from entry
object FnInlinePass extends ProgramPass {

  // Substitute root names in any Term.Select inside a tree.
  private def substTerms(tree: p.Function, table: Map[p.Named, p.Term]): p.Function =
    tree.modifyAll[p.Term] {
      case p.Term.Select(root, steps, tpe) =>
        table.get(root) match {
          case Some(p.Term.Select(rRoot, rSteps, _)) => p.Term.Select(rRoot, rSteps ::: steps, tpe)
          case Some(other) if steps.isEmpty          => other
          case Some(_) =>
            // Replacement is a non-Select Term but the call wants to apply field steps;
            // shouldn't happen for well-typed substitutions, fall through.
            p.Term.Select(root, steps, tpe)
          case None => p.Term.Select(root, steps, tpe)
        }
      case x => x
    }

  // rename all var and selects to avoid collision
  private def renameAll(f: p.Function): p.Function = {
    def rename(n: p.Named) = p.Named(s"_inline_${f.mangledName}_${n.symbol}", n.tpe)
    val captureNames       = f.moduleCaptures.map(_.named).toSet
    val body = f.body
      .modifyAll[p.Term] {
        case s @ p.Term.Select(root, _, _) if captureNames.contains(root) => s
        case p.Term.Select(root, steps, tpe)                              => p.Term.Select(rename(root), steps, tpe)
        case x                                                            => x
      }
      .modifyAll[p.Stmt] {
        case p.Stmt.Var(n, expr, isMutable) => p.Stmt.Var(rename(n), expr, isMutable)
        case x                            => x
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

  private def inlineOne(ivk: p.Expr.Invoke, f: p.Function): (p.Expr, List[p.Stmt], List[p.Arg]) = {

    val concreteTpeArgs = ivk.receiver
      .map(_.tpe match {
        case p.Type.Struct(_, tpeArgs) => tpeArgs
        case _                         => Nil
      })
      .getOrElse(Nil) ++ ivk.tpeArgs

    val table = f.tpeVars.zip(concreteTpeArgs).toMap

    val renamed = renameAll(f.modifyAll[p.Type](_.mapLeaf {
      case p.Type.Var(name) if table.contains(name) => table(name)
      case x                                        => x
    }))

    // ivk.args is the flattened (args ::: moduleCaptures ::: termCaptures) per D10.
    val targetNames =
      renamed.receiver.map(_.named).toList ++
        renamed.args.map(_.named) ++
        renamed.moduleCaptures.map(_.named) ++
        renamed.termCaptures.map(_.named)
    val replacements = ivk.receiver.toList ++ ivk.args
    val substTable   = targetNames.zip(replacements).toMap
    val substituted  = substTerms(renamed, substTable)

    val returnExprs = substituted.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e }

    returnExprs match {
      case Nil =>
        throw new AssertionError(s"no return in function ${f.signature}")
      case expr :: Nil =>
        // single return: inject the return Expr at the callsite. Drop the Return stmts.
        val noReturnBody = substituted.body.flatMap(stripReturn)
        (expr, noReturnBody, renamed.moduleCaptures)
      case xs =>
        // multi-return: bind to a phi var, replace each Return with a Mut to the phi.
        val phiName             = p.Named("phi", ivk.tpe)
        val phiSelect: p.Term.Select = p.Term.Select(phiName, Nil, ivk.tpe)
        val phiDecl     = p.Stmt.Var(phiName, None, isMutable = true)
        val rebound     = substituted.body.map(rebindReturn(phiSelect))
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

  override def apply(program: p.Program, log: Log): p.Program = {

    val (n, f) = doUntilNotEq(program.entry, limit = 10) { (i, f) =>
      val (stmts, moduleCaptures) = f.body.foldMap { x =>

        val (y, xs) = x.modifyCollect[p.Expr, (List[p.Stmt], List[p.Arg])] {
          case ivk @ p.Expr.Invoke(name, tpeArgs, recv, args, rtn) =>
            val overloads = program.functions.distinct.filter(f => f.name == name && f.args.size == args.size)

            overloads.filter { f =>
              val varToTpeLut = f.tpeVars.zip(tpeArgs).toMap
              val sig = f.signature.modifyAll[p.Type](_.mapLeaf {
                case v @ p.Type.Var(n) => varToTpeLut.getOrElse(n, v)
                case x                 => x
              })
              sig.receiver.size == recv.size &&
              sig.args.zip(args.map(_.tpe)).forall(_ =:= _) &&
              sig.rtn =:= rtn
            } match {
              case Nil =>
                throw new IllegalStateException(
                  s"FnInlinePass: no matching overload for ${ivk.repr}; candidates were ${overloads.map(_.repr).mkString("; ")}"
                )
              case f :: Nil =>
                val (expr, stmts, names) = inlineOne(ivk, f)
                (expr, stmts -> names)
              case xs =>
                throw new IllegalStateException(
                  s"FnInlinePass: ambiguous overloads for ${ivk.repr}: ${xs.map(_.repr).mkString("; ")}"
                )
            }

          case x => (x, Nil -> Nil)
        }

        val (stmts, moduleCaptures) = xs.combineAll
        (stmts :+ y, moduleCaptures)
      }

      f.copy(body = stmts, moduleCaptures = (f.moduleCaptures ++ moduleCaptures).distinct)
    }

    log.info(s"converged in $n iteration(s)")
    p.Program(f, Nil, program.defs, program.phase)

  }

}
