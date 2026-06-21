package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// canonicalises derived-pointer temps back to root-anchored accesses, so each access carries its
// array/struct root; a temp collapses only when every use is clean, a bare pointer use keeps the slot
// examples:
//   v = &x; v[0]                      ->  x                  // resolveFieldAliases, whole-value deref
//   v = &x; v[0] = e                  ->  x = e              // resolveFieldAliases, whole-value store
//   p = &a.x; p[i]                    ->  a.x[i]             // resolveFieldAliases, re-root stepped use
//   p = &s.arr[i]; p[0]               ->  s.arr[i]           // resolveArrayFieldIndex, struct array field
//   r = &a[i]; s = &r[j]; s[k]        ->  a[i][j][k]         // resolveLocalArray, multi-dim chain
// edge cases:
//   temp also used bare (`f(v)`)      ->  not folded, decl kept
//   either root or temp reassigned    ->  not folded
final case class Anchor() extends ProgramPass derives PassArgCodec {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  // a temp folds only if every use is clean; any other use escapes the pointer and keeps the slot
  private def foldable(f: p.Function, syms: Set[String])(clean: List[String]): Set[String] = {
    def tally(xs: List[String]): Map[String, Int] = xs.groupMapReduce(identity)(_ => 1)(_ + _)
    val total = tally(f.collectAll[p.Term].collect { case p.Term.Select(n, _, _) if syms(n.symbol) => n.symbol })
    val cln   = tally(clean)
    syms.filter(s => total.getOrElse(s, 0) == cln.getOrElse(s, 0))
  }

  private def isIdentityRef(idx: Option[p.Term], selTpe: p.Type): Boolean = (idx, selTpe) match {
    case (None, _)                                                            => true
    case (Some(p.Term.IntS64Const(0) | p.Term.IntU64Const(0)), _: p.Type.Ptr) => true
    case _                                                                    => false
  }

  private def resolveFieldAliases(f: p.Function, reassigned: Set[String]): (p.Function, Int) = {
    val aliases = f.collectAll[p.Stmt].foldLeft(Map.empty[String, (p.Named, List[p.PathStep])]) { (m, s) =>
      s match {
        case p.Stmt.Var(n, Some(p.Expr.RefTo(p.Term.Select(b, path, selTpe), idx, _, _, _)), false)
            if isIdentityRef(idx, selTpe) && !reassigned(b.symbol) =>
          val (root, base) = m.getOrElse(b.symbol, (b, Nil))
          m + (n.symbol -> (root, base ++ path))
        case _ => m
      }
    }
    if (aliases.isEmpty) (f, 0)
    else {
      def wholeValue(sym: String, comp: p.Type): Option[p.Term.Select] = aliases.get(sym) match {
        case Some((root, Nil)) if root.tpe == comp => Some(p.Term.Select(root, Nil, comp))
        case _                                     => None
      }
      // fold a whole-value deref so the local is never reached through its Global-typed `&x` (IGC reads
      // that as shared, not per-lane private storage)
      val derefFolded = f.modifyAll[p.Expr] {
        case e @ p.Expr.Index(p.Term.Select(n, Nil, _), idx, comp) if isZeroConst(idx) =>
          wholeValue(n.symbol, comp).fold(e: p.Expr)(p.Expr.Alias(_))
        case e => e
      }
      // re-root stepped uses; a bare `_v` is the pointer value itself and keeps its slot
      val resolved = derefFolded.modifyAll[p.Term] {
        case p.Term.Select(n, steps, t) if steps.nonEmpty && aliases.contains(n.symbol) =>
          val (root, path) = aliases(n.symbol); p.Term.Select(root, path ::: steps, t)
        case t => t
      }
      val storeFolded = resolved.copy(body = mapStmtsRec(resolved.body) {
        case s @ p.Stmt.Update(p.Term.Select(n, Nil, _), idx, v) if isZeroConst(idx) =>
          wholeValue(n.symbol, v.tpe).fold(List(s))(w => List(p.Stmt.Mut(w, p.Expr.Alias(v))))
        case s => List(s)
      })
      val bareUsed = storeFolded
        .collectAll[p.Term]
        .collect { case p.Term.Select(n, Nil, _) if aliases.contains(n.symbol) => n.symbol }
        .toSet
      (storeFolded.copy(body = dropAliasDecls(storeFolded.body, aliases.keySet -- bareUsed)), aliases.size)
    }
  }

  private def isArr(t: p.Type): Boolean    = t match { case _: p.Type.Arr => true; case _ => false }
  private def isArrPtr(t: p.Type): Boolean = t match { case p.Type.Ptr(_: p.Type.Arr, _) => true; case _ => false }
  private def peelArr(t: p.Type, n: Int): p.Type =
    if (n <= 0) t else t match { case p.Type.Arr(c, _, _) => peelArr(c, n - 1); case _ => t }

  private def canonicalise(
      f: p.Function,
      syms: Set[String],
      seeds: => List[String],
      exprRw: (Set[String], p.Expr) => p.Expr,
      stmtRw: (Set[String], p.Stmt) => List[p.Stmt]
  ): (p.Function, Int) = {
    val drop = if (syms.isEmpty) Set.empty[String] else foldable(f, syms)(seeds)
    if (drop.isEmpty) (f, 0)
    else {
      val rewritten = f.modifyAll[p.Expr](exprRw(drop, _))
      val body = mapStmtsRec(rewritten.body) {
        case p.Stmt.Var(n, _, _) if drop(n.symbol) => Nil
        case s                                     => stmtRw(drop, s)
      }
      (f.copy(body = body), drop.size)
    }
  }

  private def resolveLocalArray(f: p.Function, reassigned: Set[String]): (p.Function, Int) = {
    val chain = f.collectAll[p.Stmt].foldLeft(Map.empty[String, (p.Named, List[p.Term])]) { (m, s) =>
      s match {
        case p.Stmt.Var(n, Some(p.Expr.RefTo(p.Term.Select(b, Nil, bt), Some(idx), _, _, _)), _)
            if !reassigned(b.symbol) && !reassigned(n.symbol) =>
          bt match {
            case _: p.Type.Arr => m + (n.symbol -> (b, List(idx)))
            case _             => m.get(b.symbol).fold(m) { case (root, idxs) => m + (n.symbol -> (root, idxs :+ idx)) }
          }
        case _ => m
      }
    }
    val syms = chain.keySet
    def rooted(n: p.Named, leafIdx: p.Term): (p.Term.Select, p.Term) = {
      val (root, idxs) = chain(n.symbol)
      val full         = if (isArrPtr(n.tpe)) idxs :+ leafIdx else idxs
      val steps        = full.init.map(p.PathStep.IndexDyn(_))
      (p.Term.Select(root, steps, peelArr(root.tpe, steps.size)), full.last)
    }
    canonicalise(
      f,
      syms,
      f.collectAll[p.Stmt].collect {
        case p.Stmt.Update(p.Term.Select(n, Nil, st), idx, _) if syms(n.symbol) && (isArrPtr(st) || isZeroConst(idx)) =>
          n.symbol
        case p.Stmt.Var(w, Some(p.Expr.RefTo(p.Term.Select(n, Nil, _), _, _, _, _)), _)
            if syms(n.symbol) && syms(w.symbol) =>
          n.symbol
      } ::: f.collectAll[p.Expr].collect {
        case p.Expr.Index(p.Term.Select(n, Nil, st), idx, _) if syms(n.symbol) && (isArrPtr(st) || isZeroConst(idx)) =>
          n.symbol
      },
      (drop, e) =>
        e match {
          case p.Expr.Index(p.Term.Select(n, Nil, _), idx, comp) if drop(n.symbol) =>
            val (r, a) = rooted(n, idx); p.Expr.Index(r, a, comp)
          case _ => e
        },
      (drop, s) =>
        s match {
          case p.Stmt.Update(p.Term.Select(n, Nil, _), idx, v) if drop(n.symbol) =>
            val (r, a) = rooted(n, idx); List(p.Stmt.Update(r, a, v))
          case _ => List(s)
        }
    )
  }

  // re-root `&b.fld[i]` (element address of an array-typed struct field, as a reference-returning
  // operator[] emits) to the access chain `b.fld[i]`; logical SPIR-V cannot carry the interior pointer
  private def resolveArrayFieldIndex(f: p.Function, reassigned: Set[String]): (p.Function, Int) = {
    val chain = f.collectAll[p.Stmt].foldLeft(Map.empty[String, (p.Named, List[p.PathStep])]) { (m, s) =>
      s match {
        case p.Stmt
              .Var(n, Some(p.Expr.RefTo(p.Term.Select(b, List(p.PathStep.Field(fld)), selTpe), Some(idx), _, _, _)), _)
            if isArr(selTpe) && !reassigned(b.symbol) && !reassigned(n.symbol) =>
          val (root, path) = m.getOrElse(b.symbol, (b, Nil))
          m + (n.symbol -> (root, path ::: List[p.PathStep](p.PathStep.Field(fld), p.PathStep.IndexDyn(idx))))
        case _ => m
      }
    }
    val syms = chain.keySet
    def rooted(n: p.Named, tpe: p.Type): p.Term.Select = {
      val (root, path) = chain(n.symbol); p.Term.Select(root, path, tpe)
    }
    canonicalise(
      f,
      syms,
      f.collectAll[p.Stmt].collect {
        case p.Stmt.Update(p.Term.Select(n, Nil, _), idx, _) if syms(n.symbol) && isZeroConst(idx) => n.symbol
        case p.Stmt.Var(w, Some(p.Expr.RefTo(p.Term.Select(n, List(p.PathStep.Field(_)), selTpe), _, _, _, _)), _)
            if syms(n.symbol) && syms(w.symbol) && isArr(selTpe) =>
          n.symbol
      } ::: f.collectAll[p.Expr].collect {
        case p.Expr.Index(p.Term.Select(n, Nil, _), idx, _) if syms(n.symbol) && isZeroConst(idx) => n.symbol
      },
      (drop, e) =>
        e match {
          case p.Expr.Index(p.Term.Select(n, Nil, _), idx, comp) if drop(n.symbol) && isZeroConst(idx) =>
            p.Expr.Alias(rooted(n, comp))
          case _ => e
        },
      (drop, s) =>
        s match {
          case p.Stmt.Update(p.Term.Select(n, Nil, _), idx, v) if drop(n.symbol) && isZeroConst(idx) =>
            List(p.Stmt.Mut(rooted(n, v.tpe), p.Expr.Alias(v)))
          case _ => List(s)
        }
    )
  }

  private def run(f: p.Function): (p.Function, Int) = {
    // index operands stay sound across the fold: the remapper emits each &-chain immediately before its
    // one access, so moving operand evaluation from chain-def to use site cannot observe a different value
    val reassigned    = Provenance.reassignedIn(f)
    val (f1, aliased) = resolveFieldAliases(f, reassigned)
    val (f2, arrFlds) = resolveArrayFieldIndex(f1, reassigned)
    val (f3, locals)  = resolveLocalArray(f2, reassigned)
    (f3, aliased + arrFlds + locals)
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val (entry, ec)      = run(program.entry)
    val (functions, fcs) = program.functions.map(run).unzip
    val total            = ec + fcs.sum
    if (total > 0) log.info(s"canonicalised $total derived-pointer temp(s) to root-anchored accesses")
    program.copy(entry = entry, functions = functions)
  }
}
