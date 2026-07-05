package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// online partial evaluator over the ANF IR: one flow-sensitive forward walk per function, iterated to a
// fixed point, that folds constants, propagates copies and addresses, prunes static control
// flow and drops dead pure bindings. it threads a store binding each immutable, non-reassigned,
// non-address-taken val to a literal (constant prop), another lvalue path (copy prop), or the address of an
// lvalue (from `RefTo(x)`, so a `*p`/`p[0]` deref folds back to the lvalue). two post-passes in the same
// fixpoint reassociate constants across integer +/* chains and CSE identical register-arithmetic; both emit
// aliases the next iteration copy-propagates away. exclusion (reassigned union address-taken, whole-
// function) is flow-insensitive so a binding forwards into loop/branch bodies soundly, and the outer
// fixpoint recovers what a single walk misses
// examples:
//   a = 2; b = a + 1              ->  b = 3                    // const fold + propagate
//   b = a + 1; c = b + 2          ->  c = a + 3                // reassociate, dead b dropped
//   a = x*y; b = x*y; a + b       ->  a = x*y; a + a           // CSE (b aliased to a, dropped)
//   p = &x; *p                    ->  x                        // address/deref fold, dead &x dropped
//   x + 0 / x * 1 / x ^ x         ->  x / x / 0                // algebraic identities
//   if (true) { t } else { f }    ->  t                        // static branch; taken env escapes
//   if (c) { r } else { r }       ->  r                        // identical branches collapse (c is pure)
//   while (false) body            ->  drop
//   return v; dead()              ->  return v                 // unreachable tail dropped
//   s.#empty_struct_storage = 0   ->  drop                     // padding-store peephole
// edge cases:
//   integral Div/Rem by 0                        ->  not folded (left to trap at runtime)
//   mutated / address-taken name                 ->  never const/copy-bound; substitution stays sound
//   float x + 0 / x == x                         ->  not simplified (IEEE signed-zero / NaN)
//   Math/Spec/OffsetOf/SizeOf/Invoke/load/Alloc  ->  left residual, never folded or dropped
//   memory-read operand                          ->  never CSE'd or reassociated (a store/call may change it)
//
// `canonicaliseAddresses=true` runs only the address canonicalisation - root-anchoring the derived-pointer
// temps logical SPIR-V cannot carry (`&x`, `&s.arr[i]`, multi-dim `&a[i][j]`) - with no folding or DCE, so
// it stays safe PostMono after StructuredExit injects its own derived pointers, leaving the #error writes intact
final case class PartialEval(canonicaliseAddresses: Boolean = false) extends ProgramPass derives PassArgCodec {

  // vals: name -> constant literal or aliased lvalue Select; addrs: pointer name -> the lvalue it addresses
  private final case class St(vals: Map[p.Named, p.Term], addrs: Map[p.Named, p.Term.Select]) {
    def bindVal(n: p.Named, t: p.Term): St         = St(vals + (n -> t), addrs)
    def bindAddr(n: p.Named, s: p.Term.Select): St = St(vals, addrs + (n -> s))
  }
  private object St { val empty: St = St(Map.empty, Map.empty) }

  override def apply(program: p.Program, log: Log): p.Program =
    if (canonicaliseAddresses) {
      val (entry, ec)      = canonicaliseFn(program.entry)
      val (functions, fcs) = program.functions.map(canonicaliseFn).unzip
      val total            = ec + fcs.sum
      if (total > 0) log.info(s"canonicalised $total derived-pointer temp(s) to root-anchored accesses")
      program.copy(entry = entry, functions = functions)
    } else
      program.copy(
        entry = foldFn(program.entry, log.subLog(s"PartialEval on ${program.entry.name}")),
        functions = program.functions.map(f => foldFn(f, log.subLog(s"PartialEval on ${f.name}")))
      )

  private def foldFn(f: p.Function, log: Log): p.Function = {
    val (n, reduced) = doUntilNotEq(f) { (_, f) =>
      // a name is unsafe to substitute if it is ever reassigned (Mut) or address-taken (RefTo) anywhere in
      // the function; whole-function so forward substitution into loop/branch bodies stays sound
      val addrTaken = addressTakenNames(f)
      val excluded  = mutatedNames(f) ++ addrTaken
      // addr-bind only cares about a bare re-aim (`a = ...`); a stepped write-through leaves `a`'s slot stable
      val reassigned = f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(n, Nil, _), _) => n }.toSet
      // fold; reassociate constants across integer +/* chains; CSE identical register-arithmetic; drop
      // dead bindings. reassoc/CSE emit aliases that the next fold iteration copy-propagates and drops.
      val folded  = f.copy(body = evalStmts(f.body, St.empty, excluded, reassigned, log)._1)
      val reassoc = folded.copy(body = reassocStmts(folded.body, Map.empty)._1)
      val cse     = reassoc.copy(body = cseStmts(reassoc.body, Map.empty, addrTaken)._1)
      cse.copy(body = dropDeadBindings(cse.body, selectRoots(cse.body)))
    }
    log.info(s"PartialEval stable after $n passes")
    reduced
  }

  private def mutatedNames(f: p.Function): Set[p.Named] =
    f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(name, _, _), _) => name }.toSet

  private def addressTakenNames(f: p.Function): Set[p.Named] =
    f.collectAll[p.Expr].collect { case p.Expr.RefTo(p.Term.Select(name, _, _), _, _, _, _) => name }.toSet

  // fold-left threading the store; once a statement emits an unconditional terminator the rest of the block
  // is unreachable and dropped
  private def evalStmts(
      stmts: List[p.Stmt],
      st0: St,
      excluded: Set[p.Named],
      reassigned: Set[p.Named],
      log: Log
  ): (List[p.Stmt], St) = {
    @annotation.tailrec
    def loop(rem: List[p.Stmt], acc: List[p.Stmt], st: St): (List[p.Stmt], St) = rem match {
      case Nil => (acc.reverse, st)
      case s :: rest =>
        val (out, st1) = evalStmt(s, st, excluded, reassigned, log)
        val acc1       = out reverse_::: acc
        if (out.exists(isTerminator)) (acc1.reverse, st1)
        else loop(rest, acc1, st1)
    }
    loop(stmts, Nil, st0)
  }

  private def isTerminator(s: p.Stmt): Boolean = s match {
    case _: p.Stmt.Return | p.Stmt.Break | p.Stmt.Cont => true
    case p.Stmt.Annotated(inner, _, _)                 => isTerminator(inner)
    case _                                             => false
  }

  private def evalStmt(
      s: p.Stmt,
      st: St,
      excluded: Set[p.Named],
      reassigned: Set[p.Named],
      log: Log
  ): (List[p.Stmt], St) = s match {
    case p.Stmt.Var(name, None, mut) => (p.Stmt.Var(name, None, mut) :: Nil, st)
    case p.Stmt.Var(name, Some(e), mut) =>
      val folded = evalExpr(e, st)
      val st2 =
        if (mut || excluded.contains(name)) st
        else
          folded match {
            case p.Expr.Alias(c) if Fold.isConstTerm(c) =>
              log.info(s"const-bind ${name.repr} = ${c.repr}")
              st.bindVal(name, c)
            case p.Expr.Alias(sel: p.Term.Select) if !excluded.contains(sel.root) =>
              log.info(s"copy-bind ${name.repr} = ${sel.repr}")
              st.bindVal(name, sel)
            case p.Expr.RefTo(sel: p.Term.Select, None, _, _, _) if !reassigned.contains(sel.root) =>
              log.info(s"addr-bind ${name.repr} = &${sel.repr}")
              st.bindAddr(name, sel)
            case _ => st
          }
      (p.Stmt.Var(name, Some(folded), mut) :: Nil, st2)

    case p.Stmt.Mut(lhs, e) =>
      if (isPaddingMut(lhs) && isPure(e)) {
        log.info(s"drop padding store ${lhs.repr}")
        (Nil, st)
      } else (p.Stmt.Mut(resolveSelect(lhs, st), evalExpr(e, st)) :: Nil, st)

    case p.Stmt.Update(lhs, idx, value) =>
      val idx2 = resolveTerm(idx, st)
      val v2   = resolveTerm(value, st)
      lhs match {
        // (&lvalue)[0] = v  ->  lvalue = v  (whole-value store through a tracked address)
        case p.Term.Select(ptr, Nil, _) if isZeroConst(idx2) && st.addrs.contains(ptr) =>
          val a = st.addrs(ptr)
          (p.Stmt.Mut(p.Term.Select(a.root, a.steps, v2.tpe), p.Expr.Alias(v2)) :: Nil, st)
        case _ => (p.Stmt.Update(resolveSelect(lhs, st), idx2, v2) :: Nil, st)
      }

    case p.Stmt.While(cond, body) =>
      resolveTerm(cond, st) match {
        case p.Term.Bool1Const(false) =>
          log.info("while(false) -> drop")
          (Nil, st)
        case c2 => (p.Stmt.While(c2, evalStmts(body, st, excluded, reassigned, log)._1) :: Nil, st)
      }

    case p.Stmt.ForRange(induction, lb, ub, step, body) =>
      val lb2   = resolveTerm(lb, st)
      val ub2   = resolveTerm(ub, st)
      val step2 = resolveTerm(step, st)
      val empty = (Fold.asLong(lb2), Fold.asLong(ub2), Fold.asLong(step2)) match {
        case (Some(l), Some(u), Some(s)) if s > 0 => l >= u
        case (Some(l), Some(u), Some(s)) if s < 0 => l <= u
        case _                                    => false
      }
      if (empty) {
        log.info(s"for(${induction.repr} = ${lb2.repr}; < ${ub2.repr}; += ${step2.repr}) is empty -> drop")
        (Nil, st)
      } else
        (p.Stmt.ForRange(induction, lb2, ub2, step2, evalStmts(body, st, excluded, reassigned, log)._1) :: Nil, st)

    case p.Stmt.Break => (p.Stmt.Break :: Nil, st)
    case p.Stmt.Cont  => (p.Stmt.Cont :: Nil, st)

    case p.Stmt.Cond(cond, t, f) =>
      resolveTerm(cond, st) match {
        case p.Term.Bool1Const(true) =>
          log.info("if(true) { t } else { f } -> t")
          evalStmts(t, st, excluded, reassigned, log)
        case p.Term.Bool1Const(false) =>
          log.info("if(false) { t } else { f } -> f")
          evalStmts(f, st, excluded, reassigned, log)
        case c2 =>
          val tS = evalStmts(t, st, excluded, reassigned, log)._1
          val fS = evalStmts(f, st, excluded, reassigned, log)._1
          // identical branches (e.g. both folded to the same value, or both empty) make the guard dead;
          // the guard is a pure Term so it can be dropped
          if (tS == fS) {
            if (tS.nonEmpty) log.info("if(c) { s } else { s } -> s")
            (tS, st)
          } else (p.Stmt.Cond(c2, tS, fS) :: Nil, st)
      }

    case p.Stmt.Return(value) => (p.Stmt.Return(evalExpr(value, st)) :: Nil, st)

    case p.Stmt.Annotated(inner, pos, c) =>
      val (xs, st2) = evalStmt(inner, st, excluded, reassigned, log)
      (xs.map(p.Stmt.Annotated(_, pos, c)), st2)
  }

  private def evalExpr(e: p.Expr, st: St): p.Expr = e match {
    case p.Expr.Alias(t)   => p.Expr.Alias(resolveTerm(t, st))
    case p.Expr.SpecOp(op) => p.Expr.SpecOp(op.modifyAll[p.Term](resolveTerm(_, st)))
    case p.Expr.MathOp(op) => p.Expr.MathOp(op.modifyAll[p.Term](resolveTerm(_, st)))
    case p.Expr.IntrOp(op) =>
      val op2 = op.modifyAll[p.Term](resolveTerm(_, st))
      Fold.tryFoldIntr(op2).orElse(Fold.trySimplifyIntr(op2)) match {
        case Some(c) => p.Expr.Alias(c)
        case None    => p.Expr.IntrOp(op2)
      }
    case p.Expr.Cast(from, as) =>
      val from2 = resolveTerm(from, st)
      Fold.tryFoldCast(from2, as) match {
        case Some(c) => p.Expr.Alias(c)
        case None    => p.Expr.Cast(from2, as)
      }
    case p.Expr.Index(lhs, idx, comp) =>
      val lhs2 = resolveTerm(lhs, st)
      val idx2 = resolveTerm(idx, st)
      lhs2 match {
        // (&lvalue)[0] loads *(&lvalue) == the lvalue
        case p.Term.Select(ptr, Nil, _) if isZeroConst(idx2) && st.addrs.contains(ptr) =>
          val a = st.addrs(ptr)
          p.Expr.Alias(p.Term.Select(a.root, a.steps, comp))
        case _ => p.Expr.Index(lhs2, idx2, comp)
      }
    case p.Expr.RefTo(lhs, idx, comp, sp, r) =>
      p.Expr.RefTo(resolveTerm(lhs, st), idx.map(resolveTerm(_, st)), comp, sp, r)
    case p.Expr.Alloc(comp, size, sp, r) => p.Expr.Alloc(comp, resolveTerm(size, st), sp, r)
    case p.Expr.Invoke(n, ts, recv, args, rtn) =>
      p.Expr.Invoke(n, ts, recv.map(resolveTerm(_, st)), args.map(resolveTerm(_, st)), rtn)
    case p.Expr.ForeignCall(n, args, rtn) => p.Expr.ForeignCall(n, args.map(resolveTerm(_, st)), rtn)
    case o: p.Expr.OffsetOf               => o
    case o: p.Expr.SizeOf                 => o
  }

  // constant propagation substitutes a bound literal into a bare Select; copy propagation rebases a Select
  // onto its alias root; address folding turns a `*p` deref of a tracked `p = &lvalue` back into the lvalue
  private def resolveTerm(t: p.Term, st: St): p.Term = t match {
    case p.Term.Select(root, steps, tpe) =>
      st.vals.get(root) match {
        case Some(c) if steps.isEmpty && Fold.isConstTerm(c) => c
        case Some(s: p.Term.Select)                          => p.Term.Select(s.root, s.steps ::: steps, tpe)
        case _ =>
          (steps, st.addrs.get(root)) match {
            case (p.PathStep.Deref :: rest, Some(a)) => p.Term.Select(a.root, a.steps ::: rest, tpe)
            case _                                   => t
          }
      }
    case other => other
  }

  // a write target follows copy/address bindings but never collapses to a literal (it must stay an lvalue)
  private def resolveSelect(t: p.Term.Select, st: St): p.Term.Select = resolveTerm(t, st) match {
    case s: p.Term.Select => s
    case _                => t
  }

  private def dropDeadBindings(body: List[p.Stmt], referenced: Set[p.Named]): List[p.Stmt] =
    mapStmtsRec(body) {
      case p.Stmt.Var(n, rhs, _) if !referenced.contains(n) && rhs.forall(isPure) => Nil
      case s                                                                      => List(s)
    }

  private def isPaddingMut(lhs: p.Term.Select): Boolean =
    lhs.steps.lastOption.exists {
      case p.PathStep.Field(f) => f == p.Conventions.EmptyStructStorageField
      case _                   => false
    }

  // pure = no side effect and no memory read, so a dead binding is safe to drop; address-of (RefTo) is
  // pure and dropping it also lets a no-longer-address-taken root become foldable next iteration; loads
  // (Index), Alloc, calls and Spec ops are conservatively retained
  private def isPure(e: p.Expr): Boolean = e match {
    case _: p.Expr.Alias | _: p.Expr.IntrOp | _: p.Expr.MathOp | _: p.Expr.Cast | _: p.Expr.RefTo | _: p.Expr.OffsetOf |
        _: p.Expr.SizeOf =>
      true
    case _ => false
  }

  // ---- constant reassociation: collapse constants scattered across integer +/-/* chains
  //   (x + 1) + 2 -> x + 3 ,  (i * 4) * 2 -> i * 8
  // two's-complement add and multiply are associative under fixed-width wraparound, so this is sound for
  // any integer type; floats are excluded (reassociation is not IEEE-safe). each immutable integer binding
  // of the form `base (+|*) const` is recorded; a later op on it folds the constants together. the chain
  // map is threaded straight-line and cleared at any store/call/control-flow boundary (chain links are
  // otherwise consecutive), which keeps the recorded base's value stable between definition and reuse
  private type Chain = (Boolean, p.Term, Long, p.Type) // (isMul, base, const, tpe)

  private def reassocStmts(stmts: List[p.Stmt], env0: Map[p.Named, Chain]): (List[p.Stmt], Map[p.Named, Chain]) = {
    val (rev, env) = stmts.foldLeft((List.empty[p.Stmt], env0)) { case ((acc, env), s) =>
      val (s2, env2) = reassocStmt(s, env)
      (s2 :: acc, env2)
    }
    (rev.reverse, env)
  }

  private def reassocStmt(s: p.Stmt, env: Map[p.Named, Chain]): (p.Stmt, Map[p.Named, Chain]) = s match {
    case p.Stmt.Cond(c, t, f) =>
      (p.Stmt.Cond(c, reassocStmts(t, Map.empty)._1, reassocStmts(f, Map.empty)._1), Map.empty)
    case p.Stmt.While(c, b)                => (p.Stmt.While(c, reassocStmts(b, Map.empty)._1), Map.empty)
    case p.Stmt.ForRange(i, lb, ub, st, b) => (p.Stmt.ForRange(i, lb, ub, st, reassocStmts(b, Map.empty)._1), Map.empty)
    case p.Stmt.Annotated(inner, pos, c) =>
      val (i2, e2) = reassocStmt(inner, env); (p.Stmt.Annotated(i2, pos, c), e2)
    case p.Stmt.Var(n, Some(p.Expr.IntrOp(op)), false) if !op.terms.exists(termReadsMemory) =>
      reassocOp(op, env) match {
        case Some(ch @ (isMul, base, k, t)) =>
          val rhs = Fold
            .intTerm(t, k)
            .map(kc => p.Expr.IntrOp(if (isMul) p.Intr.Mul(base, kc, t) else p.Intr.Add(base, kc, t)))
            .getOrElse(p.Expr.IntrOp(op))
          (p.Stmt.Var(n, Some(rhs), false), env + (n -> ch))
        case None => (s, env)
      }
    case p.Stmt.Var(_, Some(e), _) =>
      e match { case _: p.Expr.Invoke | _: p.Expr.ForeignCall => (s, Map.empty); case _ => (s, env) }
    case _: p.Stmt.Mut | _: p.Stmt.Update => (s, Map.empty)
    case other                            => (other, env)
  }

  private def reassocOp(op: p.Intr, env: Map[p.Named, Chain]): Option[Chain] = op match {
    case p.Intr.Add(a, b, t) if t.kind == p.Type.Kind.Integral => constOperand(false, a, b, t, env)
    case p.Intr.Mul(a, b, t) if t.kind == p.Type.Kind.Integral => constOperand(true, a, b, t, env)
    case p.Intr.Sub(a, b, t) if t.kind == p.Type.Kind.Integral => Fold.asLong(b).map(c => extend(false, a, -c, t, env))
    case _                                                     => None
  }
  private def constOperand(isMul: Boolean, a: p.Term, b: p.Term, t: p.Type, env: Map[p.Named, Chain]): Option[Chain] =
    (Fold.asLong(a), Fold.asLong(b)) match {
      case (_, Some(c)) => Some(extend(isMul, a, c, t, env))
      case (Some(c), _) => Some(extend(isMul, b, c, t, env))
      case _            => None
    }
  private def extend(isMul: Boolean, base: p.Term, c: Long, t: p.Type, env: Map[p.Named, Chain]): Chain =
    base match {
      case p.Term.Select(m, Nil, _) =>
        env.get(m) match {
          case Some((`isMul`, b0, k0, _)) => (isMul, b0, if (isMul) k0 * c else k0 + c, t)
          case _                          => (isMul, base, c, t)
        }
      case _ => (isMul, base, c, t)
    }

  // ---- common subexpression elimination: dedup identical pure register-arithmetic expressions (no memory
  // reads). the available-expression map is threaded straight-line; a later Var with a structurally-equal
  // RHS aliases the earlier binding (the next fold iteration copy-propagates and drops it). an entry is
  // dropped when a store or call could change one of its operands (its root is reassigned, or a
  // pointer-store/call could alias an address-taken operand); control-flow bodies start fresh. each entry
  // caches its operand roots so invalidation is a set test rather than a re-traversal of the expression
  private type Avail = Map[p.Expr, (p.Named, Set[p.Named])]
  private def cseStmts(stmts: List[p.Stmt], avail0: Avail, addrTaken: Set[p.Named]): (List[p.Stmt], Avail) = {
    val (rev, av) = stmts.foldLeft((List.empty[p.Stmt], avail0)) { case ((acc, avail), s) =>
      val (s2, avail2) = cseStmt(s, avail, addrTaken)
      (s2 :: acc, avail2)
    }
    (rev.reverse, av)
  }

  private def cseStmt(s: p.Stmt, avail: Avail, addrTaken: Set[p.Named]): (p.Stmt, Avail) =
    s match {
      case p.Stmt.Cond(c, t, f) =>
        (p.Stmt.Cond(c, cseStmts(t, Map.empty, addrTaken)._1, cseStmts(f, Map.empty, addrTaken)._1), Map.empty)
      case p.Stmt.While(c, b) => (p.Stmt.While(c, cseStmts(b, Map.empty, addrTaken)._1), Map.empty)
      case p.Stmt.ForRange(i, lb, ub, st, b) =>
        (p.Stmt.ForRange(i, lb, ub, st, cseStmts(b, Map.empty, addrTaken)._1), Map.empty)
      case p.Stmt.Annotated(inner, pos, c) =>
        val (i2, av) = cseStmt(inner, avail, addrTaken); (p.Stmt.Annotated(i2, pos, c), av)
      case p.Stmt.Var(n, Some(e), false) if cseEligible(e) =>
        avail.get(e) match {
          case Some((m, _)) => (p.Stmt.Var(n, Some(p.Expr.Alias(p.Term.Select(m, Nil, n.tpe))), false), avail)
          case None         => (s, avail + (e -> (n, selectRoots(e))))
        }
      case _ =>
        val kill = killedRoots(s, addrTaken)
        (s, if (kill.isEmpty) avail else avail.filterNot { case (_, (_, roots)) => roots.exists(kill) })
    }

  private def cseEligible(e: p.Expr): Boolean = (e match {
    case _: p.Expr.IntrOp | _: p.Expr.MathOp | _: p.Expr.Cast | _: p.Expr.OffsetOf | _: p.Expr.SizeOf => true
    case _                                                                                            => false
  }) && !readsMemory(e)

  private def readsMemory(e: p.Expr): Boolean =
    e.collectFirst_[p.Term] { case t if termReadsMemory(t) => t }.isDefined

  // a Select reads memory if it dereferences/indexes a pointer, or projects a field off a pointer root
  // (a `ptr->field` load) - the latter only appears PostMono, but classifying it here keeps CSE and reassoc
  // sound wherever the pass is scheduled, rather than relying on running at Initial phase (where every
  // operand is a register value or a struct-value projection and the only memory reads are array Index)
  private def termReadsMemory(t: p.Term): Boolean = t match {
    case p.Term.Select(root, steps, _) => steps.exists(isMemStep) || (Provenance.isPtr(root.tpe) && steps.nonEmpty)
    case _                             => false
  }
  private def isMemStep(s: p.PathStep): Boolean = s match {
    case p.PathStep.Deref                             => true
    case _: p.PathStep.Index | _: p.PathStep.IndexDyn => true
    case _                                            => false
  }

  // roots whose value a statement may change: a Mut/Update target, plus every address-taken root when the
  // statement stores through a pointer (Update) or calls out (which can write aliased memory)
  private def killedRoots(s: p.Stmt, addrTaken: Set[p.Named]): Set[p.Named] = s match {
    case p.Stmt.Mut(p.Term.Select(root, steps, _), e) =>
      Set(root) ++ (if (steps.isEmpty) Set.empty else addrTaken) ++ callKills(e, addrTaken)
    case p.Stmt.Update(p.Term.Select(root, _, _), _, _) => Set(root) ++ addrTaken
    case p.Stmt.Var(_, Some(e), _)                      => callKills(e, addrTaken)
    case p.Stmt.Return(e)                               => callKills(e, addrTaken)
    case _                                              => Set.empty
  }
  private def callKills(e: p.Expr, addrTaken: Set[p.Named]): Set[p.Named] = e match {
    case _: p.Expr.Invoke | _: p.Expr.ForeignCall => addrTaken
    case _                                        => Set.empty
  }

  // ---- address canonicalisation (canonicaliseAddresses=true): root-anchor derived-pointer temps back to
  // array/struct-rooted access chains, so each access carries its root; logical SPIR-V cannot carry the
  // interior pointer. a temp collapses only when every use is clean; a bare pointer use keeps the slot
  //   v = &x; v[0]                ->  x            (whole-value deref)
  //   v = &x; v[0] = e            ->  x = e        (whole-value store)
  //   p = &a.x; p[i]              ->  a.x[i]       (re-root stepped use)
  //   p = &s.arr[i]; p[0]         ->  s.arr[i]     (struct array field element)
  //   r = &a[i]; s = &r[j]; s[k]  ->  a[i][j][k]   (multi-dim local array chain)

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
      val derefFolded = f.modifyAll[p.Expr] {
        case e @ p.Expr.Index(p.Term.Select(n, Nil, _), idx, comp) if isZeroConst(idx) =>
          wholeValue(n.symbol, comp).fold(e: p.Expr)(p.Expr.Alias(_))
        case e => e
      }
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
      stmtRw: (Set[String], p.Stmt) => List[p.Stmt],
      termRw: (Set[String], p.Term) => p.Term = (_, t) => t
  ): (p.Function, Int) = {
    val drop = if (syms.isEmpty) Set.empty[String] else foldable(f, syms)(seeds)
    if (drop.isEmpty) (f, 0)
    else {
      val rewritten = f.modifyAll[p.Expr](exprRw(drop, _)).modifyAll[p.Term](termRw(drop, _))
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

  // re-root `&b.fld[i]` (element address of an array-typed struct field) to the access chain `b.fld[i]`
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
      } ::: f.collectAll[p.Expr].collect {
        case p.Expr.Index(p.Term.Select(n, Nil, _), idx, _) if syms(n.symbol) && isZeroConst(idx) => n.symbol
      } ::: f.collectAll[p.Term].collect {
        case p.Term.Select(n, steps, _) if syms(n.symbol) && steps.nonEmpty => n.symbol
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
        },
      (drop, t) =>
        t match {
          case p.Term.Select(n, steps, tp) if steps.nonEmpty && drop(n.symbol) =>
            val (root, path) = chain(n.symbol); p.Term.Select(root, path ::: steps, tp)
          case _ => t
        }
    )
  }

  private def canonicaliseFn(f: p.Function): (p.Function, Int) = {
    val reassigned    = Provenance.reassignedIn(f)
    val (f1, aliased) = resolveFieldAliases(f, reassigned)
    val (f2, arrFlds) = resolveArrayFieldIndex(f1, reassigned)
    val (f3, locals)  = resolveLocalArray(f2, reassigned)
    (f3, aliased + arrFlds + locals)
  }

}
