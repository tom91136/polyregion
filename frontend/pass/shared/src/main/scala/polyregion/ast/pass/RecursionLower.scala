package polyregion.ast.pass

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// lowers self- and mutually-recursive offload functions to an iterative explicit-stack driver so they run on
// backends with no call stack (SPIR-V/OpenCL/Metal); must run before FnInline, which would otherwise try to
// flatten a self-Invoke that never converges. `f` keeps its signature - its body becomes a `while(sp>0)` loop
// over a private fixed-depth `Frame[maxDepth]` stack. the body is already in ANF (each call is its own
// `Var(t, Invoke)`), so it splits into pc-numbered blocks at every recursive call; a var live across a block
// boundary (and every param) is promoted to a frame field, the rest stay loop-locals. each block ends in a
// branch (set pc), a recursive call (set resume pc, push a callee frame) or a return (write the return
// register, pop). a purely tail-recursive `f` skips the frame stack for a plain loop over mutable param
// locals; a mutually-recursive cluster is a strongly-connected component (SCC) of the call graph
// examples:
//   fib(n) = fib(n-1) + fib(n-2)  ->  Frame{pc,n,t0}; push(0,n); while(sp>0){ dispatch }; ret  // explicit stack
//   fact(n,a) = fact(n-1, a*n)    ->  while(more){ n,a := n-1, a*n }; ret a                     // tail, no frame
//   f called inside a for/while   ->  the loop is lowered into pc-blocks (header CondBr, body loops back)
//   is_even(n) / is_odd(n)        ->  merged into one tag-dispatched __scc(tag, ..); members -> thin wrappers
// edge cases:
//   stack would exceed maxDepth       ->  guard raises assert(RecursionLimit)
//   mutual recursion (call-graph SCC) ->  merged into __scc, then lowered as one self-recursive driver
//   mutual w/ mixed returns or sigs   ->  rejected: members must share one return type and parameter signature
//   mutual with captures              ->  rejected: a merged frame cannot carry per-member captures
final case class RecursionLower(maxDepth: Int = 1024) extends ProgramPass derives PassArgCodec {

  private val i32                           = p.Type.IntS32
  private val bool                          = p.Type.Bool1
  private val ctr                           = new AtomicLong(0L)
  private def fresh(prefix: String): String = s"$prefix${ctr.incrementAndGet()}"

  private def i32c(v: Int): p.Term                = p.Term.IntS32Const(v)
  private def alias(t: p.Term): p.Expr            = p.Expr.Alias(t)
  private def add1(n: p.Named): p.Expr            = p.Expr.IntrOp(p.Intr.Add(sel(n), i32c(1), i32))
  private def sub1(n: p.Named): p.Expr            = p.Expr.IntrOp(p.Intr.Sub(sel(n), i32c(1), i32))
  private def assignVar(n: p.Named, e: p.Expr)    = p.Stmt.Var(n, Some(e), isMutable = true)
  private def setVar(n: p.Named, e: p.Expr)       = p.Stmt.Mut(sel(n), e)
  private def setSel(s: p.Term.Select, e: p.Expr) = p.Stmt.Mut(s, e)

  // block terminators
  private enum Terminator {
    case Goto(target: Int)
    case CondBr(cond: p.Term, t: Int, f: Int)
    case Call(args: List[p.Term], resume: Int)
    case Ret(value: p.Expr)
    case Pop
  }
  private case class Block(stmts: List[p.Stmt], end: Terminator)

  override def apply(program: p.Program, log: Log): p.Program = {
    val byName = program.functions.map(f => f.name -> f).toMap
    val edges = program.functions
      .map(f =>
        f.name -> f.body.collectWhere[p.Expr] { case i: p.Expr.Invoke if byName.contains(i.name) => i.name }.distinct
      )
      .toMap
    // a recursive cluster is an SCC of size > 1 (mutual) or a singleton with a self-edge (direct self-call)
    val recComps = stronglyConnected(program.functions.map(_.name), edges)
      .filter(c => c.sizeIs > 1 || edges.getOrElse(c.head, Nil).contains(c.head))
    if (recComps.isEmpty) program
    else {
      val replace = mutable.Map.empty[p.Sym, p.Function]
      val extra   = mutable.ListBuffer.empty[p.Function]
      val frames  = mutable.ListBuffer.empty[p.StructDef]
      recComps.foreach { comp =>
        if (comp.sizeIs == 1) {
          val (g, frame) = transform(byName(comp.head))
          replace(comp.head) = g
          frame.foreach(frames += _)
        } else {
          // mutual recursion: merge the cluster into one tag-dispatched self-recursive driver, then reuse the
          // single-function transform (TCO when the merge is tail, generic stack otherwise); members become
          // thin wrappers that seed the driver with their tag and arguments
          val members = comp.map(byName)
          if (members.map(_.rtn).distinct.sizeIs != 1)
            sys.error(s"RecursionLower: mutual recursion with mixed return types (${comp.map(_.repr).mkString(", ")})")
          if (members.map(_.args.map(_.named.tpe)).distinct.sizeIs != 1)
            sys.error(
              s"RecursionLower: mutual recursion with differing parameter signatures (${comp.map(_.repr).mkString(", ")})"
            )
          if (members.exists(m => m.receiver.nonEmpty || m.moduleCaptures.nonEmpty || m.termCaptures.nonEmpty))
            sys.error(s"RecursionLower: mutual recursion with captures (${comp.map(_.repr).mkString(", ")})")
          val (merged, wrappers) = buildMerged(members)
          val (mergedT, frame)   = transform(merged)
          extra += mergedT
          frame.foreach(frames += _)
          wrappers.foreach(w => replace(w.name) = w)
        }
      }
      log.info(s"lowered ${recComps.size} recursive component(s)")
      program.copy(
        functions = program.functions.map(f => replace.getOrElse(f.name, f)) ++ extra.toList,
        defs = program.defs ++ frames.toList
      )
    }
  }

  // Tarjan's SCCs of the call graph
  private def stronglyConnected(nodes: List[p.Sym], edges: Map[p.Sym, List[p.Sym]]): List[List[p.Sym]] = {
    var counter = 0
    val index   = mutable.Map.empty[p.Sym, Int]
    val low     = mutable.Map.empty[p.Sym, Int]
    val onStack = mutable.Set.empty[p.Sym]
    val stack   = mutable.Stack.empty[p.Sym]
    val out     = mutable.ListBuffer.empty[List[p.Sym]]
    def connect(v: p.Sym): Unit = {
      index(v) = counter; low(v) = counter; counter += 1
      stack.push(v); onStack += v
      edges.getOrElse(v, Nil).foreach { w =>
        if (!index.contains(w)) { connect(w); low(v) = math.min(low(v), low(w)) }
        else if (onStack(w)) low(v) = math.min(low(v), index(w))
      }
      if (low(v) == index(v)) {
        val comp = mutable.ListBuffer.empty[p.Sym]
        var w    = stack.pop(); onStack -= w; comp += w
        while (w != v) { w = stack.pop(); onStack -= w; comp += w }
        out += comp.toList
      }
    }
    nodes.foreach(v => if (!index.contains(v)) connect(v))
    out.toList
  }

  // merge a mutually-recursive cluster into one self-recursive `__scc(tag, <all members' params>)` whose body
  // dispatches on tag to each member (params/locals alpha-renamed per member), with every intra-cluster call
  // rewritten to `__scc(callee-tag, args-in-callee-slot)`. each member becomes `m(args) = __scc(m-tag, args)`.
  private def buildMerged(members: List[p.Function]): (p.Function, List[p.Function]) = {
    val sccName = p.Sym(List(fresh("_scc")))
    val idxOf   = members.zipWithIndex.map((m, i) => m.name -> i).toMap
    val rtn     = members.head.rtn
    val tagN    = p.Named(fresh("_scc_tag"), i32)
    // members share one parameter signature (enforced in apply), so one shared param slot per position; the
    // tag selects which member's body runs. sharing keeps the driver small (tag + the common params)
    val sharedParams              = members.head.args.zipWithIndex.map((a, j) => p.Named(s"_scc_p$j", a.named.tpe))
    val mergedParams: List[p.Arg] = p.Arg(tagN) :: sharedParams.map(p.Arg(_))

    def mergedArgs(j: Int, callArgs: List[p.Term]): List[p.Term] = i32c(j) :: callArgs

    def memberBody(m: p.Function, i: Int): List[p.Stmt] = {
      val paramRename             = m.args.zipWithIndex.map((a, j) => a.named.symbol -> sharedParams(j)).toMap
      val prefix                  = s"_m${i}_"
      def rn(n: p.Named): p.Named = paramRename.getOrElse(n.symbol, p.Named(prefix + n.symbol, n.tpe))
      m.body
        .modifyAll[p.Term] { case p.Term.Select(r, s, t) => p.Term.Select(rn(r), s, t); case x => x }
        .modifyAll[p.Stmt] {
          case p.Stmt.Var(n, e, mut)               => p.Stmt.Var(rn(n), e, mut)
          case p.Stmt.ForRange(ind, lb, ub, st, b) => p.Stmt.ForRange(rn(ind), lb, ub, st, b)
          case x                                   => x
        }
        .modifyAll[p.Expr] {
          case ivk: p.Expr.Invoke if idxOf.contains(ivk.name) =>
            p.Expr.Invoke(sccName, Nil, None, mergedArgs(idxOf(ivk.name), ivk.args), rtn)
          case x => x
        }
    }

    val body = members.zipWithIndex.foldRight(List.empty[p.Stmt]) { case ((m, i), elseBr) =>
      if (i == members.size - 1) memberBody(m, i)
      else {
        val isName = p.Named(fresh("_scc_is"), bool)
        List(
          assignVar(isName, p.Expr.IntrOp(p.Intr.LogicEq(sel(tagN), i32c(i)))),
          p.Stmt.Cond(sel(isName), memberBody(m, i), elseBr)
        )
      }
    }
    val merged = p.Function(
      sccName,
      Nil,
      None,
      mergedParams,
      Nil,
      Nil,
      rtn,
      body,
      p.Function.Visibility.Internal,
      p.Function.FpMode.Relaxed,
      isEntry = false,
      p.Function.Affinity.Offload
    )
    // bind every merged argument to a mutable temp before the call: if the driver is tail-recursive it
    // reassigns its params, and FnInline substitutes a param with the literal call argument - a constant
    // (tag, unused slots) would then become the target of a Mut. a temp is an lvalue, so the Mut is valid
    val wrappers = members.zipWithIndex.map { (m, i) =>
      val args  = mergedArgs(i, m.args.map(a => sel(a.named)))
      val temps = args.map(a => p.Named(fresh("_scc_w"), a.tpe))
      val binds = temps.zip(args).map((t, a) => p.Stmt.Var(t, Some(alias(a)), isMutable = true))
      m.copy(body = binds :+ p.Stmt.Return(p.Expr.Invoke(sccName, Nil, None, temps.map(sel), rtn)))
    }
    (merged, wrappers)
  }

  private def containsSelfCall(f: p.Function, stmts: List[p.Stmt]): Boolean =
    stmts.collectWhere[p.Expr] { case ivk: p.Expr.Invoke if ivk.name == f.name => () }.nonEmpty

  // a function is tail-recursive when every self-call is a direct `return f(...)` and no return sits inside
  // a loop (the simple loop rewrite escapes only via the maintained while-condition, not from a nested loop)
  private def isTailRecursive(f: p.Function): Boolean = {
    val allSelf  = f.body.collectWhere[p.Expr] { case i: p.Expr.Invoke if i.name == f.name => i }
    val tailSelf = f.body.collectWhere[p.Stmt] { case p.Stmt.Return(i: p.Expr.Invoke) if i.name == f.name => i }
    allSelf.nonEmpty && allSelf.sizeIs == tailSelf.size && !returnInLoop(f.body)
  }

  private def returnInLoop(stmts: List[p.Stmt]): Boolean = stmts.exists {
    case p.Stmt.While(_, b)             => hasReturn(b)
    case p.Stmt.ForRange(_, _, _, _, b) => hasReturn(b)
    case p.Stmt.Cond(_, t, e)           => returnInLoop(t) || returnInLoop(e)
    case p.Stmt.Annotated(i, _, _)      => returnInLoop(List(i))
    case _                              => false
  }
  private def hasReturn(stmts: List[p.Stmt]): Boolean =
    stmts.collectWhere[p.Stmt] { case _: p.Stmt.Return => () }.nonEmpty

  private def transform(f: p.Function): (p.Function, Option[p.StructDef]) =
    if (isTailRecursive(f)) (transformTail(f), None)
    else { val (g, frame) = transformGeneric(f); (g, Some(frame)) }

  // tail-recursive -> a plain loop: the params are copied into mutable locals once at the top (so the
  // incoming params are read exactly once, like the generic path - FnInline then substitutes those single
  // reads even when the caller is a synthesised wrapper passing constants); the loop mutates the locals.
  // every `return f(args)` snapshots its args and reassigns the locals; every other return stops the loop
  // with the result. no frame stack, no #pc dispatch - a test confirms TCO by the absence of a frame struct
  private def transformTail(f: p.Function): p.Function = {
    val params  = f.args.map(_.named)
    val locals  = params.map(pp => p.Named(fresh("#tco_p"), pp.tpe))
    val toLocal = params.map(_.symbol).zip(locals).toMap
    val resultN = p.Named(fresh("#tco_result"), f.rtn)
    val contN   = p.Named(fresh("#tco_continue"), bool)

    def relocal(t: p.Term): p.Term = t match {
      case p.Term.Select(root, steps, tpe) if toLocal.contains(root.symbol) =>
        p.Term.Select(toLocal(root.symbol), steps, tpe)
      case x => x
    }
    val body = f.body.map(_.modifyAll[p.Term](relocal))

    def repl(s: p.Stmt): List[p.Stmt] = s match {
      case p.Stmt.Return(i: p.Expr.Invoke) if i.name == f.name =>
        require(i.args.sizeIs == locals.size, s"RecursionLower: tail arity mismatch in ${f.name.repr}")
        val temps = i.args.map(a => p.Named(fresh("#tco_arg"), a.tpe))
        temps.zip(i.args).map((t, a) => p.Stmt.Var(t, Some(alias(a)), isMutable = false)) ++
          locals.zip(temps).map((l, t) => p.Stmt.Mut(sel(l), alias(sel(t))))
      case p.Stmt.Return(e) =>
        List(p.Stmt.Mut(sel(resultN), e), p.Stmt.Mut(sel(contN), alias(p.Term.Bool1Const(false))))
      case p.Stmt.Cond(c, t, e)       => List(p.Stmt.Cond(c, t.flatMap(repl), e.flatMap(repl)))
      case p.Stmt.Annotated(i, p0, c) => repl(i).map(p.Stmt.Annotated(_, p0, c))
      case other                      => List(other)
    }

    val newBody =
      locals.zip(params).map((l, pp) => assignVar(l, alias(sel(pp)))) ++
        List(
          assignVar(resultN, defaultExpr(f.rtn)),
          assignVar(contN, alias(p.Term.Bool1Const(true))),
          p.Stmt.While(sel(contN), body.flatMap(repl)),
          p.Stmt.Return(alias(sel(resultN)))
        )
    f.copy(body = newBody)
  }

  private def transformGeneric(f: p.Function): (p.Function, p.StructDef) = {
    val retName                             = p.Named(fresh("#rec_ret"), f.rtn)
    val blocks                              = mutable.ArrayBuffer.empty[(Int, Block)]
    var nextId                              = 0
    def freshId(): Int                      = { val i = nextId; nextId += 1; i }
    def emit(id: Int, b: Block): Unit       = blocks += (id -> b)
    def isSelf(ivk: p.Expr.Invoke): Boolean = ivk.name == f.name

    // pop sentinel: a frame that fell off the end (void fall-through) just pops
    val popId = freshId()

    // lower a statement list, continuing to `next` on fall-through; returns the entry block id
    def lower(stmts: List[p.Stmt], next: Int): Int = {
      val entry = freshId()
      def go(rem: List[p.Stmt], acc: List[p.Stmt], curId: Int): Unit = rem match {
        case Nil                                   => emit(curId, Block(acc.reverse, Terminator.Goto(next)))
        case p.Stmt.Annotated(inner, _, _) :: rest => go(inner :: rest, acc, curId)
        case p.Stmt.Var(t, Some(ivk: p.Expr.Invoke), _) :: rest if isSelf(ivk) =>
          val resume = lower(p.Stmt.Var(t, Some(alias(sel(retName))), isMutable = true) :: rest, next)
          emit(curId, Block(acc.reverse, Terminator.Call(ivk.args, resume)))
        case p.Stmt.Mut(x, ivk: p.Expr.Invoke) :: rest if isSelf(ivk) =>
          val resume = lower(p.Stmt.Mut(x, alias(sel(retName))) :: rest, next)
          emit(curId, Block(acc.reverse, Terminator.Call(ivk.args, resume)))
        case p.Stmt.Return(ivk: p.Expr.Invoke) :: _ if isSelf(ivk) =>
          val t = p.Named(fresh("#rec_tail"), f.rtn)
          go(p.Stmt.Var(t, Some(ivk), isMutable = false) :: p.Stmt.Return(alias(sel(t))) :: Nil, acc, curId)
        case p.Stmt.Return(e) :: _ => emit(curId, Block(acc.reverse, Terminator.Ret(e)))
        case p.Stmt.Cond(c, tBr, fBr) :: rest =>
          val contId = if (rest.isEmpty) next else lower(rest, next)
          val tId    = lower(tBr, contId)
          val fId    = lower(fBr, contId)
          emit(curId, Block(acc.reverse, Terminator.CondBr(c, tId, fId)))
        // a loop with a recursive call inside: lower it into a header block (eval cond, branch to body or
        // exit) with the body looping back; the loop state crosses blocks so it is promoted to frame fields
        case p.Stmt.While(cond, body) :: rest if containsSelfCall(f, body) =>
          val contId    = if (rest.isEmpty) next else lower(rest, next)
          val header    = freshId()
          val bodyEntry = lower(body, header)
          emit(header, Block(Nil, Terminator.CondBr(cond, bodyEntry, contId)))
          emit(curId, Block(acc.reverse, Terminator.Goto(header)))
        case p.Stmt.ForRange(ind, lb, ub, step, body) :: rest if containsSelfCall(f, body) =>
          // desugar to a while: capture ub/step once, maintain the bound condition, increment after the body
          val ubV  = p.Named(fresh("#rec_ub"), ub.tpe)
          val stV  = p.Named(fresh("#rec_st"), step.tpe)
          val cmp  = p.Named(fresh("#rec_for"), bool)
          val cmpE = p.Expr.IntrOp(p.Intr.LogicLt(sel(ind), sel(ubV)))
          val desugared = List(
            p.Stmt.Var(ubV, Some(alias(ub)), isMutable = false),
            p.Stmt.Var(stV, Some(alias(step)), isMutable = false),
            p.Stmt.Var(ind, Some(alias(lb)), isMutable = true),
            p.Stmt.Var(cmp, Some(cmpE), isMutable = true),
            p.Stmt.While(
              sel(cmp),
              body ++ List(
                p.Stmt.Mut(sel(ind), p.Expr.IntrOp(p.Intr.Add(sel(ind), sel(stV), ind.tpe))),
                p.Stmt.Mut(sel(cmp), cmpE)
              )
            )
          )
          go(desugared ::: rest, acc, curId)
        case s :: rest => go(rest, s :: acc, curId)
      }
      go(stmts, Nil, entry)
      entry
    }

    val bodyEntry = lower(f.body, popId)
    emit(popId, Block(Nil, Terminator.Pop))

    val ids = blocks.map(_._1).toList

    // promotion: a var touching more than one block (or any parameter) must live in the frame
    val defBlk = mutable.Map.empty[String, mutable.Set[Int]]
    val useBlk = mutable.Map.empty[String, mutable.Set[Int]]
    blocks.foreach { case (id, b) =>
      def bump(s: p.Term.Select): Unit = useBlk.getOrElseUpdate(s.root.symbol, mutable.Set.empty) += id
      def usesIn(e: p.Expr): Unit      = e.collectWhere[p.Term] { case s: p.Term.Select => s }.foreach(bump)
      b.stmts.foreach {
        case p.Stmt.Var(n, e, _) => defBlk.getOrElseUpdate(n.symbol, mutable.Set.empty) += id; e.foreach(usesIn)
        case other               => other.collectWhere[p.Term] { case s: p.Term.Select => s }.foreach(bump)
      }
      b.end match {
        case Terminator.CondBr(c, _, _) => c.collectWhere[p.Term] { case s: p.Term.Select => s }.foreach(bump)
        case Terminator.Call(args, _) =>
          args.foreach(_.collectWhere[p.Term] { case s: p.Term.Select => s }.foreach(bump))
        case Terminator.Ret(e) => usesIn(e)
        case _                 => ()
      }
    }

    val params    = f.args.map(_.named)
    val paramSyms = params.map(_.symbol).toSet
    // gather every declared/used Named so promoted ones carry their type
    val declared = blocks.flatMap(_._2.stmts).collect { case p.Stmt.Var(n, _, _) => n.symbol -> n }.toMap
    val namedOf  = declared ++ params.map(n => n.symbol -> n).toMap
    val touched  = defBlk.keySet ++ useBlk.keySet
    // params always live in the frame; everything else only if it crosses a block boundary
    val promotedSym = paramSyms ++ touched.filter { s =>
      (defBlk.getOrElse(s, mutable.Set.empty) ++ useBlk.getOrElse(s, mutable.Set.empty)).size > 1
    }
    // a frame field per promoted name; deriving pset from the resolved members keeps the rewrite and the
    // struct layout in lockstep and excludes synthetic driver vars (ret/sp/ci) that never resolve to a field
    val promoted = promotedSym.toList.flatMap(namedOf.get)
    val pset     = promoted.map(_.symbol).toSet

    val frameSym = p.Sym(List(s"#RecFrame_${f.name.fqn.mkString("_")}"))
    val pcName   = "#pc"
    val members  = p.Named(pcName, i32) :: promoted.sortBy(_.symbol)
    val frameDef = p.StructDef(frameSym, Nil, members, Nil)
    val frameTpe = p.Type.Struct(frameSym, Nil)
    val stackTpe = p.Type.Arr(frameTpe, maxDepth, p.Type.Space.Private)

    val stackName = p.Named(fresh("#rec_stack"), stackTpe)
    val spName    = p.Named(fresh("#rec_sp"), i32)
    val ciName    = p.Named(fresh("#rec_ci"), i32)
    val pcLocal   = p.Named(fresh("#rec_pc"), i32)

    def frameField(idx: p.Named, name: String, tpe: p.Type): p.Term.Select =
      p.Term.Select(stackName, List(p.PathStep.IndexDyn(sel(idx)), p.PathStep.Field(name)), tpe)
    def curField(name: String, tpe: p.Type): p.Term.Select = frameField(ciName, name, tpe)
    def newField(name: String, tpe: p.Type): p.Term.Select = frameField(spName, name, tpe)

    // rewrite reads of promoted vars to the current frame slot
    def termRw(t: p.Term): p.Term = t match {
      case p.Term.Select(root, steps, tpe) if pset.contains(root.symbol) =>
        p.Term.Select(stackName, p.PathStep.IndexDyn(sel(ciName)) :: p.PathStep.Field(root.symbol) :: steps, tpe)
      case other => other
    }
    def exprRw(e: p.Expr): p.Expr = e.modifyAll[p.Term](termRw)
    def rewriteStmts(stmts: List[p.Stmt]): List[p.Stmt] =
      stmts
        .map {
          case p.Stmt.Var(n, Some(e), _) if pset.contains(n.symbol) => setSel(curField(n.symbol, n.tpe), e)
          case s                                                    => s
        }
        .map(_.modifyAll[p.Term](termRw))

    def overflowGuard: List[p.Stmt] = {
      val g = p.Named(fresh("#rec_ovf"), bool)
      // StructuredExit lowers this `'assert` to the flag + error-buffer drain; the RecursionLimit code lets the
      // host tell a stack overflow apart from a user assert
      val trap = p.Stmt.Var(
        p.Named(fresh("#rec_trap"), p.Type.Unit0),
        Some(
          p.Expr.SpecOp(
            p.Spec.Assert(
              p.Term.IntU32Const(p.Enums.AssertCode.RecursionLimit.value),
              p.Term.StringConst(s"recursion depth exceeded $maxDepth")
            )
          )
        )
      )
      List(
        assignVar(g, p.Expr.IntrOp(p.Intr.LogicGte(sel(spName), i32c(maxDepth)))),
        p.Stmt.Cond(sel(g), List(trap), Nil)
      )
    }

    def pushFrame(args: List[p.Term]): List[p.Stmt] = {
      require(args.size == params.size, s"RecursionLower: arity mismatch in ${f.name.repr}")
      overflowGuard ++
        (setSel(newField(pcName, i32), alias(i32c(bodyEntry))) ::
          params.zip(args).map { case (pp, a) => setSel(newField(pp.symbol, pp.tpe), alias(termRw(a))) }) :+
        setVar(spName, add1(spName))
    }

    // initial push seeds the entry pc and copies the real incoming parameters into the first frame
    def pushFrameInitial(entryPc: Int): List[p.Stmt] =
      overflowGuard ++
        (setSel(newField(pcName, i32), alias(i32c(entryPc))) ::
          params.map(pp => setSel(newField(pp.symbol, pp.tpe), alias(sel(pp))))) :+
        setVar(spName, add1(spName))

    def setPc(idx: p.Named, target: Int): p.Stmt = setSel(frameField(idx, pcName, i32), alias(i32c(target)))

    def terminatorStmts(t: Terminator): List[p.Stmt] = t match {
      case Terminator.Goto(target)    => List(setPc(ciName, target))
      case Terminator.CondBr(c, a, b) => List(p.Stmt.Cond(termRw(c), List(setPc(ciName, a)), List(setPc(ciName, b))))
      case Terminator.Call(args, resume) =>
        setPc(ciName, resume) :: pushFrame(args)
      case Terminator.Ret(value) => List(setVar(retName, exprRw(value)), setVar(spName, sub1(spName)))
      case Terminator.Pop        => List(setVar(spName, sub1(spName)))
    }

    // dispatch chain: if (pc==0) {b0} else if (pc==1) {b1} ... else {bLast}
    val byId = blocks.toMap
    val dispatch = ids.sorted.foldRight(List.empty[p.Stmt]) { (id, elseBr) =>
      val b      = byId(id)
      val body   = rewriteStmts(b.stmts) ++ terminatorStmts(b.end)
      val isName = p.Named(fresh("#rec_is"), bool)
      List(
        assignVar(isName, p.Expr.IntrOp(p.Intr.LogicEq(sel(pcLocal), i32c(id)))),
        p.Stmt.Cond(sel(isName), body, elseBr)
      )
    }

    // a maintained loop-condition var (recomputed at the end of each iteration) keeps the driver a plain
    // `while(cond)` instead of `while(true){ if(empty) break }`; a structured Break inside a Cond branch
    // double-terminates the block in the LLVM backend
    val moreName = p.Named(fresh("#rec_more"), bool)
    val nonEmpty = p.Expr.IntrOp(p.Intr.LogicGt(sel(spName), i32c(0)))
    val loopBody = List(
      assignVar(ciName, sub1(spName)),
      assignVar(pcLocal, alias(curField(pcName, i32)))
    ) ++ dispatch ++ List(setVar(moreName, nonEmpty))

    val newBody =
      List(
        p.Stmt.Var(stackName, None, isMutable = true),
        assignVar(spName, alias(i32c(0))),
        assignVar(retName, defaultExpr(f.rtn))
      ) ++
        pushFrameInitial(bodyEntry) ++
        List(
          assignVar(moreName, nonEmpty),
          p.Stmt.While(sel(moreName), loopBody),
          p.Stmt.Return(alias(sel(retName)))
        )

    (f.copy(body = newBody), frameDef)
  }
}
