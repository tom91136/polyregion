package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

import scala.collection.mutable

final case class EscapeAnalysis(refuse: Boolean = false) extends ProgramPass derives PassArgCodec {

  import EscapeAnalysis.*

  override def apply(program: p.Program, log: Log): p.Program = {
    val defs      = program.defs.map(d => d.name -> d).toMap
    val functions = program.entry.toList ::: program.functions
    val summaries = summarise(functions)
    val analysed  = functions.map(f => f -> Analysis(f, summaries).solve())

    val refused = analysed.flatMap { case (f, analysis) =>
      if (refuse && f.affinity == p.Function.Affinity.Offload) analysis.escapingSites.map(diagnostic(f, _)) else Nil
    }
    if (refused.nonEmpty)
      throw IllegalStateException(
        s"offload code may not let an allocation escape:\n${refused.map("  " + _).mkString("\n")}"
      )

    val rewritten = analysed.map { case (f, analysis) => rewrite(f, analysis, defs) }
    log.info(s"analysed ${functions.size} function(s)")
    if (program.entry.isDefined) program.copy(entry = rewritten.headOption, functions = rewritten.drop(1))
    else program.copy(functions = rewritten)
  }
}

object EscapeAnalysis {

  private final case class Signature(name: p.Sym, params: List[p.Type])
  private type Summaries = Map[Signature, Map[Int, String]]

  private sealed trait Obj
  private final case class Site(symbol: String) extends Obj
  private final case class Param(index: Int)    extends Obj
  private case object Top                       extends Obj

  private final case class SiteInfo(
      named: p.Named,
      comp: p.Type,
      size: p.Term,
      pos: Option[p.SourcePosition]
  ) {
    def display: String = named.origin.source.getOrElse(named.symbol)
  }

  private def flatParams(f: p.Function): List[(p.Named, String)] =
    f.receiver.toList.map(_.named -> "receiver") :::
      f.moduleCaptures.map(_.named -> "module capture") :::
      f.termCaptures.map(_.named -> "capture") :::
      f.args.map(_.named -> "argument")

  private def mayHoldPointer(t: p.Type): Boolean = t match {
    case p.Type.Ptr(_, _)       => true
    case p.Type.Struct(_, _)    => true
    case p.Type.Arr(comp, _, _) => mayHoldPointer(comp)
    case p.Type.Var(_, _)       => true
    case _                      => false
  }

  private def isArr(t: p.Type): Boolean = t match {
    case p.Type.Arr(comp = _) => true
    case _                    => false
  }

  private def isPtr(t: p.Type): Boolean = t match {
    case p.Type.Ptr(_, _) => true
    case _                => false
  }

  private def constIndex(t: p.Term): Option[Long] = t match {
    case p.Term.IntS8Const(v)            => Some(v.toLong)
    case p.Term.IntS16Const(v)           => Some(v.toLong)
    case p.Term.IntS32Const(v)           => Some(v.toLong)
    case p.Term.IntS64Const(v)           => Some(v)
    case p.Term.IntU8Const(v)            => Some(v.toLong & 0xff)
    case p.Term.IntU16Const(v)           => Some(v.toLong)
    case p.Term.IntU32Const(v)           => Some(v.toLong & 0xffffffffL)
    case p.Term.IntU64Const(v) if v >= 0 => Some(v)
    case _                               => None
  }

  private def isHeapAlloc(a: p.Expr.Alloc): Boolean =
    a.space == p.Type.Space.Global && a.region == p.Region.Opaque

  private def loadsOf(steps: List[p.PathStep]): Int = steps.count {
    case p.PathStep.Field(_) => false
    case _                   => true
  }

  private final class Analysis(f: p.Function, summaries: Summaries) {

    private val vals    = mutable.Map.empty[String, Set[Obj]].withDefaultValue(Set.empty)
    private val mem     = mutable.Map.empty[Obj, Set[Obj]].withDefaultValue(Set.empty)
    private val sites   = mutable.LinkedHashMap.empty[String, SiteInfo]
    private val reasons = mutable.LinkedHashMap.empty[Obj, String]
    private val params  = flatParams(f)
    private var changed = false

    private val anonymous = java.util.IdentityHashMap[p.Expr.Alloc, p.Named]()

    params.zipWithIndex.foreach { case ((named, _), i) =>
      if (mayHoldPointer(named.tpe)) vals(named.symbol) = Set(Param(i))
    }

    def escaped(site: String): Boolean = reasons.contains(Site(site))

    def siteInfo: Map[String, SiteInfo] = sites.toMap

    def escapingSites: List[(SiteInfo, String)] = sites.toList.collect {
      case (symbol, info) if reasons.contains(Site(symbol)) => info -> reasons(Site(symbol))
    }

    def paramEscapes: Map[Int, String] = reasons.collect { case (Param(i), why) => i -> why }.toMap

    def solve(): Analysis = {
      changed = true
      while (changed) {
        changed = false
        block(f.body, None)
        close()
      }
      this
    }

    private def describe(o: Obj): String = o match {
      case Site(s)  => s"allocation `${sites.get(s).fold(s)(_.display)}`"
      case Param(i) => val (n, kind) = params(i); s"$kind `${n.symbol}`"
      case Top      => "memory of unknown origin"
    }

    private def addVal(symbol: String, objs: Set[Obj]): Unit = {
      val before = vals(symbol)
      val after  = before ++ objs
      if (after.size != before.size) { vals(symbol) = after; changed = true }
    }

    private def addMem(o: Obj, objs: Set[Obj]): Unit = {
      val before = mem(o)
      val after  = before ++ objs
      if (after.size != before.size) { mem(o) = after; changed = true }
    }

    private def escape(objs: Set[Obj], why: => String): Unit = objs.foreach {
      case Top => ()
      case o =>
        if (!reasons.contains(o)) { reasons(o) = why; changed = true }
    }

    private def close(): Unit =
      reasons.toList.foreach { case (o, _) => escape(mem(o), s"stored into escaping ${describe(o)}") }

    private def load(objs: Set[Obj]): Set[Obj] = objs.flatMap {
      case s: Site => mem(s)
      case _       => Set[Obj](Top)
    }

    private def reach(base: Set[Obj], depth: Int): Set[Obj] =
      (1 to depth)
        .foldLeft((base, base)) { case ((acc, frontier), _) =>
          val next = load(frontier)
          (acc ++ next, next)
        }
        ._1

    private def term(t: p.Term): Set[Obj] = t match {
      case p.Term.Select(root, steps, tpe) if mayHoldPointer(tpe) => reach(vals(root.symbol), loadsOf(steps))
      case _                                                      => Set.empty
    }

    private def store(lhs: p.Term.Select, value: Set[Obj], throughPointer: Boolean): Unit =
      if (value.nonEmpty) {
        val loads = loadsOf(lhs.steps)
        if (throughPointer) into(term(lhs), value)
        else if (loads == 0) addVal(lhs.root.symbol, value)
        else into(reach(vals(lhs.root.symbol), loads - 1), value)
      }

    private def into(targets: Set[Obj], value: Set[Obj]): Unit = targets.foreach {
      case s: Site => addMem(s, value)
      case o       => escape(value, s"stored into ${describe(o)}")
    }

    private def summary(name: p.Sym, actuals: List[p.Term]): Option[Map[Int, String]] = {
      val candidates  = summaries.keys.filter(k => k.name == name && k.params.size == actuals.size).toList
      val actualTypes = actuals.map(_.tpe)
      val exact       = candidates.filter(_.params == actualTypes)
      (candidates, exact) match {
        case (single :: Nil, _) => summaries.get(single)
        case (_, single :: Nil) => summaries.get(single)
        case _                  => None
      }
    }

    private def returnReason: String =
      if (f.convention == p.CallConvention.OffloadEntry) "returned from the offload entry"
      else s"returned from `${f.name.repr}`"

    private def expr(e: p.Expr, ctx: Option[p.Named], pos: Option[p.SourcePosition]): Set[Obj] = e match {
      case p.Expr.Alias(t) => term(t)
      case p.Expr.Cast(from, as) =>
        if (isPtr(from.tpe) && !isPtr(as)) { escape(term(from), "converted to an integer"); Set.empty }
        else if (!isPtr(from.tpe) && isPtr(as)) Set(Top)
        else term(from)
      case p.Expr.BitCast(from, as) =>
        if (isPtr(from.tpe) && !isPtr(as)) { escape(term(from), "converted to an integer"); Set.empty }
        else if (!isPtr(from.tpe) && isPtr(as)) Set(Top)
        else term(from)
      case p.Expr.Index(lhs, _, comp) =>
        if (mayHoldPointer(comp)) { val base = term(lhs); base ++ load(base) }
        else Set.empty
      case p.Expr.RefTo(lhs = lhs) => term(lhs) + Top
      case a @ p.Expr.Alloc(comp, size, _, _) =>
        if (isHeapAlloc(a)) {
          val named = ctx.getOrElse {
            anonymous.computeIfAbsent(
              a,
              _ => p.Named(s"<allocation ${anonymous.size}>", p.Type.Ptr(comp, p.Type.Space.Global))
            )
          }
          sites.getOrElseUpdate(named.symbol, SiteInfo(named, comp, size, pos.orElse(named.origin.pos)))
          Set(Site(named.symbol))
        } else Set(Top)
      case ivk: p.Expr.Invoke =>
        val actuals = ivk.receiver.toList ::: ivk.args
        val name    = ivk.calleeName
        summary(name, actuals) match {
          case Some(esc) =>
            actuals.zipWithIndex.foreach { case (a, i) =>
              esc.get(i).foreach(why => escape(term(a), s"passed to `${name.repr}`, which lets it escape: $why"))
            }
          case None => actuals.foreach(a => escape(term(a), s"passed to unresolved call `${name.repr}`"))
        }
        if (mayHoldPointer(ivk.rtn)) Set(Top) else Set.empty
      case p.Expr.ForeignCall(name, args, rtn) =>
        args.foreach(a => escape(term(a), s"passed to foreign call `$name`"))
        if (mayHoldPointer(rtn)) Set(Top) else Set.empty
      case p.Expr.IntrOp(op) =>
        op match {
          case _: (p.Intr.LogicEq | p.Intr.LogicNeq | p.Intr.LogicLt | p.Intr.LogicLte | p.Intr.LogicGt |
                p.Intr.LogicGte) =>
            ()
          case _ => op.terms.foreach(a => escape(term(a), "used in pointer arithmetic"))
        }
        Set.empty
      case p.Expr.MathOp(_) => Set.empty
      case p.Expr.SpecOp(op) =>
        op.terms.foreach(a => escape(term(a), s"passed to `${op.getClass.getSimpleName.stripSuffix("$")}`"))
        if (mayHoldPointer(op.tpe)) Set(Top) else Set.empty
      case p.Expr.OffsetOf(_, _) | p.Expr.SizeOf(_) => Set.empty
    }

    private def block(stmts: List[p.Stmt], pos: Option[p.SourcePosition]): Unit = stmts.foreach(stmt(_, pos))

    private def stmt(s: p.Stmt, pos: Option[p.SourcePosition]): Unit = s match {
      case p.Stmt.Var(n, Some(e), _) => addVal(n.symbol, expr(e, Some(n), pos))
      case p.Stmt.Var(_, None, _)    => ()
      case p.Stmt.Mut(lhs, e) =>
        val v = expr(e, Some(lhs.root), pos)
        if (lhs.steps.isEmpty) addVal(lhs.root.symbol, v) else store(lhs, v, throughPointer = false)
      case p.Stmt.Update(lhs, _, value) => store(lhs, term(value), throughPointer = isPtr(lhs.tpe))
      case p.Stmt.Return(e)             => escape(expr(e, None, pos), returnReason)
      case p.Stmt.Raise(value, _, cleanup) =>
        escape(term(value), "thrown")
        block(cleanup, pos)
      case p.Stmt.While(_, body)        => block(body, pos)
      case p.Stmt.ForRange(body = body) => block(body, pos)
      case p.Stmt.Cond(_, t, e)         => block(t, pos); block(e, pos)
      case p.Stmt.Try(body, handlers, fin) =>
        block(body, pos)
        handlers.foreach { h =>
          h.binder.foreach(b => if (mayHoldPointer(b.tpe)) addVal(b.symbol, Set(Top)))
          block(h.body, pos)
        }
        block(fin, pos)
      case p.Stmt.Annotated(inner, at, _)              => stmt(inner, at.orElse(pos))
      case p.Stmt.Break | p.Stmt.Cont | p.Stmt.Rethrow => ()
    }
  }

  private def summarise(functions: List[p.Function]): Summaries = {
    def key(f: p.Function)   = Signature(f.name, flatParams(f).map(_._1.tpe))
    var summaries: Summaries = functions.map(f => key(f) -> Map.empty[Int, String]).toMap
    var again                = true
    while (again) {
      again = false
      functions.foreach { f =>
        val found = Analysis(f, summaries).solve().paramEscapes
        val known = summaries(key(f))
        if (!found.keySet.subsetOf(known.keySet)) {
          summaries = summaries.updated(key(f), found ++ known)
          again = true
        }
      }
    }
    summaries
  }

  private def diagnostic(f: p.Function, site: (SiteInfo, String)): String = {
    val (info, why) = site
    val at          = info.pos.fold("")(pos => s" at ${pos.repr}")
    val size = info.size match {
      case t if constIndex(t).isDefined => constIndex(t).get.toString
      case _                            => "n"
    }
    s"allocation `${info.display}` (${info.comp.repr}[$size]) in `${f.name.repr}`$at escapes: $why"
  }

  private def mentions(s: p.Stmt, symbols: Set[String]): Boolean =
    s.collectWhere[p.Term] { case p.Term.Select(root, _, _) if symbols(root.symbol) => () }.nonEmpty

  private def symbolsOf(f: p.Function): Set[String] =
    (flatParams(f).map(_._1.symbol) :::
      f.collectAll[p.Stmt].flatMap {
        case p.Stmt.Var(n, _, _)                    => List(n.symbol)
        case p.Stmt.ForRange(induction = induction) => List(induction.symbol)
        case p.Stmt.Try(_, handlers, _)             => handlers.flatMap(_.binder.map(_.symbol))
        case _                                      => Nil
      }).toSet

  private def declCounts(f: p.Function): Map[String, Int] =
    (flatParams(f).map(_._1.symbol) :::
      f.collectAll[p.Stmt].flatMap {
        case p.Stmt.Var(n, _, _)                    => List(n.symbol)
        case p.Stmt.ForRange(induction = induction) => List(induction.symbol)
        case p.Stmt.Try(_, handlers, _)             => handlers.flatMap(_.binder.map(_.symbol))
        case _                                      => Nil
      }).groupMapReduce(identity)(_ => 1)(_ + _)

  private def uniqueName(taken: mutable.Set[String], base: String): String = {
    val name = Iterator.single(base).concat(Iterator.from(0).map(i => s"${base}_$i")).find(taken.add)
    name.get
  }

  private def rewrite(f: p.Function, analysis: Analysis, defs: Map[p.Sym, p.StructDef]): p.Function = {
    val taken = mutable.Set.from(symbolsOf(f))
    val a     = replaceAllocs(f, analysis, defs, taken)
    replaceStructs(a, defs, taken)
  }

  @SuppressWarnings(Array("bonsai:loop-boxing"))
  private def replaceAllocs(
      f: p.Function,
      analysis: Analysis,
      defs: Map[p.Sym, p.StructDef],
      taken: mutable.Set[String]
  ): p.Function = {
    val decls = declCounts(f)
    val terms = f.collectAll[p.Term]
    val occurrences =
      terms.collect { case p.Term.Select(root, _, _) => root.symbol }.groupMapReduce(identity)(_ => 1)(_ + _)
    val candidates = analysis.siteInfo.toList.flatMap { case (symbol, info) =>
      val declared = f
        .collectWhere[p.Stmt] { case p.Stmt.Var(n, Some(a: p.Expr.Alloc), _) if n.symbol == symbol => a }
        .nonEmpty
      val total = occurrences.getOrElse(symbol, 0)
      val reads = f.collectWhere[p.Expr] {
        case p.Expr.Index(p.Term.Select(root, Nil, _), idx, _) if root.symbol == symbol => constIndex(idx)
      }
      val writes = f.collectWhere[p.Stmt] {
        case p.Stmt.Update(p.Term.Select(root, Nil, _), idx, _) if root.symbol == symbol => constIndex(idx)
      }
      val indices = (reads ::: writes).flatten
      val limit   = constIndex(info.size)
      Option.when(
        declared && decls.get(symbol).contains(1) && !analysis.escaped(symbol) && limit.isDefined &&
          Zero.declare(defs, info.named.copy(tpe = info.comp), mutable = false).isDefined &&
          reads.size + writes.size == total && indices.size == total && indices.forall(i => i >= 0 && i < limit.get)
      )(symbol -> (info, indices.distinct.sorted, writes.flatten))
    }
    if (candidates.isEmpty) f
    else {
      val sites = candidates.map { case (symbol, (info, indices, written)) =>
        val named = indices.map(i => i -> info.named.copy(symbol = uniqueName(taken, s"${symbol}_$i"), tpe = info.comp))
        symbol -> (named.toMap, named.flatMap((i, c) => Zero.declare(defs, c, written.contains(i)).get))
      }.toMap
      val cells                                             = sites.view.mapValues(_._1).toMap
      def cell(root: p.Named, idx: p.Term): Option[p.Named] = cells.get(root.symbol).flatMap(_.get(constIndex(idx).get))
      val exprs = f.modifyAll[p.Expr] {
        case p.Expr.Index(p.Term.Select(root, Nil, _), idx, comp) if cell(root, idx).isDefined =>
          val c = cell(root, idx).get
          p.Expr.Alias(p.Term.Select(c, Nil, comp))
        case other => other
      }
      val body = Binders.mapStmts(exprs.body) {
        case p.Stmt.Var(n, Some(p.Expr.Alloc(comp = _)), _) if cells.contains(n.symbol) =>
          sites(n.symbol)._2
        case p.Stmt.Update(p.Term.Select(root, Nil, _), idx, value) if cell(root, idx).isDefined =>
          val c = cell(root, idx).get
          List(p.Stmt.Mut(p.Term.Select(c, Nil, c.tpe), p.Expr.Alias(value)))
        case other => List(other)
      }
      fuseInit(exprs.copy(body = body), cells.values.flatMap(_.values.map(_.symbol)).toSet)
    }
  }

  private def memberType(defs: Map[p.Sym, p.StructDef], tpe: p.Type, field: String): Option[p.Type] = tpe match {
    case p.Type.Struct(name, args) =>
      defs.get(name).filter(d => !d.isUnion && d.tpeVars.size == args.size).flatMap { d =>
        val env = d.tpeVars.map(_.name).zip(args).toMap
        d.members.find(_.symbol == field).map(m => Binders.substType(m.tpe, env))
      }
    case _ => None
  }

  private def structMembers(defs: Map[p.Sym, p.StructDef], tpe: p.Type): Option[List[(String, p.Type)]] =
    Zero.members(defs, tpe).filter(_.forall((_, t) => !isArr(t)))

  private def expandCopies(f: p.Function, defs: Map[p.Sym, p.StructDef]): p.Function = {
    val decls = declCounts(f)
    val terms = f.collectAll[p.Term]
    val addressed = f
      .collectWhere[p.Expr] { case p.Expr.RefTo(p.Term.Select(root, _, _), _, _, _, _) =>
        root.symbol
      }
      .toSet
    val plain = f.collectWhere[p.Stmt] { case p.Stmt.Var(n, None, _) => n.symbol }.toSet
    val copies = f.collectWhere[p.Stmt] {
      case p.Stmt.Var(y, Some(p.Expr.Alias(p.Term.Select(x, Nil, _))), _) if structMembers(defs, y.tpe).isDefined =>
        x.symbol
    }
    val wholeUses =
      terms.collect { case p.Term.Select(root, Nil, _) => root.symbol }.groupMapReduce(identity)(_ => 1)(_ + _)
    val copyCounts = copies.groupMapReduce(identity)(_ => 1)(_ + _)
    val sources = copies.distinct.filter { x =>
      plain(x) && decls.get(x).contains(1) && !addressed(x) && wholeUses.getOrElse(x, 0) == copyCounts(x)
    }.toSet
    if (sources.isEmpty) f
    else
      f.copy(body = Binders.mapStmts(f.body) {
        case p.Stmt.Var(y, Some(p.Expr.Alias(p.Term.Select(x, Nil, _))), m) if sources(x.symbol) =>
          val members = structMembers(defs, y.tpe).get
          p.Stmt.Var(y, None, m) :: members.map { case (name, tpe) =>
            p.Stmt.Mut(
              p.Term.Select(y, List(p.PathStep.Field(name)), tpe),
              p.Expr.Alias(p.Term.Select(x, List(p.PathStep.Field(name)), tpe))
            )
          }
        case other => List(other)
      })
  }

  private def replaceStructs(f0: p.Function, defs: Map[p.Sym, p.StructDef], taken: mutable.Set[String]): p.Function = {
    val f     = expandCopies(f0, defs)
    val decls = declCounts(f)
    val terms = f.collectAll[p.Term]
    val addressed = f
      .collectWhere[p.Expr] { case p.Expr.RefTo(p.Term.Select(root, _, _), _, _, _, _) =>
        root.symbol
      }
      .toSet
    val candidates = f.collectWhere[p.Stmt] { case p.Stmt.Var(n, None, _) => n }.distinct.flatMap { n =>
      val uses   = terms.collect { case p.Term.Select(root, steps, _) if root.symbol == n.symbol => steps }
      val fields = uses.collect { case p.PathStep.Field(name) :: _ => name }.distinct
      val types  = fields.map(name => name -> memberType(defs, n.tpe, name))
      Option.when(
        n.tpe.isInstanceOf[p.Type.Struct] && decls.get(n.symbol).contains(1) && !addressed(n.symbol) &&
          uses.forall(_.headOption.exists(_.isInstanceOf[p.PathStep.Field])) &&
          types.forall(_._2.exists {
            case p.Type.Arr(comp = _) => false
            case _                    => true
          })
      )(n -> types.map { case (name, tpe) => name -> tpe.get })
    }
    if (candidates.isEmpty) f
    else {
      val cells = candidates.map { case (n, fields) =>
        n.symbol -> fields.map { case (name, tpe) =>
          name -> n.copy(
            symbol = uniqueName(taken, s"${n.symbol}_${name.filter(c => c.isLetterOrDigit || c == '_')}"),
            tpe = tpe
          )
        }.toMap
      }.toMap
      val terms2 = f.modifyAll[p.Term] {
        case p.Term.Select(root, p.PathStep.Field(name) :: rest, tpe)
            if cells.get(root.symbol).exists(_.contains(name)) =>
          p.Term.Select(cells(root.symbol)(name), rest, tpe)
        case other => other
      }
      val body = Binders.mapStmts(terms2.body) {
        case p.Stmt.Var(n, None, mutable) if cells.contains(n.symbol) =>
          candidates.find(_._1.symbol == n.symbol).get._2.map { case (name, _) =>
            p.Stmt.Var(cells(n.symbol)(name), None, mutable)
          }
        case other => List(other)
      }
      val next = fuseInit(terms2.copy(body = body), cells.values.flatMap(_.values.map(_.symbol)).toSet)
      replaceStructs(next, defs, taken)
    }
  }

  private def fuseInit(f: p.Function, cells: Set[String]): p.Function = {
    val writes = f
      .collectAll[p.Stmt]
      .flatMap {
        case p.Stmt.Mut(p.Term.Select(root, _, _), _)       => List(root.symbol)
        case p.Stmt.Update(p.Term.Select(root, _, _), _, _) => List(root.symbol)
        case _                                              => Nil
      }
      .groupMapReduce(identity)(_ => 1)(_ + _)
    val addressed = f
      .collectWhere[p.Expr] { case p.Expr.RefTo(p.Term.Select(root, _, _), _, _, _, _) =>
        root.symbol
      }
      .toSet
    def fusable(c: String) = cells(c) && writes.get(c).contains(1) && !addressed(c)
    def isZero(tpe: p.Type, e: p.Expr) = e match {
      case p.Expr.Alias(t) => Zero.term(tpe).contains(t)
      case _               => false
    }

    def go(stmts: List[p.Stmt]): List[p.Stmt] = stmts match {
      case Nil => Nil
      case (decl @ p.Stmt.Var(n, init, _)) :: rest
          if fusable(n.symbol) && !isArr(n.tpe) && init.forall(isZero(n.tpe, _)) =>
        val (before, after) = rest.span(!mentions(_, Set(n.symbol)))
        after match {
          case p.Stmt.Mut(p.Term.Select(root, Nil, _), e) :: tail
              if root.symbol == n.symbol && !mentions(p.Stmt.Return(e), Set(n.symbol)) =>
            go(before ::: p.Stmt.Var(n, Some(e)) :: tail)
          case _ => decl :: go(rest)
        }
      case other :: rest => other :: go(rest)
    }
    f.copy(body = Binders.mapBlocks(f.body)(go))
  }
}
