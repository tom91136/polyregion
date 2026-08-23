package polyregion.ast.pass

import java.nio.charset.StandardCharsets

import scala.collection.mutable.ListBuffer

import polyregion.ast.Traversal.*
import polyregion.ast.{Log, PolyAST as p, *, given}

// lowers `assert(code, message)` to a structured exit: an asserting lane raises a flag, fences its
// remaining per-lane work, and drains out - but still runs every collective barrier (and the code that feeds
// it) so a divergent assert cannot deadlock the workgroup. runs post-mono after inlining but before the arena
// passes, so a runtime message's copy loop is arena-lowered with the rest; a no-op without asserts. an entry
// that asserts gains `var #asserted = false` and a leading `#error: i8*` arg (leading so the arena passes'
// view args cannot displace it) the dispatch fills with a zeroed `[code:u32 little-endian][message: NUL-
// terminated, up to AssertMessageLimit]` buffer. the flag + break + guard are pure control flow and lower
// identically on every backend
// examples:
//   assert(c, m)             ->  #asserted = true; #error[0..4) = c (le);        // then m's bytes inlined
//   while (b) { p[i]=v; a }  ->  while (b) { if (#asserted) break; p[i]=v; a }   // a barrier-free loop drains
//   p[i]=v; a; q[i]=w        ->  p[i]=v; a; if (!#asserted) { q[i]=w }           // the tail is fenced
//   a; barrier(); q          ->  a; barrier(); if (!#asserted) { q }             // the barrier stays unfenced
// edge cases:
//   value-returning entry        ->  drains to `return poison`; the host discards it once it reads the buffer
//   string-literal message       ->  inlined as byte stores (rusticl's SPIR-V loader panics on a string global)
//   assert in a barrier loop     ->  rejected: draining it would skip the barrier and hang the workgroup
// try/catch/finally uses the same drain flag plus a dense type tag and one payload slot per raised type:
// the first compatible handler clears the flag and binds its projected payload; an unmatched raise keeps
// draining. finally saves pending exception state, runs once on normal, exceptional, and abrupt exits, and
// replaces the pending exit when the finalizer itself exits.
object StructuredExit extends ProgramPass {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private val AssertedSym            = p.Conventions.AssertedFlag
  private val ErrorSym               = p.Conventions.ErrorArg
  private val ErrorPtr               = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)
  private val CodeBytes              = 4 // the [code:u32 little-endian] prefix; matches polyrt::assertCodeBytes
  private val MessageLimit           = p.Conventions.assertMessageLimit
  private val TagSym                 = "#exn_tag"
  private val SlotPrefix             = "#exn_v"
  private val ExceptionSym           = p.Conventions.ExceptionValue
  private val ExceptionWhatSym       = p.Conventions.ExceptionWhat
  private val ExceptionWhatBufferSym = "#exn_what"
  private val ExceptionWhatBuffer    = p.Type.Arr(p.Type.IntS8, MessageLimit, p.Type.Space.Private)
  private val ExceptionWhatPtr       = p.Type.Ptr(p.Type.IntS8, p.Type.Space.Private)
  private val ExceptionCodeSym       = p.Conventions.ExceptionCode
  private val AssertTag              = 0

  private val asserted = sel(p.Named(AssertedSym, p.Type.Bool1))
  private val error    = sel(p.Named(ErrorSym, ErrorPtr))
  private val tagSel   = sel(p.Named(TagSym, p.Type.IntS32))

  private def declarations(body: List[p.Stmt]): List[p.Named] = body.flatMap {
    case p.Stmt.Var(n, _, _)            => List(n)
    case p.Stmt.While(_, b)             => declarations(b)
    case p.Stmt.ForRange(i, _, _, _, b) => i :: declarations(b)
    case p.Stmt.Cond(_, t, f)           => declarations(t) ::: declarations(f)
    case p.Stmt.Raise(_, _, cleanup)    => declarations(cleanup)
    case p.Stmt.Try(b, hs, fin) =>
      declarations(b) ::: hs.flatMap(h => h.binder.toList ::: declarations(h.body)) ::: declarations(fin)
    case p.Stmt.Annotated(inner, _, _) => declarations(List(inner))
    case _                             => Nil
  }

  private final class FinallySymbols(body: List[p.Stmt]) {
    private var counter = 0L
    private val occupied = scala.collection.mutable.Set.from(
      declarations(body).map(_.symbol) ::: body.flatMap(
        _.collectWhere[p.Term] { case p.Term.Select(root, _, _) => root.symbol }
      )
    )

    def fresh(suffix: String): String = {
      counter += 1
      var candidate = s"#finally${counter}_$suffix"
      while (occupied(candidate)) {
        counter += 1
        candidate = s"#finally${counter}_$suffix"
      }
      occupied += candidate
      candidate
    }
  }

  private def materialiseFinally(stmts: List[p.Stmt], symbols: FinallySymbols): List[p.Stmt] = {

    def freshCopy(stmts: List[p.Stmt]): List[p.Stmt] = {
      val declared           = declarations(stmts)
      val names              = declared.distinct.map(n => n -> n.copy(symbol = symbols.fresh(n.symbol))).toMap
      def rename(n: p.Named) = names.getOrElse(n, n)
      stmts
        .modifyAll[p.Term] {
          case p.Term.Select(root, steps, tpe) => p.Term.Select(rename(root), steps, tpe)
          case x                               => x
        }
        .modifyAll[p.Stmt] {
          case p.Stmt.Var(n, expr, mut)            => p.Stmt.Var(rename(n), expr, mut)
          case p.Stmt.ForRange(i, lb, ub, step, b) => p.Stmt.ForRange(rename(i), lb, ub, step, b)
          case p.Stmt.Try(b, hs, fin) =>
            p.Stmt.Try(b, hs.map(h => h.copy(binder = h.binder.map(rename))), fin)
          case x => x
        }
    }

    def rewrite(body: List[p.Stmt], cleanups: List[(List[p.Stmt], Int)], loopDepth: Int): List[p.Stmt] =
      body.flatMap {
        case r: p.Stmt.Return => cleanups.flatMap((fin, _) => freshCopy(fin)) :+ r
        case p.Stmt.Break =>
          cleanups.filter(_._2 >= loopDepth).flatMap((fin, _) => freshCopy(fin)) :+ p.Stmt.Break
        case p.Stmt.Cont =>
          cleanups.filter(_._2 >= loopDepth).flatMap((fin, _) => freshCopy(fin)) :+ p.Stmt.Cont
        case p.Stmt.Cond(c, t, f) =>
          List(p.Stmt.Cond(c, rewrite(t, cleanups, loopDepth), rewrite(f, cleanups, loopDepth)))
        case p.Stmt.While(c, b) => List(p.Stmt.While(c, rewrite(b, cleanups, loopDepth + 1)))
        case p.Stmt.ForRange(i, lb, ub, step, b) =>
          List(p.Stmt.ForRange(i, lb, ub, step, rewrite(b, cleanups, loopDepth + 1)))
        case p.Stmt.Try(b, hs, fin) =>
          if (fin.isEmpty)
            List(
              p.Stmt.Try(
                rewrite(b, cleanups, loopDepth),
                hs.map(h => h.copy(body = rewrite(h.body, cleanups, loopDepth))),
                Nil
              )
            )
          else {
            val completed    = p.Named(symbols.fresh("done"), p.Type.Bool1)
            val rewrittenFin = rewrite(fin, cleanups, loopDepth)
            val runOnce      = p.Stmt.Mut(sel(completed), p.Expr.Alias(p.Term.Bool1Const(true))) :: rewrittenFin
            val nested       = (runOnce -> loopDepth) :: cleanups
            List(
              p.Stmt.Var(completed, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true),
              p.Stmt.Try(
                rewrite(b, nested, loopDepth),
                hs.map { h =>
                  val body = rewrite(h.body, nested, loopDepth)
                  h.copy(body = List(p.Stmt.Cond(sel(completed), List(p.Stmt.Rethrow), body)))
                },
                List(p.Stmt.Cond(sel(completed), Nil, runOnce))
              )
            )
          }
        case p.Stmt.Annotated(inner, pos, comment) =>
          rewrite(List(inner), cleanups, loopDepth).map(p.Stmt.Annotated(_, pos, comment))
        case other => List(other)
      }

    rewrite(stmts, Nil, 0)
  }

  private def isAssert(s: p.Stmt): Boolean = s match {
    case p.Stmt.Var(_, Some(p.Expr.SpecOp(_: p.Spec.Assert)), _) => true
    case _                                                       => false
  }

  private def isBarrier(s: p.Stmt): Boolean = s match {
    case p.Stmt.Var(
          _,
          Some(p.Expr.SpecOp(p.Spec.GpuBarrierGlobal | p.Spec.GpuBarrierLocal | p.Spec.GpuBarrierAll)),
          _
        ) =>
      true
    case _ => false
  }

  private def mayAssert(s: p.Stmt): Boolean       = s.collectFirst_[p.Stmt] { case x if isAssert(x) => x }.isDefined
  private def containsBarrier(s: p.Stmt): Boolean = s.collectFirst_[p.Stmt] { case x if isBarrier(x) => x }.isDefined

  private final class Flow(defs: List[p.StructDef]) {
    private val byName      = defs.map(d => d.name -> d).toMap
    private val baseByField = defs.map(d => s"${p.Conventions.BaseFieldPrefix}_${d.name.repr}" -> d).toMap

    def projection(from: p.Type, to: p.Type): Option[List[p.PathStep]] =
      if (from == to) Some(Nil)
      else
        (from, to) match {
          case (s: p.Type.Struct, target: p.Type.Struct) =>
            byName
              .get(s.name)
              .toList
              .flatMap(_.members)
              .collect {
                case p.Named(name, parent: p.Type.Struct, _) if name.startsWith(p.Conventions.BaseFieldPrefix) =>
                  name -> parent
              }
              .iterator
              .flatMap { (name, parent) =>
                val direct = baseByField.get(name)
                if (direct.exists(d => d.name == target.name || (d.name != parent.name && d.parents.contains(target))))
                  Some(List(p.PathStep.Field(name)))
                else projection(parent, to).map(p.PathStep.Field(name) :: _)
              }
              .nextOption()
          case _ => None
        }

    private def convertible(raised: p.Type, caught: p.Type): Boolean = (raised, caught) match {
      case (s: p.Type.Struct, _: p.Type.Struct) =>
        byName.get(s.name).exists(_.parents.contains(caught)) && projection(raised, caught).isDefined
      case (p.Type.Ptr(p.Type.Nothing, p.Type.Space.Constant), _: p.Type.Ptr)           => true
      case (p.Type.Ptr(from, _), p.Type.Ptr(p.Type.Unit0, _)) if from != p.Type.Nothing => true
      case (p.Type.Ptr(from: p.Type.Struct, _), p.Type.Ptr(to: p.Type.Struct, _)) =>
        byName.get(from.name).exists(_.parents.contains(to)) && projection(from, to).isDefined
      case _ => false
    }

    def catches(handler: p.Handler, raised: p.ExceptionKind): Boolean =
      handler.caught.forall(_.catches(raised)(convertible))

    private def rethrowsCurrent(stmts: List[p.Stmt]): Boolean = stmts.exists {
      case p.Stmt.Rethrow                 => true
      case p.Stmt.Cond(_, t, f)           => rethrowsCurrent(t) || rethrowsCurrent(f)
      case p.Stmt.While(_, b)             => rethrowsCurrent(b)
      case p.Stmt.ForRange(_, _, _, _, b) => rethrowsCurrent(b)
      case p.Stmt.Annotated(inner, _, _)  => rethrowsCurrent(List(inner))
      case p.Stmt.Try(body, _, fin)       => rethrowsCurrent(body) || rethrowsCurrent(fin)
      case _                              => false
    }

    private def unhandled(t: p.Stmt.Try): Set[p.ExceptionKind] = raised(t.body).filter { thrown =>
      t.handlers.find(h => catches(h, thrown)).forall(h => rethrowsCurrent(h.body))
    }

    def raised(stmts: List[p.Stmt]): Set[p.ExceptionKind] = stmts.iterator.flatMap(raised).toSet

    private def raised(s: p.Stmt): Set[p.ExceptionKind] = s match {
      case p.Stmt.Raise(_, exceptionKind, _) => Set(exceptionKind)
      case p.Stmt.Rethrow                    => Set.empty
      case p.Stmt.Cond(_, t, f)              => raised(t) ++ raised(f)
      case p.Stmt.While(_, b)                => raised(b)
      case p.Stmt.ForRange(_, _, _, _, b)    => raised(b)
      case p.Stmt.Annotated(inner, _, _)     => raised(inner)
      case t: p.Stmt.Try                     => unhandled(t) ++ raised(t.handlers.flatMap(_.body)) ++ raised(t.fin)
      case _                                 => Set.empty
    }

    def escapes(s: p.Stmt): Boolean = s match {
      case _ if isAssert(s)               => true
      case _: p.Stmt.Raise                => true
      case p.Stmt.Rethrow                 => true
      case p.Stmt.Cond(_, t, f)           => t.exists(escapes) || f.exists(escapes)
      case p.Stmt.While(_, b)             => b.exists(escapes)
      case p.Stmt.ForRange(_, _, _, _, b) => b.exists(escapes)
      case p.Stmt.Annotated(inner, _, _)  => escapes(inner)
      case t: p.Stmt.Try =>
        t.body.exists(mayAssert) || t.handlers.exists(_.body.exists(escapes)) || t.fin.exists(escapes) ||
        unhandled(t).nonEmpty
      case _ => false
    }

    def hasBarrierEscape(s: p.Stmt): Boolean = s
      .collectFirst_[p.Stmt] {
        case l @ p.Stmt.While(_, b) if b.exists(escapes) && b.exists(containsBarrier)             => l
        case l @ p.Stmt.ForRange(_, _, _, _, b) if b.exists(escapes) && b.exists(containsBarrier) => l
      }
      .isDefined
  }

  private def hasStorage(t: p.Type): Boolean = t != p.Type.Unit0 && t != p.Type.Nothing

  private final case class Raised(exceptionKind: p.ExceptionKind, cleanup: List[p.Stmt]) {
    def thrown: p.ExceptionKind = exceptionKind
    def tpe: p.Type             = exceptionKind.tpe
  }

  // repr ordering keeps the dense tags stable in emitted images
  private def tagTable(f: p.Function): List[Raised] =
    f.collectWhere[p.Stmt] { case p.Stmt.Raise(_, exceptionKind, cleanup) => Raised(exceptionKind, cleanup) }
      .distinctBy(_.thrown)
      .sortBy(x => (x.tpe.repr, x.exceptionKind.sourceName))

  private def observesExceptionWhat(stmts: List[p.Stmt]): Boolean = {
    def termUsesWhat(s: p.Stmt): Boolean = s
      .collectFirst_[p.Term] {
        case t @ p.Term.Select(root, _, _)
            if root.symbol == ExceptionWhatSym || root.symbol.endsWith(p.Conventions.ExceptionWhatSuffix) =>
          t
      }
      .isDefined
    stmts.exists {
      case p.Stmt.Mut(p.Term.Select(root, Nil, _), _) if root.symbol == ExceptionWhatSym => false
      case p.Stmt.While(_, body)                                                         => observesExceptionWhat(body)
      case p.Stmt.ForRange(_, _, _, _, body)                                             => observesExceptionWhat(body)
      case p.Stmt.Cond(_, ifTrue, ifFalse) => observesExceptionWhat(ifTrue) || observesExceptionWhat(ifFalse)
      case p.Stmt.Try(body, handlers, fin) =>
        observesExceptionWhat(body) || handlers.exists(h => observesExceptionWhat(h.body)) || observesExceptionWhat(fin)
      case p.Stmt.Annotated(inner, _, _) => observesExceptionWhat(List(inner))
      case other                         => termUsesWhat(other)
    }
  }

  private final class Lower(tags: List[Raised], flow: Flow, usesWhat: Boolean, finallySymbols: FinallySymbols) {

    private val tagOf   = tags.map(_.thrown).zipWithIndex.map((t, i) => t -> (i + 1)).toMap
    private val raised  = tags.map(x => x.thrown -> x).toMap
    private var counter = 0L

    private def fresh(tpe: p.Type): p.Named = {
      counter += 1
      p.Named(s"#as$counter", tpe)
    }

    private def freshCopy(stmts: List[p.Stmt]): List[p.Stmt] = {
      def declarations(body: List[p.Stmt]): List[p.Named] = body.flatMap {
        case p.Stmt.Var(n, _, _)            => List(n)
        case p.Stmt.While(_, b)             => declarations(b)
        case p.Stmt.ForRange(i, _, _, _, b) => i :: declarations(b)
        case p.Stmt.Cond(_, t, f)           => declarations(t) ::: declarations(f)
        case p.Stmt.Try(b, hs, fin) =>
          declarations(b) ::: hs.flatMap(h => h.binder.toList ::: declarations(h.body)) ::: declarations(fin)
        case p.Stmt.Annotated(inner, _, _) => declarations(List(inner))
        case _                             => Nil
      }
      val names                       = declarations(stmts).distinct.map(n => n -> fresh(n.tpe)).toMap
      def rename(n: p.Named): p.Named = names.getOrElse(n, n)
      stmts
        .modifyAll[p.Term] {
          case p.Term.Select(root, steps, tpe) => p.Term.Select(rename(root), steps, tpe)
          case other                           => other
        }
        .modifyAll[p.Stmt] {
          case p.Stmt.Var(n, expr, mut)            => p.Stmt.Var(rename(n), expr, mut)
          case p.Stmt.ForRange(i, lb, ub, step, b) => p.Stmt.ForRange(rename(i), lb, ub, step, b)
          case p.Stmt.Try(b, hs, fin) =>
            p.Stmt.Try(b, hs.map(h => h.copy(binder = h.binder.map(rename))), fin)
          case other => other
        }
    }

    private def let(tpe: p.Type, e: p.Expr, into: ListBuffer[p.Stmt]): p.Term = {
      val n = fresh(tpe)
      into += p.Stmt.Var(n, Some(e), isMutable = false)
      sel(n)
    }

    private def ifThen(test: p.Expr, taken: List[p.Stmt], els: List[p.Stmt]): List[p.Stmt] = {
      val c = fresh(p.Type.Bool1)
      List(p.Stmt.Var(c, Some(test), isMutable = false), p.Stmt.Cond(sel(c), taken, els))
    }

    private def selectTag(
        subject: p.Term,
        cases: List[(p.ExceptionKind, List[p.Stmt])],
        otherwise: List[p.Stmt] = Nil
    ): List[p.Stmt] =
      cases.foldRight(otherwise) { case ((tpe, taken), els) =>
        ifThen(p.Expr.IntrOp(p.Intr.LogicEq(subject, p.Term.IntS32Const(tagOf(tpe)))), taken, els)
      }

    // the error buffer is a byte protocol, not a sequence of UTF-16 chars
    private def writeMessage(s: String): List[p.Stmt] =
      s.getBytes(StandardCharsets.UTF_8).take(MessageLimit).toList.zipWithIndex.map { (byte, k) =>
        p.Stmt.Update(error, p.Term.IntU32Const(CodeBytes + k), p.Term.IntS8Const(byte))
      }

    private def slotOf(t: p.ExceptionKind): p.Named = p.Named(s"$SlotPrefix${tagOf(t)}", t.tpe)

    private val messageBuffer          = p.Named(ExceptionWhatBufferSym, ExceptionWhatBuffer)
    private val deferredAssertMessages = ListBuffer.empty[(p.Named, p.Named)]

    private def messageRef(buffer: p.Named): p.Expr =
      p.Expr.RefTo(
        sel(buffer),
        Some(p.Term.IntU32Const(0)),
        p.Type.IntS8,
        p.Type.Space.Private,
        p.Region.Rooted(buffer)
      )

    def slotDecls: List[p.Stmt] =
      tags.map(_.thrown).filter(t => hasStorage(t.tpe)).map(t => p.Stmt.Var(slotOf(t), None, isMutable = true))

    def messageDecls: List[p.Stmt] =
      Option
        .when(usesWhat)(
          List(
            p.Stmt.Var(messageBuffer, None, isMutable = true),
            p.Stmt.Var(
              p.Named(ExceptionWhatSym, ExceptionWhatPtr),
              Some(messageRef(messageBuffer)),
              isMutable = false
            )
          )
        )
        .toList
        .flatten

    private def copyCString(
        source: p.Term,
        bound: Int,
        initiallyActive: p.Term = p.Term.Bool1Const(true)
    )(write: (p.Term, p.Term) => List[p.Stmt]): List[p.Stmt] = {
      require(bound > 0)
      val i      = fresh(p.Type.IntU32)
      val active = fresh(p.Type.Bool1)
      val body   = ListBuffer.empty[p.Stmt]
      val ch     = let(p.Type.IntS8, p.Expr.Index(source, sel(i), p.Type.IntS8), body)
      body ++= write(sel(i), ch)
      val next = let(
        p.Type.IntU32,
        p.Expr.IntrOp(p.Intr.Add(sel(i), p.Term.IntU32Const(1), p.Type.IntU32)),
        body
      )
      body += p.Stmt.Mut(sel(i), p.Expr.Alias(next))
      val notNul = let(p.Type.Bool1, p.Expr.IntrOp(p.Intr.LogicNeq(ch, p.Term.IntS8Const(0))), body)
      val within = let(
        p.Type.Bool1,
        p.Expr.IntrOp(p.Intr.LogicLt(next, p.Term.IntU32Const(bound))),
        body
      )
      body += p.Stmt.Mut(sel(active), p.Expr.IntrOp(p.Intr.LogicAnd(notNul, within)))
      List(
        p.Stmt.Var(i, Some(p.Expr.Alias(p.Term.IntU32Const(0))), isMutable = true),
        p.Stmt.Var(active, Some(p.Expr.Alias(initiallyActive)), isMutable = true),
        p.Stmt.While(sel(active), body.toList)
      )
    }

    def assertMessageDecls: List[p.Stmt] = deferredAssertMessages.toList.flatMap { (message, active) =>
      List(
        p.Stmt.Var(message, None, isMutable = true),
        p.Stmt.Var(active, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
      )
    }

    def assertMessageEpilogue: List[p.Stmt] = deferredAssertMessages.toList.flatMap { (message, enabled) =>
      copyCString(sel(message), MessageLimit, sel(enabled)) { (i, ch) =>
        val write = ListBuffer.empty[p.Stmt]
        val off = let(
          p.Type.IntU32,
          p.Expr.IntrOp(p.Intr.Add(p.Term.IntU32Const(CodeBytes), i, p.Type.IntU32)),
          write
        )
        write += p.Stmt.Update(error, off, ch)
        write.toList
      }
    }

    private def copyMessage(source: p.Term, target: p.Named = messageBuffer): List[p.Stmt] = {
      source match {
        case p.Term.StringConst(s) =>
          val bytes = s.getBytes(StandardCharsets.UTF_8).take(MessageLimit - 1).toList
          return bytes.zipWithIndex.map((byte, i) =>
            p.Stmt.Update(sel(target), p.Term.IntU32Const(i), p.Term.IntS8Const(byte))
          ) :+ p.Stmt.Update(sel(target), p.Term.IntU32Const(bytes.size), p.Term.IntS8Const(0))
        case _ =>
      }
      copyCString(source, MessageLimit - 1)((i, ch) => List(p.Stmt.Update(sel(target), i, ch))) :+
        p.Stmt.Update(sel(target), p.Term.IntU32Const(MessageLimit - 1), p.Term.IntS8Const(0))
    }

    private def copyBuffer(source: p.Named, target: p.Named): List[p.Stmt] = {
      val i  = fresh(p.Type.IntU32)
      val ch = fresh(p.Type.IntS8)
      List(
        p.Stmt.ForRange(
          i,
          p.Term.IntU32Const(0),
          p.Term.IntU32Const(MessageLimit),
          p.Term.IntU32Const(1),
          List(
            p.Stmt.Var(ch, Some(p.Expr.Index(sel(source), sel(i), p.Type.IntS8)), isMutable = false),
            p.Stmt.Update(sel(target), sel(i), sel(ch))
          )
        )
      )
    }

    private def lowerAssert(code: p.Term, message: p.Term): List[p.Stmt] = {
      val out = ListBuffer[p.Stmt](p.Stmt.Mut(asserted, p.Expr.Alias(p.Term.Bool1Const(true))))
      // do not let a handled raise turn a later assertion into a catchable exception
      if (tags.nonEmpty) out += p.Stmt.Mut(tagSel, p.Expr.Alias(p.Term.IntS32Const(AssertTag)))
      (0 until CodeBytes).foreach { k =>
        val shifted =
          let(p.Type.IntU32, p.Expr.IntrOp(p.Intr.BSR(code, p.Term.IntU32Const(8 * k), p.Type.IntU32)), out)
        val byte = let(p.Type.IntS8, p.Expr.Cast(shifted, p.Type.IntS8), out)
        out += p.Stmt.Update(error, p.Term.IntU32Const(k), byte)
      }
      message match {
        case p.Term.StringConst(s) => out ++= writeMessage(s)
        case _ =>
          val saved  = fresh(message.tpe)
          val active = fresh(p.Type.Bool1)
          deferredAssertMessages += saved -> active
          out += p.Stmt.Mut(sel(saved), p.Expr.Alias(message))
          out += p.Stmt.Mut(sel(active), p.Expr.Alias(p.Term.Bool1Const(true)))
      }
      out.toList
    }

    private def lowerRaise(v: p.Term, exceptionKind: p.ExceptionKind): List[p.Stmt] =
      List(
        p.Stmt.Mut(asserted, p.Expr.Alias(p.Term.Bool1Const(true))),
        p.Stmt.Mut(tagSel, p.Expr.Alias(p.Term.IntS32Const(tagOf(exceptionKind))))
      ) ::: (if (hasStorage(v.tpe)) List(p.Stmt.Mut(sel(slotOf(exceptionKind)), p.Expr.Alias(v))) else Nil)

    private def materialiseRethrow(
        stmts: List[p.Stmt],
        value: p.Term,
        raised: Raised,
        marker: p.Term.Select,
        what: p.Term,
        code: p.Term
    ): (List[p.Stmt], Boolean) = {
      var found = false
      def rewrite(body: List[p.Stmt]): List[p.Stmt] = body.flatMap {
        case p.Stmt.Rethrow =>
          found = true
          List(
            p.Stmt.Mut(marker, p.Expr.Alias(p.Term.Bool1Const(true))),
            p.Stmt.Mut(sel(p.Named(ExceptionWhatSym, ExceptionWhatPtr)), p.Expr.Alias(what)),
            p.Stmt.Mut(sel(p.Named(ExceptionCodeSym, p.Type.IntS32)), p.Expr.Alias(code)),
            p.Stmt.Raise(value, raised.exceptionKind, raised.cleanup)
          )
        case p.Stmt.Cond(c, t, f)            => List(p.Stmt.Cond(c, rewrite(t), rewrite(f)))
        case p.Stmt.While(c, b)              => List(p.Stmt.While(c, rewrite(b)))
        case p.Stmt.ForRange(i, l, u, s, b)  => List(p.Stmt.ForRange(i, l, u, s, rewrite(b)))
        case p.Stmt.Try(b, hs, fin)          => List(p.Stmt.Try(rewrite(b), hs, rewrite(fin)))
        case p.Stmt.Annotated(inner, pos, c) => rewrite(List(inner)).map(p.Stmt.Annotated(_, pos, c))
        case other                           => List(other)
      }
      rewrite(stmts) -> found
    }

    private def cleanupFor(thrown: p.ExceptionKind, storage: p.Named): List[p.Stmt] =
      freshCopy(raised(thrown).cleanup).modifyAll[p.Term] {
        case p.Term.Select(root, steps, selected) if root.symbol == ExceptionSym =>
          p.Term.Select(storage, steps, selected)
        case other => other
      }

    private def runFinally(fin: List[p.Stmt], pendingPossible: Boolean): List[p.Stmt] =
      if (fin.isEmpty) Nil
      else if (!pendingPossible) guard(fin)
      else {
        val savedFlag  = fresh(p.Type.Bool1)
        val savedTag   = Option.when(tags.nonEmpty)(fresh(p.Type.IntS32))
        val savedWhat  = Option.when(tags.nonEmpty && usesWhat)(fresh(ExceptionWhatBuffer))
        val savedCode  = Option.when(tags.nonEmpty)(fresh(p.Type.IntS32))
        val savedSlots = tags.collect { case x if hasStorage(x.tpe) => x.thrown -> fresh(x.tpe) }
        val slotDecls  = savedSlots.map((_, saved) => p.Stmt.Var(saved, None, isMutable = true))
        val saveSlots = selectTag(
          tagSel,
          savedSlots.map((tpe, saved) => tpe -> List(p.Stmt.Mut(sel(saved), p.Expr.Alias(sel(slotOf(tpe))))))
        )
        val restoreSlots = savedTag.fold(List.empty[p.Stmt])(tag =>
          selectTag(
            sel(tag),
            savedSlots.map((tpe, saved) => tpe -> List(p.Stmt.Mut(sel(slotOf(tpe)), p.Expr.Alias(sel(saved)))))
          )
        )
        val restoreRaised =
          savedTag.toList.map(t => p.Stmt.Mut(tagSel, p.Expr.Alias(sel(t)))) :::
            savedWhat.toList.flatMap(t => copyBuffer(t, messageBuffer)) :::
            savedCode.toList.map(t =>
              p.Stmt.Mut(sel(p.Named(ExceptionCodeSym, p.Type.IntS32)), p.Expr.Alias(sel(t)))
            ) :::
            restoreSlots
        val restore =
          p.Stmt.Mut(asserted, p.Expr.Alias(sel(savedFlag))) ::
            ifThen(p.Expr.Alias(sel(savedFlag)), restoreRaised, Nil)
        val discardPending = savedTag.toList.flatMap(tag =>
          ifThen(
            p.Expr.Alias(sel(savedFlag)),
            selectTag(
              sel(tag),
              savedSlots.map((thrown, saved) => thrown -> cleanupFor(thrown, saved))
            ),
            Nil
          )
        )
        def discardBeforeAbrupt(stmts: List[p.Stmt], loopDepth: Int = 0): List[p.Stmt] = stmts.flatMap {
          case exit: p.Stmt.Return            => freshCopy(discardPending) :+ exit
          case p.Stmt.Break if loopDepth == 0 => freshCopy(discardPending) :+ p.Stmt.Break
          case p.Stmt.Cont if loopDepth == 0  => freshCopy(discardPending) :+ p.Stmt.Cont
          case p.Stmt.While(cond, body) =>
            List(p.Stmt.While(cond, discardBeforeAbrupt(body, loopDepth + 1)))
          case p.Stmt.ForRange(induction, lb, ub, step, body) =>
            List(p.Stmt.ForRange(induction, lb, ub, step, discardBeforeAbrupt(body, loopDepth + 1)))
          case p.Stmt.Cond(cond, ifTrue, ifFalse) =>
            List(
              p.Stmt.Cond(
                cond,
                discardBeforeAbrupt(ifTrue, loopDepth),
                discardBeforeAbrupt(ifFalse, loopDepth)
              )
            )
          case p.Stmt.Try(body, handlers, nestedFin) =>
            List(
              p.Stmt.Try(
                discardBeforeAbrupt(body, loopDepth),
                handlers.map(h => h.copy(body = discardBeforeAbrupt(h.body, loopDepth))),
                discardBeforeAbrupt(nestedFin, loopDepth)
              )
            )
          case p.Stmt.Annotated(inner, pos, comment) =>
            discardBeforeAbrupt(List(inner), loopDepth).map(p.Stmt.Annotated(_, pos, comment))
          case other => List(other)
        }

        List(p.Stmt.Var(savedFlag, Some(p.Expr.Alias(asserted)), isMutable = false)) :::
          savedTag.toList.map(t => p.Stmt.Var(t, Some(p.Expr.Alias(tagSel)), isMutable = false)) :::
          savedWhat.toList.flatMap(t => p.Stmt.Var(t, None, isMutable = true) :: copyBuffer(messageBuffer, t)) :::
          savedCode.toList.map(t =>
            p.Stmt.Var(t, Some(p.Expr.Alias(sel(p.Named(ExceptionCodeSym, p.Type.IntS32)))), isMutable = false)
          ) :::
          slotDecls ::: saveSlots :::
          List(p.Stmt.Mut(asserted, p.Expr.Alias(p.Term.Bool1Const(false)))) :::
          Option.when(tags.nonEmpty)(p.Stmt.Mut(tagSel, p.Expr.Alias(p.Term.IntS32Const(AssertTag)))).toList :::
          discardBeforeAbrupt(guard(fin)) :::
          ifThen(p.Expr.Alias(asserted), discardPending, restore)
      }

    private def bindHandler(
        handler: p.Handler,
        thrown: p.ExceptionKind,
        storage: p.Named,
        body: List[p.Stmt]
    ): List[p.Stmt] =
      (handler.caught.map(_.tpe), handler.binder) match {
        case (Some(caught), Some(binder)) if hasStorage(caught) =>
          val slot = storage
          (thrown.tpe, caught) match {
            case (from: p.Type.Ptr, to: p.Type.Ptr) if from != to =>
              val source = sel(slot)
              val init = (from.comp, to.comp) match {
                case (p.Type.Nothing, _) if from.space == p.Type.Space.Constant =>
                  p.Expr.Alias(p.Term.NullPtrConst(to.comp, to.space, p.Region.Opaque))
                case (_, p.Type.Unit0) => p.Expr.Cast(source, to)
                case (fromComp: p.Type.Struct, toComp: p.Type.Struct) =>
                  val projection = flow.projection(fromComp, toComp).get
                  val projected  = p.Term.Select(slot, p.PathStep.Deref :: projection, toComp)
                  val nonNull = p.Expr.IntrOp(
                    p.Intr.LogicNeq(source, p.Term.NullPtrConst(from.comp, from.space, p.Region.Opaque))
                  )
                  val projectedBody = body.modifyAll[p.Term] {
                    case p.Term.Select(root, steps, tpe) if root == binder && steps.nonEmpty =>
                      p.Term.Select(slot, p.PathStep.Deref :: projection ::: steps, tpe)
                    case other => other
                  }
                  val usesPointerValue = body
                    .collectFirst_[p.Term] {
                      case term @ p.Term.Select(root, Nil, _) if root == binder => term
                    }
                    .isDefined
                  if (!usesPointerValue)
                    return p.Stmt.Var(
                      binder,
                      Some(p.Expr.RefTo(projected, None, toComp, to.space, p.Region.Opaque)),
                      isMutable = false
                    ) :: projectedBody
                  val nullBinder = fresh(to)
                  val nullBody = freshCopy(body.modifyAll[p.Term] {
                    case p.Term.Select(root, steps, tpe) if root == binder =>
                      p.Term.Select(nullBinder, steps, tpe)
                    case other => other
                  })
                  return ifThen(
                    nonNull,
                    p.Stmt.Var(
                      binder,
                      Some(p.Expr.RefTo(projected, None, toComp, to.space, p.Region.Opaque)),
                      isMutable = false
                    ) :: projectedBody,
                    p.Stmt.Var(
                      nullBinder,
                      Some(p.Expr.Alias(p.Term.NullPtrConst(to.comp, to.space, p.Region.Opaque))),
                      isMutable = false
                    ) :: nullBody
                  )
                case _ => throw IllegalStateException(s"invalid pointer handler conversion ${from.repr} -> ${to.repr}")
              }
              p.Stmt.Var(binder, Some(init), isMutable = false) :: body
            case _ =>
              val source = p.Term.Select(slot, flow.projection(thrown.tpe, caught).get, caught)
              val init = binder.tpe match {
                case p.Type.Ptr(component, space) if component == caught =>
                  p.Expr.RefTo(source, None, caught, space, p.Region.Rooted(slot))
                case _ => p.Expr.Alias(source)
              }
              p.Stmt.Var(binder, Some(init), isMutable = false) :: body
          }
        case _ => body
      }

    private def handlerState(handler: p.Handler, suffix: String, tpe: p.Type, value: p.Term): List[p.Stmt] =
      handler.binder.toList.flatMap { binder =>
        val state = p.Named(s"${binder.symbol}$suffix", tpe)
        Option
          .when(
            handler.body.collectFirst_[p.Term] { case s @ p.Term.Select(root, _, _) if root == state => s }.isDefined
          )(
            p.Stmt.Var(state, Some(p.Expr.Alias(value)), isMutable = false)
          )
          .toList
      }

    private def lowerTry(
        body: List[p.Stmt],
        handlers: List[p.Handler],
        fin: List[p.Stmt]
    ): List[p.Stmt] = {
      val raised = flow.raised(body)
      val chain =
        if (raised.isEmpty) Nil
        else
          handlers.foldRight(List.empty[p.Stmt]) { (h, nextHandler) =>
            val matches = raised.toList.filter(flow.catches(h, _)).sortBy(tagOf)
            selectTag(
              tagSel,
              matches.map { thrown =>
                val saved        = fresh(thrown.tpe)
                val savedWhat    = Option.when(usesWhat)(fresh(ExceptionWhatBuffer))
                val savedWhatPtr = Option.when(usesWhat)(fresh(ExceptionWhatPtr))
                val savedCode    = fresh(p.Type.IntS32)
                val rethrowing   = fresh(p.Type.Bool1)
                val info         = this.raised(thrown)
                val what = savedWhatPtr
                  .map(sel)
                  .getOrElse(p.Term.NullPtrConst(p.Type.IntS8, p.Type.Space.Private, p.Region.Opaque))
                val (handlerBody, rethrows) =
                  materialiseRethrow(h.body, sel(saved), info, sel(rethrowing), what, sel(savedCode))
                val cleanup = cleanupFor(thrown, saved)
                val finalizer =
                  if (cleanup.isEmpty) Nil
                  else if (rethrows) List(p.Stmt.Cond(sel(rethrowing), Nil, cleanup))
                  else cleanup
                val protectedBody =
                  if (finalizer.isEmpty) handlerBody
                  else materialiseFinally(List(p.Stmt.Try(handlerBody, Nil, finalizer)), finallySymbols)
                val setup = List(p.Stmt.Var(saved, Some(p.Expr.Alias(sel(slotOf(thrown)))), isMutable = false)) :::
                  savedWhat.toList.flatMap(t =>
                    p.Stmt.Var(t, None, isMutable = true) :: copyBuffer(messageBuffer, t)
                  ) :::
                  savedWhat
                    .zip(savedWhatPtr)
                    .toList
                    .map((buffer, ptr) => p.Stmt.Var(ptr, Some(messageRef(buffer)), isMutable = false)) :::
                  List(
                    p.Stmt.Var(
                      savedCode,
                      Some(p.Expr.Alias(sel(p.Named(ExceptionCodeSym, p.Type.IntS32)))),
                      isMutable = false
                    )
                  ) :::
                  savedWhatPtr.toList.flatMap(t =>
                    handlerState(h, p.Conventions.ExceptionWhatSuffix, ExceptionWhatPtr, sel(t))
                  ) :::
                  handlerState(h, p.Conventions.ExceptionCodeSuffix, p.Type.IntS32, sel(savedCode)) :::
                  Option
                    .when(rethrows)(
                      p.Stmt.Var(
                        rethrowing,
                        Some(p.Expr.Alias(p.Term.Bool1Const(false))),
                        isMutable = true
                      )
                    )
                    .toList
                val taken = setup :::
                  p.Stmt.Mut(asserted, p.Expr.Alias(p.Term.Bool1Const(false))) ::
                  bindHandler(h, thrown, saved, guard(protectedBody))
                thrown -> taken
              },
              nextHandler
            )
          }
      val dispatch = if (chain.isEmpty) Nil else List(p.Stmt.Cond(asserted, chain, Nil))
      val pendingPossible = body.exists(mayAssert) ||
        raised.exists(t => !handlers.exists(h => flow.catches(h, t))) || handlers.exists(
          _.body.exists(flow.escapes)
        )
      guard(body) ::: dispatch ::: runFinally(fin, pendingPossible)
    }

    private def lower(s: p.Stmt, canBreak: Boolean, breakOnRaise: Boolean): List[p.Stmt] = s match {
      case p.Stmt.Var(_, Some(p.Expr.SpecOp(p.Spec.Assert(code, message))), _) => lowerAssert(code, message)
      case p.Stmt.Mut(p.Term.Select(root, Nil, _), p.Expr.Alias(source)) if root.symbol == ExceptionWhatSym =>
        if (usesWhat) copyMessage(source) else Nil
      case p.Stmt.Raise(v, exceptionKind, _) =>
        lowerRaise(v, exceptionKind) ::: Option.when(breakOnRaise)(p.Stmt.Break).toList
      case p.Stmt.Rethrow =>
        throw IllegalStateException("rethrow reached structured-exit lowering outside a handler")
      case p.Stmt.Try(b, hs, f)              => lowerTry(b, hs, f)
      case p.Stmt.While(c, b)                => List(p.Stmt.While(c, drainTop(b)))
      case p.Stmt.ForRange(i, lb, ub, st, b) => List(p.Stmt.ForRange(i, lb, ub, st, drainTop(b)))
      case p.Stmt.Cond(c, t, f) =>
        List(p.Stmt.Cond(c, guard(t, canBreak, breakOnRaise), guard(f, canBreak, breakOnRaise)))
      case p.Stmt.Annotated(inner, pos, c) =>
        lower(inner, canBreak, breakOnRaise).map(p.Stmt.Annotated(_, pos, c))
      case other => List(other)
    }

    private def drainTop(body: List[p.Stmt]): List[p.Stmt] = {
      val lowered = guard(body, canBreak = true, breakOnRaise = true)
      if (body.exists(flow.escapes)) p.Stmt.Cond(asserted, List(p.Stmt.Break), Nil) :: lowered else lowered
    }

    def guard(
        stmts: List[p.Stmt],
        canBreak: Boolean = false,
        breakOnRaise: Boolean = false
    ): List[p.Stmt] = stmts match {
      case Nil => Nil
      case s :: rest =>
        val head = lower(s, canBreak, breakOnRaise)
        if (flow.escapes(s)) head ::: fence(rest, canBreak, breakOnRaise)
        else head ::: guard(rest, canBreak, breakOnRaise)
    }

    private def fence(stmts: List[p.Stmt], canBreak: Boolean, breakOnRaise: Boolean): List[p.Stmt] = {
      val (free, rest) = stmts.span(s => !containsBarrier(s))
      rest match {
        case Nil if free.isEmpty => Nil
        case Nil if canBreak =>
          p.Stmt.Cond(asserted, List(p.Stmt.Break), Nil) :: guard(free, canBreak, breakOnRaise)
        case Nil => List(p.Stmt.Cond(asserted, Nil, guard(free, canBreak, breakOnRaise)))
        case barrier :: tail =>
          guard(free, canBreak, breakOnRaise) :::
            lower(barrier, canBreak, breakOnRaise) :::
            fence(tail, canBreak, breakOnRaise)
      }
    }

    // only an escaping raise writes the exception report; assertions already filled the buffer at their site
    def escapeReport: List[p.Stmt] = {
      val code = p.Enums.AssertCode.Exception.value
      val head = (0 until CodeBytes).toList.map { k =>
        p.Stmt.Update(error, p.Term.IntU32Const(k), p.Term.IntS8Const(((code >> (8 * k)) & 0xff).toByte))
      }
      val chain = tags.zipWithIndex.foldRight(List.empty[p.Stmt]) { case ((Raised(exceptionKind, _), i), els) =>
        ifThen(
          p.Expr.IntrOp(p.Intr.LogicEq(tagSel, p.Term.IntS32Const(i + 1))),
          writeMessage(exceptionKind.sourceName),
          els
        )
      }
      if (chain.isEmpty) Nil
      else {
        val cleanup = selectTag(tagSel, tags.map(x => x.thrown -> cleanupFor(x.thrown, slotOf(x.thrown))))
        val report = ifThen(
          p.Expr.IntrOp(p.Intr.LogicNeq(tagSel, p.Term.IntS32Const(AssertTag))),
          head ::: chain ::: cleanup,
          Nil
        )
        List(p.Stmt.Cond(asserted, report, Nil))
      }
    }
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val finallySymbols = FinallySymbols(program.entry.body)
    val e              = program.entry.copy(body = materialiseFinally(program.entry.body, finallySymbols))
    val flow           = Flow(program.defs)
    val touched = e
      .collectFirst_[p.Stmt] {
        case s if isAssert(s)                      => s
        case s @ (_: p.Stmt.Raise | _: p.Stmt.Try) => s
      }
      .isDefined
    if (!touched) program
    else if (e.body.exists(flow.hasBarrierEscape))
      throw RuntimeException(
        "raise escaping a loop that also carries a collective barrier is unsupported: draining the loop would " +
          "skip the barrier and deadlock the workgroup"
      )
    else {
      val tags    = tagTable(e)
      val lower   = Lower(tags, flow, observesExceptionWhat(e.body), finallySymbols)
      val lowered = lower.guard(e.body)
      // the deferred path predicts the #error arg before lowering, so pure try/finally must not change the ABI
      if (tags.isEmpty && !e.body.exists(mayAssert)) program.copy(entry = e.copy(body = lowered))
      else {
        val tagDecls =
          if (tags.isEmpty) Nil
          else
            p.Stmt.Var(p.Named(TagSym, p.Type.IntS32), Some(p.Expr.Alias(p.Term.IntS32Const(AssertTag))), true) ::
              lower.messageDecls :::
              p.Stmt.Var(
                p.Named(ExceptionCodeSym, p.Type.IntS32),
                Some(p.Expr.Alias(p.Term.IntS32Const(0))),
                true
              ) :: lower.slotDecls
        val decls =
          p.Stmt.Var(p.Named(AssertedSym, p.Type.Bool1), Some(p.Expr.Alias(p.Term.Bool1Const(false))), true) ::
            tagDecls ::: lower.assertMessageDecls
        val sentinel = if (e.rtn == p.Type.Unit0) p.Term.Unit0Const else p.Term.Poison(e.rtn)
        val exit     = p.Stmt.Return(p.Expr.Alias(sentinel))
        val epilogue =
          if (e.body.exists(flow.escapes)) lower.assertMessageEpilogue ::: lower.escapeReport ::: List(exit) else Nil
        log.info(s"${e.signatureRepr}: lowered ${tags.size} raised type(s) to a structured drain + error buffer")
        val needsError = e.body.exists(mayAssert) || e.body.exists(flow.escapes)
        val args =
          if (!needsError || e.args.exists(_.named.symbol == ErrorSym)) e.args
          else p.Arg(p.Named(ErrorSym, ErrorPtr)) +: e.args
        program.copy(
          entry = e.copy(decl = e.decl.remapArgs(args), body = decls ::: lowered ::: epilogue)
        )
      }
    }
  }

  private[pass] def lowerHandledFunction(f: p.Function, defs: List[p.StructDef], log: Log): p.Function = {
    val lowered = apply(p.Program(f, Nil, defs), log).entry
    if (lowered.args != f.args)
      throw IllegalStateException(s"RecursionLower: an exception escapes recursive function ${f.name.repr}")
    lowered
  }

}
