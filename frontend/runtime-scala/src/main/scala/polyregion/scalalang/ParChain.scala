package polyregion.scalalang

import scala.collection.mutable
import scala.quoted.*

trait ParChain {
  val q: Quotes
  import q.reflect.*
  given Quotes = q

  enum StageKind { case Map, Filter }

  case class Stage(kind: StageKind, fn: Term, to: TypeRepr)

  enum Source {
    case Elements(array: Term, elem: TypeRepr)
    case Indices(range: Term)
  }

  case class Chain(source: Source, stages: List[Stage], context: Term, bindings: List[Definition]) {
    def elem: TypeRepr = stages.lastOption.fold(source match {
      case Source.Elements(_, elem) => elem
      case Source.Indices(_)        => TypeRepr.of[Int]
    })(_.to)
  }

  private object Call {
    def unapply(term: Term): Option[(Symbol, List[TypeTree], List[Term])] = term match {
      case Apply(TypeApply(fn, targs), args) => Some((fn.symbol, targs, args))
      case Apply(fn, args)                   => Some((fn.symbol, Nil, args))
      case _                                 => None
    }
  }

  def decode(chain: Term): Chain = {
    val module   = Symbol.requiredModule("polyregion.par.stages")
    val array    = module.declaredMethod("array").head
    val range    = module.declaredMethod("range").head
    val on       = module.declaredMethod("on").head
    val map      = module.declaredMethod("map").head
    val filter   = module.declaredMethod("filter").head
    val bindings = mutable.ListBuffer.empty[Definition]
    val contexts = mutable.ListBuffer.empty[Term]

    def go(term: Term): (Source, List[Stage], Term) = term match {
      case Inlined(_, bs, body) => bindings ++= bs; go(body)
      case Block(stats, body) =>
        stats.foreach {
          case definition: Definition => bindings += definition
          case other                  => unsupported(other)
        }
        go(body)
      case Typed(body, _)                                      => go(body)
      case Call(sym, targs, List(xs, context)) if sym == array => (Source.Elements(xs, targs.head.tpe), Nil, context)
      case Call(sym, _, List(xs, context)) if sym == range     => (Source.Indices(xs), Nil, context)
      case Call(sym, _, List(prev, context)) if sym == on =>
        contexts += context
        if (contexts.size > 1)
          report.errorAndAbort("a .par chain selects its context once", context.pos)
        val (source, stages, _) = go(prev)
        (source, stages, context)
      case Call(sym, targs, List(prev, fn)) if sym == map =>
        val (source, stages, context) = go(prev)
        (source, stages :+ Stage(StageKind.Map, fn, targs(1).tpe), context)
      case Call(sym, targs, List(prev, fn)) if sym == filter =>
        val (source, stages, context) = go(prev)
        (source, stages :+ Stage(StageKind.Filter, fn, targs.head.tpe), context)
      case other => unsupported(other)
    }

    def unsupported(tree: Tree): Nothing =
      report.errorAndAbort(
        s"a .par chain must be a single expression ending in a terminal operation, found `${tree.show}`",
        tree.pos
      )

    val (source, stages, context) = go(chain)
    Chain(source, stages, context, bindings.toList)
  }
}
