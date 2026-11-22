package polyregion.scalalang

import polyregion.jvm.runtime as rt

import scala.annotation.tailrec
import scala.quoted.*

object ParLowering {

  object Spectra {
    val Transform: String                = "spectra:spectra.transform"
    val Tabulate: String                 = "spectra:spectra.tabulate"
    val CopyIf: String                   = "spectra:spectra.copy_if"
    val TransformReduce: String          = "spectra:spectra.transform_reduce"
    val TransformReduceUnordered: String = "spectra:spectra.transform_reduce_unordered"
    val AnyOf: String                    = "spectra:spectra.any_of"
    val AllOf: String                    = "spectra:spectra.all_of"
    val CountIf: String                  = "spectra:spectra.count_if"
    val Sort: String                     = "spectra:spectra.sort"
    val StableSort: String               = "spectra:spectra.stable_sort"
  }

  def toArray[T: Type](xs: Expr[Any])(using q: Quotes): Expr[Array[T]]    = Lowerer(q).toArray[T](xs)
  def foreach(xs: Expr[Any], f: Expr[Any])(using q: Quotes): Expr[Unit]   = Lowerer(q).foreach(xs, f)
  def sum[T: Type](xs: Expr[Any])(using q: Quotes): Expr[T]               = Lowerer(q).sum[T](xs)
  def product[T: Type](xs: Expr[Any])(using q: Quotes): Expr[T]           = Lowerer(q).product[T](xs)
  def min[T: Type](xs: Expr[Any])(using q: Quotes): Expr[T]               = Lowerer(q).extremum[T](xs, "min")
  def max[T: Type](xs: Expr[Any])(using q: Quotes): Expr[T]               = Lowerer(q).extremum[T](xs, "max")
  def size(xs: Expr[Any])(using q: Quotes): Expr[Int]                     = Lowerer(q).size(xs)
  def count(xs: Expr[Any], f: Expr[Any])(using q: Quotes): Expr[Int]      = Lowerer(q).count(xs, f)
  def exists(xs: Expr[Any], f: Expr[Any])(using q: Quotes): Expr[Boolean] = Lowerer(q).exists(xs, f)
  def forall(xs: Expr[Any], f: Expr[Any])(using q: Quotes): Expr[Boolean] = Lowerer(q).forall(xs, f)
  def fold[T: Type](xs: Expr[Any], z: Expr[T], op: Expr[Any])(using q: Quotes): Expr[T] = Lowerer(q).fold[T](xs, z, op)
  def reduce[T: Type](xs: Expr[Any], op: Expr[Any])(using q: Quotes): Expr[T]           = Lowerer(q).reduce[T](xs, op)
  def sorted[T: Type](xs: Expr[Any])(using q: Quotes): Expr[Array[T]]                   = Lowerer(q).sorted[T](xs, None)
  def sortWith[T: Type](xs: Expr[Any], lt: Expr[Any])(using q: Quotes): Expr[Array[T]] =
    Lowerer(q).sorted[T](xs, Some(lt))
}

class Lowerer(val q: Quotes) extends ParChain {
  import q.reflect.*

  private enum Order {
    case Ordered, Unordered
  }

  private enum Bound {
    case Held(array: Term, elem: TypeRepr)
    case Indexed(start: Term, step: Term)
  }

  private case class Frame(source: Bound, length: Term, context: Term, stages: List[Stage], elem: TypeRepr) {
    def sourceElem: TypeRepr = source match {
      case Bound.Held(_, tpe)  => tpe
      case Bound.Indexed(_, _) => TypeRepr.of[Int]
    }
    def filtered: Boolean = stages.exists(_.kind == StageKind.Filter)
    def maps: List[Stage] = stages.filter(_.kind == StageKind.Map)
  }

  private lazy val primitives = List(
    TypeRepr.of[Int],
    TypeRepr.of[Long],
    TypeRepr.of[Float],
    TypeRepr.of[Double],
    TypeRepr.of[Short],
    TypeRepr.of[Byte],
    TypeRepr.of[Char],
    TypeRepr.of[Boolean]
  )

  private lazy val arithmetic = primitives.take(4)

  private def fail(message: String): Nothing = report.errorAndAbort(message, Position.ofMacroExpansion)

  private def is(tpe: TypeRepr, of: List[TypeRepr]): Boolean = of.exists(_ =:= tpe.widen.dealias)

  private def stored(elem: TypeRepr): Unit =
    if (!is(elem, arithmetic))
      fail(
        s".par arrays hold Int, Long, Float or Double elements (Spectra implements 4 and 8 byte elements), found ${elem.show}"
      )

  private def bind(xs: Expr[Any])(k: Frame => Term): Term = {
    val chain = decode(xs.asTerm)
    chain.source match {
      case Source.Elements(_, elem) => stored(elem)
      case _                        =>
    }
    chain.stages.foreach { stage =>
      if (!is(stage.to, primitives))
        fail(
          s".par supports elements of Int, Long, Float, Double, Short, Byte, Char and Boolean, found ${stage.to.show}"
        )
    }
    val owner = Symbol.spliceOwner
    val body = chain.source match {
      case Source.Elements(array, elem) =>
        ValDef.let(owner, "source", array) { a =>
          ValDef.let(owner, "context", chain.context) { ctx =>
            ValDef.let(owner, "length", Select.unique(a, "length")) { n =>
              k(Frame(Bound.Held(a, elem), n, ctx, chain.stages, chain.elem))
            }
          }
        }
      case Source.Indices(range) =>
        ValDef.let(owner, "range", range) { r =>
          ValDef.let(owner, "context", chain.context) { ctx =>
            ValDef.let(owner, "start", Select.unique(r, "start")) { start =>
              ValDef.let(owner, "step", Select.unique(r, "step")) { step =>
                ValDef.let(owner, "length", Select.unique(r, "length")) { n =>
                  k(Frame(Bound.Indexed(start, step), n, ctx, chain.stages, chain.elem))
                }
              }
            }
          }
        }
    }
    if (chain.bindings.isEmpty) body else Block(chain.bindings, body)
  }

  @tailrec
  private def stripped(term: Term): Term = term match {
    case Inlined(_, Nil, body) => stripped(body)
    case Typed(body, _)        => stripped(body)
    case other                 => other
  }

  private def applied(f: Term, args: Term*): Term =
    Expr.betaReduce(Select.unique(stripped(f), "apply").appliedToArgs(args.toList).asExprOf[Any]).asTerm

  private def lambda(params: List[(String, TypeRepr)], result: TypeRepr)(body: (Symbol, List[Term]) => Term): Term =
    Lambda(
      Symbol.spliceOwner,
      MethodType(params.map(_._1))(_ => params.map(_._2), _ => result),
      (owner, args) => body(owner, args.collect { case term: Term => term }).changeOwner(owner)
    )

  private def literal(tpe: TypeRepr, value: Int): Term =
    if (tpe =:= TypeRepr.of[Int]) Literal(IntConstant(value))
    else if (tpe =:= TypeRepr.of[Long]) Literal(LongConstant(value))
    else if (tpe =:= TypeRepr.of[Float]) Literal(FloatConstant(value.toFloat))
    else if (tpe =:= TypeRepr.of[Double]) Literal(DoubleConstant(value))
    else if (tpe =:= TypeRepr.of[Short]) Literal(ShortConstant(value.toShort))
    else if (tpe =:= TypeRepr.of[Byte]) Literal(ByteConstant(value.toByte))
    else if (tpe =:= TypeRepr.of[Char]) Literal(CharConstant(value.toChar))
    else Literal(BooleanConstant(value != 0))

  private def infix(a: Term, name: String, b: Term): Term = Select.overloaded(a, name, Nil, List(b))

  private def lessThan(tpe: TypeRepr, a: Term, b: Term): Term =
    if (is(tpe, List(TypeRepr.of[Float], TypeRepr.of[Double]))) {
      val no       = Literal(BooleanConstant(false))
      val yes      = Literal(BooleanConstant(true))
      val one      = literal(tpe, 1)
      val negative = (x: Term) => infix(infix(one, "/", x), "<", literal(tpe, 0))
      If(
        infix(a, "<", b),
        yes,
        If(
          infix(a, ">", b),
          no,
          If(
            infix(b, "!=", b),
            infix(a, "==", a),
            If(infix(a, "!=", a), no, If(negative(a), Select.unique(negative(b), "unary_!"), no))
          )
        )
      )
    } else infix(a, "<", b)

  private def combine(elem: TypeRepr, result: TypeRepr)(op: (Term, Term) => Term): Term =
    lambda(List("a" -> elem, "b" -> elem), result)((_, xs) => op(xs(0), xs(1)))

  private def combine(elem: TypeRepr)(op: (Term, Term) => Term): Term = combine(elem, elem)(op)

  private def program(stages: List[Stage], owner: Symbol)(x: Term, leaf: Term => Term, otherwise: => Term): Term =
    stages match {
      case Nil => leaf(x)
      case Stage(StageKind.Map, f, _) :: rest =>
        ValDef.let(owner, "mapped", applied(f, x))(y => program(rest, owner)(y, leaf, otherwise))
      case Stage(StageKind.Filter, p, _) :: rest =>
        If(applied(p, x), program(rest, owner)(x, leaf, otherwise), otherwise)
    }

  private def elementAt(source: Bound, index: Term): Term = source match {
    case Bound.Held(array, _)       => Apply(Select.unique(array, "apply"), List(index))
    case Bound.Indexed(start, step) => infix(start, "+", infix(index, "*", step))
  }

  private def over(frame: Frame, result: TypeRepr)(leaf: Term => Term, otherwise: => Term): Term =
    lambda(List("x" -> frame.sourceElem), result)((owner, xs) => program(frame.stages, owner)(xs.head, leaf, otherwise))

  private def overIndex(frame: Frame, result: TypeRepr, stages: List[Stage])(
      leaf: Term => Term,
      otherwise: => Term
  ): Term =
    lambda(List("i" -> TypeRepr.of[Long]), result) { (owner, is) =>
      ValDef.let(owner, "x", elementAt(frame.source, Select.unique(is.head, "toInt")))(x =>
        program(stages, owner)(x, leaf, otherwise)
      )
    }

  private def call[R: Type](id: String, ctx: Term, args: List[Term]): Term =
    '{
      polyregion.scalalang.launch
        .call[R](${ Expr(id) }, ${ ctx.asExprOf[rt.Context] }, ${ Varargs(args.map(_.asExprOf[Any])) }*)
    }.asTerm

  private def coreForEach(ctx: Term, n: Term, body: Term): Term =
    '{
      polyregion.offload.for_each(
        ${ ctx.asExprOf[rt.Context] },
        ${ n.asExprOf[Long] },
        ${ body.asExprOf[Long => Unit] }
      )
    }.asTerm

  private def coreReduce[R: Type](order: Order, ctx: Term, n: Term, init: Term, map: Term, op: Term): Term =
    if (order == Order.Ordered)
      '{
        polyregion.offload.transform_reduce[R](
          ${ ctx.asExprOf[rt.Context] },
          ${ n.asExprOf[Long] },
          ${ init.asExprOf[R] },
          ${ map.asExprOf[Long => R] },
          ${ op.asExprOf[(R, R) => R] }
        )
      }.asTerm
    else
      '{
        polyregion.offload.transform_reduce_unordered[R](
          ${ ctx.asExprOf[rt.Context] },
          ${ n.asExprOf[Long] },
          ${ init.asExprOf[R] },
          ${ map.asExprOf[Long => R] },
          ${ op.asExprOf[(R, R) => R] }
        )
      }.asTerm

  private def newArray(elem: TypeRepr, n: Term): Term = elem.asType match {
    case '[t] =>
      '{ Array.ofDim[t](${ n.asExprOf[Int] })(using ${ Expr.summon[scala.reflect.ClassTag[t]].get }) }.asTerm
  }

  private def take(elem: TypeRepr, array: Term, n: Term): Term = elem.asType match {
    case '[t] => '{ ${ array.asExprOf[Array[t]] }.take(${ n.asExprOf[Int] }) }.asTerm
  }

  private def whenPositive[R: Type](n: Term, otherwise: Term)(body: => Term): Term =
    '{ if (${ n.asExprOf[Int] } > 0) ${ body.asExprOf[R] } else ${ otherwise.asExprOf[R] } }.asTerm

  private def unit: Term = Literal(UnitConstant())

  private def materialise(frame: Frame)(k: (Term, Term) => Term): Term = {
    val owner = Symbol.spliceOwner
    val ctx   = frame.context
    val n     = frame.length
    val value = (frame.maps, frame.sourceElem) match {
      case (Nil, elem) => elem
      case (maps, _)   => maps.last.to
    }
    (frame.source, frame.stages) match {
      case (Bound.Held(array, _), Nil) =>
        ValDef.let(owner, "copy", Select.unique(array, "clone").appliedToNone)(a => k(a, n))
      case _ if !frame.filtered =>
        ValDef.let(owner, "out", newArray(value, n)) { out =>
          val launch = frame.source match {
            case Bound.Held(array, _) =>
              call[Unit](
                ParLowering.Spectra.Transform,
                ctx,
                List(array, out, n, over(frame, value)(identity, literal(value, 0)))
              )
            case Bound.Indexed(_, _) =>
              call[Unit](
                ParLowering.Spectra.Tabulate,
                ctx,
                List(
                  out,
                  n,
                  lambda(List("i" -> TypeRepr.of[Int]), value) { (o, is) =>
                    ValDef.let(o, "x", elementAt(frame.source, is.head))(x =>
                      program(frame.stages, o)(x, identity, literal(value, 0))
                    )
                  }
                )
              )
          }
          Block(
            List('{ if (${ n.asExprOf[Int] } > 0) ${ launch.asExprOf[Unit] } }.asTerm),
            k(out, n)
          )
        }
      case _ =>
        val held = frame.source match {
          case Bound.Held(array, _) => (a: Term => Term) => a(array)
          case Bound.Indexed(_, _) =>
            (cont: Term => Term) =>
              ValDef.let(owner, "indices", newArray(TypeRepr.of[Int], n)) { indices =>
                Block(
                  List(
                    '{
                      if (${ n.asExprOf[Int] } > 0) ${
                        call[Unit](
                          ParLowering.Spectra.Tabulate,
                          ctx,
                          List(
                            indices,
                            n,
                            lambda(List("i" -> TypeRepr.of[Int]), TypeRepr.of[Int])((_, is) =>
                              elementAt(frame.source, is.head)
                            )
                          )
                        ).asExprOf[Unit]
                      }
                    }.asTerm
                  ),
                  cont(indices)
                )
              }
        }
        held { src =>
          val elem = frame.sourceElem
          val validity = lambda(List("x" -> elem), TypeRepr.of[Boolean])((o, xs) =>
            program(frame.stages, o)(xs.head, _ => Literal(BooleanConstant(true)), Literal(BooleanConstant(false)))
          )
          ValDef.let(owner, "survivors", newArray(elem, n)) { tmp =>
            ValDef.let(
              owner,
              "kept",
              whenPositive[Int](n, Literal(IntConstant(0)))(
                call[Int](ParLowering.Spectra.CopyIf, ctx, List(src, n, tmp, validity))
              )
            ) { m =>
              if (frame.maps.isEmpty) ValDef.let(owner, "trimmed", take(elem, tmp, m))(trimmed => k(trimmed, m))
              else
                ValDef.let(owner, "out", newArray(value, m)) { out =>
                  val transformed = call[Unit](
                    ParLowering.Spectra.Transform,
                    ctx,
                    List(
                      tmp,
                      out,
                      m,
                      lambda(List("x" -> elem), value)((o, xs) =>
                        program(frame.maps, o)(xs.head, identity, literal(value, 0))
                      )
                    )
                  )
                  Block(List('{ if (${ m.asExprOf[Int] } > 0) ${ transformed.asExprOf[Unit] } }.asTerm), k(out, m))
                }
            }
          }
        }
    }
  }

  def toArray[T: Type](xs: Expr[Any]): Expr[Array[T]] = {
    stored(TypeRepr.of[T])
    bind(xs)(frame => materialise(frame)((array, _) => array)).asExprOf[Array[T]]
  }

  def foreach(xs: Expr[Any], f: Expr[Any]): Expr[Unit] =
    bind(xs) { frame =>
      val body = overIndex(frame, TypeRepr.of[Unit], frame.stages)(y => applied(f.asTerm, y), unit)
      whenPositive[Unit](frame.length, unit)(coreForEach(frame.context, Select.unique(frame.length, "toLong"), body))
    }.asExprOf[Unit]

  private def reduceNeutral[R: Type](order: Order, frame: Frame, neutral: Term, op: Term, leaf: Term => Term): Term =
    whenPositive[R](frame.length, neutral) {
      frame.source match {
        case Bound.Held(array, _) =>
          call[R](
            if (order == Order.Ordered) ParLowering.Spectra.TransformReduce
            else ParLowering.Spectra.TransformReduceUnordered,
            frame.context,
            List(array, frame.length, neutral, over(frame, TypeRepr.of[R])(leaf, neutral), op)
          )
        case Bound.Indexed(_, _) =>
          coreReduce[R](
            order,
            frame.context,
            Select.unique(frame.length, "toLong"),
            neutral,
            overIndex(frame, TypeRepr.of[R], frame.stages)(leaf, neutral),
            op
          )
      }
    }

  private def numeric[T: Type](what: String): TypeRepr = {
    val tpe = TypeRepr.of[T]
    if (!is(tpe, arithmetic)) fail(s"$what is only defined for Int, Long, Float and Double, found ${tpe.show}")
    tpe
  }

  private def arithmetic[T: Type](xs: Expr[Any], what: String, neutral: Int, op: String): Expr[T] = {
    val tpe = numeric[T](what)
    bind(xs)(frame =>
      reduceNeutral[T](Order.Unordered, frame, literal(tpe, neutral), combine(tpe)(infix(_, op, _)), identity)
    )
      .asExprOf[T]
  }

  def sum[T: Type](xs: Expr[Any]): Expr[T] = arithmetic[T](xs, "sum", 0, "+")

  def product[T: Type](xs: Expr[Any]): Expr[T] = arithmetic[T](xs, "product", 1, "*")

  def fold[T: Type](xs: Expr[Any], z: Expr[T], op: Expr[Any]): Expr[T] =
    bind(xs)(frame =>
      ValDef.let(Symbol.spliceOwner, "neutral", z.asTerm)(neutral =>
        reduceNeutral[T](Order.Ordered, frame, neutral, stripped(op.asTerm), identity)
      )
    ).asExprOf[T]

  private def reduceFirst[R: Type](order: Order, frame: Frame, op: Term, what: String): Term = {
    val owner = Symbol.spliceOwner
    '{
      if (${ frame.length.asExprOf[Int] } == 0) throw UnsupportedOperationException(${ Expr(s"empty.$what") })
      else
        ${
          val first = program(frame.stages, owner)(
            elementAt(frame.source, Literal(IntConstant(0))),
            identity,
            literal(TypeRepr.of[R], 0)
          )
          ValDef
            .let(owner, "first", first) { head =>
              '{
                if (${ frame.length.asExprOf[Int] } == 1) ${ head.asExprOf[R] }
                else
                  ${
                    coreReduce[R](
                      order,
                      frame.context,
                      infix(Select.unique(frame.length, "toLong"), "-", Literal(LongConstant(1))),
                      head,
                      lambda(List("i" -> TypeRepr.of[Long]), TypeRepr.of[R]) { (o, is) =>
                        ValDef.let(
                          o,
                          "x",
                          elementAt(frame.source, infix(Select.unique(is.head, "toInt"), "+", Literal(IntConstant(1))))
                        )(x => program(frame.stages, o)(x, identity, literal(TypeRepr.of[R], 0)))
                      },
                      op
                    ).asExprOf[R]
                  }
              }.asTerm
            }
            .asExprOf[R]
        }
    }.asTerm
  }

  private def reduceAny[R: Type](order: Order, xs: Expr[Any], op: Term, what: String): Term =
    bind(xs) { frame =>
      if (!frame.filtered) reduceFirst[R](order, frame, op, what)
      else
        materialise(frame) { (array, m) =>
          reduceFirst[R](order, Frame(Bound.Held(array, frame.elem), m, frame.context, Nil, frame.elem), op, what)
        }
    }

  def reduce[T: Type](xs: Expr[Any], op: Expr[Any]): Expr[T] =
    reduceAny[T](Order.Ordered, xs, stripped(op.asTerm), "reduceLeft").asExprOf[T]

  def extremum[T: Type](xs: Expr[Any], what: String): Expr[T] = {
    val tpe = numeric[T](what)
    val pick = combine(tpe) { (a, b) =>
      if (what == "min") If(lessThan(tpe, b, a), b, a) else If(lessThan(tpe, a, b), b, a)
    }
    reduceAny[T](Order.Unordered, xs, pick, what).asExprOf[T]
  }

  private def counted(xs: Expr[Any], pred: Term => Term): Term =
    bind(xs) { frame =>
      frame.source match {
        case Bound.Held(array, _) =>
          whenPositive[Int](frame.length, Literal(IntConstant(0)))(
            call[Int](
              ParLowering.Spectra.CountIf,
              frame.context,
              List(array, frame.length, over(frame, TypeRepr.of[Boolean])(pred, Literal(BooleanConstant(false))))
            )
          )
        case Bound.Indexed(_, _) =>
          reduceNeutral[Int](
            Order.Unordered,
            frame,
            Literal(IntConstant(0)),
            combine(TypeRepr.of[Int])(infix(_, "+", _)),
            y => If(pred(y), Literal(IntConstant(1)), Literal(IntConstant(0)))
          )
      }
    }

  def size(xs: Expr[Any]): Expr[Int] = {
    val filtered = decode(xs.asTerm).stages.exists(_.kind == StageKind.Filter)
    if (filtered) counted(xs, _ => Literal(BooleanConstant(true))).asExprOf[Int]
    else bind(xs)(_.length).asExprOf[Int]
  }

  def count(xs: Expr[Any], f: Expr[Any]): Expr[Int] =
    counted(xs, applied(f.asTerm, _)).asExprOf[Int]

  private def quantified(xs: Expr[Any], f: Expr[Any], all: Boolean): Expr[Boolean] =
    bind(xs) { frame =>
      val vacuous = Literal(BooleanConstant(all))
      frame.source match {
        case Bound.Held(array, _) =>
          whenPositive[Boolean](frame.length, vacuous)(
            call[Boolean](
              if (all) ParLowering.Spectra.AllOf else ParLowering.Spectra.AnyOf,
              frame.context,
              List(array, frame.length, over(frame, TypeRepr.of[Boolean])(applied(f.asTerm, _), vacuous))
            )
          )
        case Bound.Indexed(_, _) =>
          val witness = literal(TypeRepr.of[Int], if (all) 0 else 1)
          val other   = literal(TypeRepr.of[Int], if (all) 1 else 0)
          infix(
            reduceNeutral[Int](
              Order.Unordered,
              frame,
              literal(TypeRepr.of[Int], 0),
              combine(TypeRepr.of[Int])(infix(_, "+", _)),
              y => If(applied(f.asTerm, y), witness, other)
            ),
            if (all) "==" else ">",
            literal(TypeRepr.of[Int], 0)
          )
      }
    }.asExprOf[Boolean]

  def exists(xs: Expr[Any], f: Expr[Any]): Expr[Boolean] = quantified(xs, f, all = false)

  def forall(xs: Expr[Any], f: Expr[Any]): Expr[Boolean] = quantified(xs, f, all = true)

  def sorted[T: Type](xs: Expr[Any], lt: Option[Expr[Any]]): Expr[Array[T]] = {
    val tpe = TypeRepr.of[T]
    stored(tpe)
    bind(xs) { frame =>
      materialise(frame) { (array, m) =>
        val order = lt.fold(combine(tpe, TypeRepr.of[Boolean])(lessThan(tpe, _, _)))(f => stripped(f.asTerm))
        val sort  = if (lt.isDefined) ParLowering.Spectra.StableSort else ParLowering.Spectra.Sort
        Block(
          List('{
            if (${ m.asExprOf[Int] } > 1)
              ${ call[Unit](sort, frame.context, List(array, m, order)).asExprOf[Unit] }
          }.asTerm),
          array
        )
      }
    }.asExprOf[Array[T]]
  }
}
