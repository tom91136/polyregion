package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.Tree
import com.sun.tools.javac.code.{Flags, Symbol as JSymbol, Symtab, Type as JType, TypeTag, Types}
import com.sun.tools.javac.processing.JavacProcessingEnvironment
import com.sun.tools.javac.tree.{JCTree, TreeMaker, TreeScanner}
import com.sun.tools.javac.util.{List as JList, Names, Position}
import polyregion.ast.*
import polyregion.javalang.LaunchCore.{Mode, Prim}
import polyregion.javalang.LaunchSites.StepArg
import polyregion.javalang.StreamChain.*
import polyregion.scalalang.LaunchConfig

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object StreamSites {

  private val Spectra = "spectra:spectra."
  private val Core    = "polyregion.core:polyregion.core."

  private val OpsClass     = "polyregion.javalang.Ops"
  private val StreamsClass = "polyregion.javalang.Streams"

  case class Site(decoded: Decoded, within: Option[JSymbol.MethodSymbol], home: JSymbol.ClassSymbol)

  private class Finder(home: JSymbol.ClassSymbol, within: Option[JSymbol.MethodSymbol], sites: mutable.ListBuffer[Site])
      extends TreeScanner {

    override def visitClassDef(tree: JCTree.JCClassDecl): Unit = ()

    override def visitApply(a: JCTree.JCMethodInvocation): Unit =
      StreamChain.decode(a) match {
        case Some(decoded) =>
          sites += Site(decoded, within, home)
          decoded.chain.foreach { chain =>
            chain.lams.flatMap(_.args).foreach(_.accept(this))
            chain.source match {
              case Source.Array(array, _)         => array.accept(this)
              case Source.Range(lo = lo, hi = hi) => lo.accept(this); hi.accept(this)
            }
            chain.context.foreach(_.accept(this))
          }
          if (decoded.chain.isLeft) super.visitApply(a)
        case None => super.visitApply(a)
      }
  }

  def collect(scope: Tree): List[Site] = scope match {
    case cls: JCTree.JCClassDecl =>
      val sites = mutable.ListBuffer.empty[Site]
      cls.defs.asScala.foreach { member =>
        val within = member match {
          case m: JCTree.JCMethodDecl => Option(m.sym)
          case _                      => None
        }
        member.accept(Finder(cls.sym, within, sites))
      }
      sites.toList
    case _ => Nil
  }

  private enum Leaf {
    case Identity
    case Widen
    case One
    case True
    case Bit(lam: Lam)
    case NotBit(lam: Lam)
    case Consume(lam: Lam)
    case Accumulate(
        high: JSymbol.VarSymbol,
        compensation: JSymbol.VarSymbol,
        simple: JSymbol.VarSymbol,
        kept: JSymbol.VarSymbol
    )
  }

  private enum Order {
    case Ordered, Unordered
  }

  private enum Origin {
    case Held(kind: Char)
    case Counted(kind: Char)
    case Iota
  }

  private enum Program {
    case Element(kind: Char, stages: List[Stage], leaf: Leaf, ret: Char, neutral: Option[Char] = None)
    case Index(index: Char, origin: Origin, stages: List[Stage], leaf: Leaf, ret: Char, neutral: Option[Char] = None)
    case Binary(lam: Lam, kind: Char, ret: Char, neutral: Option[Char] = None)
    case Blocks(kind: Char, held: Boolean, stages: List[Stage])
  }

  private enum Arg {
    case Scalar(kind: Char)
    case Array(kind: Char)
    case Fn(program: StreamSites.Program)
    case Forward(name: String)
  }

  private case class Step(role: String, identity: String, args: List[Arg], result: Option[Char])

  private class Synth(env: JavacProcessingEnvironment) {
    private val context = env.getContext
    val make: TreeMaker = TreeMaker.instance(context).at(Position.NOPOS)
    val names: Names    = Names.instance(context)
    val syms: Symtab    = Symtab.instance(context)
    val types: Types    = Types.instance(context)

    def tpe(kind: Char): JType = kind match {
      case 'I' => syms.intType
      case 'J' => syms.longType
      case 'D' => syms.doubleType
      case 'F' => syms.floatType
      case 'Z' => syms.booleanType
      case 'V' => syms.voidType
    }

    def array(kind: Char): JType = JType.ArrayType(tpe(kind), syms.arrayClass)

    def classOf(name: String): Result[JSymbol.ClassSymbol] =
      Option(env.getElementUtils.getTypeElement(name))
        .collect { case c: JSymbol.ClassSymbol => c }
        .toRight(
          CompilerException(s"`$name` is not on the compile classpath")
        )

    def method(
        cls: String,
        name: String,
        test: JSymbol.MethodSymbol => Boolean = _ => true
    ): Result[JSymbol.MethodSymbol] =
      classOf(cls).flatMap(
        _.members()
          .getSymbolsByName(names.fromString(name))
          .asScala
          .collectFirst { case m: JSymbol.MethodSymbol if test(m) => m }
          .toRight(CompilerException(s"`$cls.$name` is not on the compile classpath"))
      )

    def call(m: JSymbol.MethodSymbol, args: List[JCTree.JCExpression]): JCTree.JCExpression =
      make.App(make.QualIdent(m), JList.from(args.asJava))

    def literal(kind: Char, value: Long): JCTree.JCExpression = kind match {
      case 'I' => make.Literal(Int.box(value.toInt))
      case 'J' => make.Literal(Long.box(value))
      case 'D' => make.Literal(Double.box(value.toDouble))
      case 'Z' => make.Literal(TypeTag.BOOLEAN, Int.box(value.toInt))
    }
  }

  private case class Frame(
      synth: Synth,
      owner: JSymbol.ClassSymbol,
      chain: Chain,
      ops: String => Result[JSymbol.MethodSymbol]
  ) {
    val lams: List[Lam] = chain.lams

    def gTypes: List[JType] = lams.flatMap { lam =>
      lam.impl.`type`.getParameterTypes.asScala.toList.take(lam.args.size)
    }

    def offsetOf(lam: Lam): Int =
      lams.takeWhile(_ ne lam).map(_.args.size).sum

    def originType: JType = chain.source match {
      case Source.Array(_, kind)  => synth.array(kind)
      case Source.Range(elem = k) => synth.tpe(k)
    }
  }

  private def suffix(kind: Char): String = kind.toString

  private def build(
      frame: Frame,
      name: String,
      originType: JType,
      neutralType: JType,
      trailing: List[JType],
      ret: JType,
      extra: List[JType] = Nil
  )(
      body: (
          JSymbol.MethodSymbol,
          List[JCTree.JCExpression],
          List[JCTree.JCExpression],
          List[JCTree.JCExpression]
      ) => Result[List[JCTree.JCStatement]]
  ): Result[(JCTree.JCMethodDecl, JSymbol.MethodSymbol, Int)] = {
    val synth  = frame.synth
    val prefix = originType :: neutralType :: frame.gTypes ::: extra
    val all    = prefix ::: trailing
    val msym = JSymbol.MethodSymbol(
      Flags.STATIC | Flags.SYNTHETIC,
      synth.names.fromString(name),
      JType.MethodType(JList.from(all.asJava), ret, JList.nil[JType](), synth.syms.methodClass),
      frame.owner
    )
    val params = all.zipWithIndex.map((t, i) => synth.make.Param(synth.names.fromString(s"p$i"), t, msym))
    msym.params = JList.from(params.map(_.sym).asJava)
    val idents      = params.map(p => synth.make.Ident(p.sym): JCTree.JCExpression)
    val (pre, post) = idents.splitAt(prefix.size)
    body(msym, pre.take(2), pre.drop(2), post).map { stmts =>
      (synth.make.MethodDef(msym, synth.make.Block(0, JList.from(stmts.asJava))), msym, prefix.size)
    }
  }

  private def applyLam(frame: Frame, lam: Lam, g: List[JCTree.JCExpression], args: List[JCTree.JCExpression]) = {
    val offset = frame.offsetOf(lam)
    frame.synth.call(lam.impl, g.slice(offset, offset + lam.args.size) ::: args)
  }

  private def program(
      frame: Frame,
      name: String,
      program: Program
  ): Result[(JCTree.JCMethodDecl, JSymbol.MethodSymbol, Int)] = {
    val synth = frame.synth
    import synth.make
    program match {
      case Program.Binary(lam, kind, ret, neutral) =>
        build(
          frame,
          name,
          frame.originType,
          synth.tpe(neutral.getOrElse(Streams.NO_NEUTRAL)),
          List(synth.tpe(kind), synth.tpe(kind)),
          synth.tpe(ret)
        ) { (_, _, g, args) =>
          List(make.Return(applyLam(frame, lam, g, args)): JCTree.JCStatement).success
        }
      case Program.Element(kind, stages, leaf, ret, neutral) =>
        build(
          frame,
          name,
          frame.originType,
          synth.tpe(neutral.getOrElse(Streams.NO_NEUTRAL)),
          List(synth.tpe(kind)),
          synth.tpe(ret)
        ) { (msym, sn, g, args) =>
          body(frame, msym, sn, g, args.head, stages, leaf)
        }
      case Program.Index(index, origin, stages, leaf, ret, neutral) =>
        build(
          frame,
          name,
          frame.originType,
          synth.tpe(neutral.getOrElse(Streams.NO_NEUTRAL)),
          List(synth.tpe(index)),
          synth.tpe(ret)
        ) { (msym, sn, g, args) =>
          val i = args.head
          val entry: Result[(Option[JCTree.JCStatement], JCTree.JCExpression)] = origin match {
            case Origin.Iota => (None, i).success
            case Origin.Held(kind) =>
              frame.ops(s"get${suffix(kind)}").map { get =>
                val x = JSymbol.VarSymbol(0L, synth.names.fromString("x"), synth.tpe(kind), msym)
                (Some(make.VarDef(x, synth.call(get, List(sn.head, i)))), make.Ident(x))
              }
            case Origin.Counted(kind) =>
              frame.ops(s"at${suffix(kind)}${suffix(index)}").map { at =>
                val x = JSymbol.VarSymbol(0L, synth.names.fromString("x"), synth.tpe(kind), msym)
                (Some(make.VarDef(x, synth.call(at, List(sn.head, i)))), make.Ident(x))
              }
          }
          entry.flatMap((decl, x) => body(frame, msym, sn, g, x, stages, leaf).map(decl.toList ::: _))
        }
      case Program.Blocks(kind, held, stages) =>
        build(
          frame,
          name,
          frame.originType,
          synth.tpe('J'),
          List(synth.tpe('J')),
          synth.tpe('V'),
          List(synth.array('D'))
        ) { (msym, sn, gOut, args) =>
          val (g, out) = (gOut.dropRight(1), gOut.last)
          def local(symbol: String, kind: Char) =
            JSymbol.VarSymbol(0L, synth.names.fromString(symbol), synth.tpe(kind), msym)
          val (j, from, to) = (local("j", 'J'), local("from", 'J'), local("to", 'J'))
          val (high, error, simple, kept) =
            (local("high", 'D'), local("error", 'D'), local("simple", 'D'), local("kept", 'D'))
          val x = local("x", kind)
          for {
            blockFrom <- frame.ops("blockFrom")
            blockTo   <- frame.ops("blockTo")
            below     <- frame.ops("ltJ")
            add       <- frame.ops("sumJ")
            put       <- frame.ops("putD")
            slot      <- frame.ops("slot")
            read      <- frame.ops(if (held) s"get${suffix(kind)}" else s"at${suffix(kind)}J")
            inner     <- body(frame, msym, sn, g, make.Ident(x), stages, Leaf.Accumulate(high, error, simple, kept))
          } yield {
            def ident(v: JSymbol.VarSymbol) = make.Ident(v)
            def store(k: Int, v: JSymbol.VarSymbol): JCTree.JCStatement =
              make.Exec(
                synth.call(put, List(out, synth.call(slot, List(args.head, make.Literal(Int.box(k)))), ident(v)))
              )
            val advance = make.Assign(ident(j), synth.call(add, List(ident(j), make.Literal(Long.box(1L)))))
            advance.setType(synth.tpe('J'))
            val loop = make.ForLoop(
              JList.of[JCTree.JCStatement](make.VarDef(j, ident(from))),
              synth.call(below, List(ident(j), ident(to))),
              JList.of(make.Exec(advance)),
              make.Block(
                0,
                JList.from(
                  ((make.VarDef(x, synth.call(read, List(sn.head, ident(j)))): JCTree.JCStatement) :: inner).asJava
                )
              )
            )
            List[JCTree.JCStatement](
              make.VarDef(from, synth.call(blockFrom, List(args.head))),
              make.VarDef(to, synth.call(blockTo, List(sn(1), args.head))),
              make.VarDef(high, make.Literal(Double.box(0.0))),
              make.VarDef(error, make.Literal(Double.box(0.0))),
              make.VarDef(simple, make.Literal(Double.box(0.0))),
              make.VarDef(kept, make.Literal(Double.box(0.0))),
              loop,
              store(0, high),
              store(1, error),
              store(2, simple),
              store(3, kept)
            )
          }
        }
    }
  }

  private def body(
      frame: Frame,
      msym: JSymbol.MethodSymbol,
      sn: List[JCTree.JCExpression],
      g: List[JCTree.JCExpression],
      start: JCTree.JCExpression,
      stages: List[Stage],
      leaf: Leaf
  ): Result[List[JCTree.JCStatement]] = {
    val synth = frame.synth
    import synth.make
    val neutral = sn(1)
    val fallback: Option[JCTree.JCStatement] = leaf match {
      case Leaf.Consume(_) | Leaf.Accumulate(_, _, _, _) => None
      case Leaf.True                                     => Some(make.Return(synth.literal('Z', 0)))
      case _                                             => Some(make.Return(neutral))
    }
    def finish(cur: JCTree.JCExpression): Result[List[JCTree.JCStatement]] = leaf match {
      case Leaf.Identity => List(make.Return(cur): JCTree.JCStatement).success
      case Leaf.Widen =>
        frame.ops("widenI").map(w => List(make.Return(synth.call(w, List(cur))): JCTree.JCStatement))
      case Leaf.One  => List(make.Return(synth.literal('J', 1)): JCTree.JCStatement).success
      case Leaf.True => List(make.Return(synth.literal('Z', 1)): JCTree.JCStatement).success
      case Leaf.Bit(lam) =>
        frame
          .ops("bit")
          .map(b => List(make.Return(synth.call(b, List(applyLam(frame, lam, g, List(cur))))): JCTree.JCStatement))
      case Leaf.NotBit(lam) =>
        frame
          .ops("notBit")
          .map(b => List(make.Return(synth.call(b, List(applyLam(frame, lam, g, List(cur))))): JCTree.JCStatement))
      case Leaf.Consume(lam) => List(make.Exec(applyLam(frame, lam, g, List(cur))): JCTree.JCStatement).success
      case Leaf.Accumulate(high, compensation, simple, kept) =>
        for {
          error <- frame.ops("compensation")
          next  <- frame.ops("compensated")
          add   <- frame.ops("sumD")
        } yield {
          val delta                       = JSymbol.VarSymbol(0L, synth.names.fromString("delta"), synth.tpe('D'), msym)
          def ident(v: JSymbol.VarSymbol) = make.Ident(v)
          List[JCTree.JCStatement](
            make.VarDef(delta, synth.call(error, List(ident(high), ident(compensation), cur))),
            make.Assignment(high, synth.call(next, List(ident(high), ident(compensation), cur))),
            make.Assignment(compensation, ident(delta)),
            make.Assignment(simple, synth.call(add, List(ident(simple), cur))),
            make.Assignment(kept, synth.call(add, List(ident(kept), make.Literal(Double.box(1.0)))))
          )
        }
    }
    def go(rest: List[Stage], cur: JCTree.JCExpression, n: Int): Result[List[JCTree.JCStatement]] = rest match {
      case Nil => finish(cur)
      case Stage.Map(lam, _, to) :: tail =>
        val y    = JSymbol.VarSymbol(0L, synth.names.fromString(s"y$n"), synth.tpe(to), msym)
        val decl = make.VarDef(y, applyLam(frame, lam, g, List(cur)))
        go(tail, make.Ident(y), n + 1).map(decl :: _)
      case Stage.Filter(lam, _) :: tail =>
        go(tail, cur, n + 1).map { inner =>
          List(
            make.If(
              applyLam(frame, lam, g, List(cur)),
              make.Block(0, JList.from(inner.asJava)),
              fallback.getOrElse(make.Block(0, JList.nil[JCTree.JCStatement]()))
            ): JCTree.JCStatement
          )
        }
      case Stage.Sorted(_) :: tail => go(tail, cur, n)
    }
    go(stages, start, 0)
  }

  private def prim(kind: Char): Prim = LaunchCore.Prims.values.find(_.code == kind).get

  private def combine(op: String, kind: Char): Arg = Arg.Forward(s"$op${suffix(kind)}")

  private def compensated(chain: Chain): Boolean = chain.elem == 'D' && (chain.terminal match {
    case Terminal.Sum | Terminal.Average => true
    case _                               => false
  })

  private def orderOf(chain: Chain): Order = chain.terminal match {
    case Terminal.Reduce(_) | Terminal.ReduceId(_, _) => Order.Ordered
    case _                                            => Order.Unordered
  }

  private def reduction(library: String, chain: Chain): String = orderOf(chain) match {
    case Order.Ordered   => s"${library}transform_reduce"
    case Order.Unordered => s"${library}transform_reduce_unordered"
  }

  private def reduceStep(chain: Chain, role: String, w: Char, leaf: Leaf, combiner: Arg): Step = chain.source match {
    case Source.Array(_, k) =>
      Step(
        role,
        reduction(Spectra, chain),
        List(
          Arg.Array(k),
          Arg.Scalar('I'),
          Arg.Scalar(w),
          Arg.Fn(Program.Element(k, chain.stages, leaf, w, Some(w))),
          combiner
        ),
        Some(w)
      )
    case Source.Range(elem = k) =>
      Step(
        role,
        reduction(Core, chain),
        List(
          Arg.Scalar('J'),
          Arg.Scalar(w),
          Arg.Fn(Program.Index('J', Origin.Counted(k), chain.stages, leaf, w, Some(w))),
          combiner
        ),
        Some(w)
      )
  }

  private def counting(chain: Chain, role: String): Step = reduceStep(chain, role, 'J', Leaf.One, combine("sum", 'J'))

  private def activeStages(chain: Chain): List[Stage] = chain.stages.filter {
    case Stage.Sorted(_) => false
    case _               => true
  }

  private def pattern(chain: Chain): String = activeStages(chain).map {
    case Stage.Map(_, _, to) => to
    case _                   => Streams.FILTER_STAGE
  }.mkString

  private def materialise(chain: Chain): List[Step] = {
    val active = activeStages(chain)
    val kinds = active
      .scanLeft(chain.source.kind) {
        case (_, Stage.Map(_, _, to)) => to
        case (kind, _)                => kind
      }
      .toVector
    val origin = chain.source.kind
    Streams.segments(pattern(chain), chain.source.isRange).asScala.toList.map { segment =>
      val in     = kinds(segment.from)
      val to     = kinds(segment.to)
      val within = active.slice(segment.from, segment.to)
      segment.kind match {
        case Streams.Kind.TABULATE =>
          Step(
            segment.role,
            s"${Spectra}tabulate",
            List(
              Arg.Array(to),
              Arg.Scalar('I'),
              Arg.Fn(Program.Index('I', Origin.Counted(origin), within, Leaf.Identity, to))
            ),
            None
          )
        case Streams.Kind.TRANSFORM =>
          Step(
            segment.role,
            s"${Spectra}transform",
            List(Arg.Array(in), Arg.Array(to), Arg.Scalar('I'), Arg.Fn(Program.Element(in, within, Leaf.Identity, to))),
            None
          )
        case Streams.Kind.FILTER =>
          Step(
            segment.role,
            s"${Spectra}copy_if",
            List(
              Arg.Array(in),
              Arg.Scalar('I'),
              Arg.Array(in),
              Arg.Fn(Program.Element(in, within, Leaf.True, 'Z'))
            ),
            Some('I')
          )
      }
    }
  }

  private def survivorStep(chain: Chain, w: Char, leaf: Leaf, combiner: Arg): Step =
    Step(
      Streams.MAIN,
      reduction(Spectra, chain),
      List(
        Arg.Array(chain.elem),
        Arg.Scalar('I'),
        Arg.Scalar(w),
        Arg.Fn(Program.Element(chain.elem, Nil, leaf, w, Some(w))),
        combiner
      ),
      Some(w)
    )

  private def tailStep(lam: Lam, out: Char): Step =
    Step(
      Streams.TAIL,
      s"${Spectra}reduce",
      List(Arg.Array(out), Arg.Scalar('I'), Arg.Scalar(out), Arg.Fn(Program.Binary(lam, out, out))),
      Some(out)
    )

  private def plan(chain: Chain): Result[List[Step]] = {
    val out      = chain.elem
    val filtered = chain.filtered
    chain.terminal match {
      case _ if chain.sorted && (chain.terminal != Terminal.ToArray || !chain.stages.lastOption.exists {
            case Stage.Sorted(_) => true
            case _               => false
          }) =>
        "`sorted` is only lowered as the last stage before `toArray`".fail
      case Terminal.Sum | Terminal.Average if compensated(chain) =>
        val (kind, held) = chain.source match {
          case Source.Array(_, k)     => (k, true)
          case Source.Range(elem = k) => (k, false)
        }
        List(
          Step(
            Streams.SUMS,
            s"${Core}for_each",
            List(Arg.Scalar('J'), Arg.Fn(Program.Blocks(kind, held, chain.stages))),
            None
          )
        ).success
      case Terminal.Sum => List(reduceStep(chain, Streams.MAIN, out, Leaf.Identity, combine("sum", out))).success
      case Terminal.Count =>
        if (filtered) List(counting(chain, Streams.MAIN)).success else Nil.success
      case Terminal.Min | Terminal.Max =>
        val combiner = combine(if (chain.terminal == Terminal.Min) "min" else "max", out)
        if (filtered) (materialise(chain) :+ survivorStep(chain, out, Leaf.Identity, combiner)).success
        else List(reduceStep(chain, Streams.MAIN, out, Leaf.Identity, combiner)).success
      case Terminal.Average =>
        val wide     = if (out == 'D') 'D' else 'J'
        val leaf     = if (out == 'I') Leaf.Widen else Leaf.Identity
        val combiner = combine("sum", wide)
        if (filtered) (materialise(chain) :+ survivorStep(chain, wide, leaf, combiner)).success
        else List(reduceStep(chain, Streams.MAIN, wide, leaf, combiner)).success
      case Terminal.Matches(kind, lam) =>
        val leaf = if (kind == Match.All) Leaf.NotBit(lam) else Leaf.Bit(lam)
        List(reduceStep(chain, Streams.MAIN, 'J', leaf, combine("sum", 'J'))).success
      case Terminal.ReduceId(_, lam) =>
        (reduceStep(chain, Streams.MAIN, out, Leaf.Identity, Arg.Fn(Program.Binary(lam, out, out, Some(out)))) ::
          Option.when(chain.source.isRange)(tailStep(lam, out)).toList).success
      case Terminal.ForEach(lam) =>
        val k = chain.source.kind
        val origin = chain.source match {
          case Source.Array(_, _) => Origin.Held(k)
          case _                  => Origin.Counted(k)
        }
        List(
          Step(
            Streams.EACH,
            s"${Core}for_each",
            List(Arg.Scalar('J'), Arg.Fn(Program.Index('J', origin, chain.stages, Leaf.Consume(lam), 'V'))),
            None
          )
        ).success
      case Terminal.ToArray =>
        val sorted = chain.stages.exists { case Stage.Sorted(_) => true; case _ => false }
        (materialise(chain) ::: Option.when(sorted)(sortStep(out)).toList).success
      case Terminal.Reduce(lam) => (materialise(chain) :+ tailStep(lam, out)).success
      case Terminal.SortInPlace => List(sortStep(out)).success
      case Terminal.SetAll(lam) =>
        val to = chain.source.kind
        List(
          Step(
            Streams.FILL,
            s"${Spectra}tabulate",
            List(
              Arg.Array(to),
              Arg.Scalar('I'),
              Arg.Fn(Program.Index('I', Origin.Iota, List(Stage.Map(lam, 'I', to)), Leaf.Identity, to))
            ),
            None
          )
        ).success
      case Terminal.Prefix(lam) =>
        List(
          Step(
            Streams.SCAN,
            s"${Spectra}inclusive_scan",
            List(Arg.Array(out), Arg.Array(out), Arg.Scalar('I'), Arg.Fn(Program.Binary(lam, out, out))),
            None
          )
        ).success
    }
  }

  private def sortStep(kind: Char): Step =
    Step(
      Streams.ORDER,
      s"${Spectra}sort",
      List(Arg.Array(kind), Arg.Scalar('I'), Arg.Forward(s"lt${suffix(kind)}")),
      None
    )

  private def shapeOf(chain: Chain): String = {
    val op = chain.terminal match {
      case Terminal.Sum                     => Streams.Op.SUM
      case Terminal.Count                   => Streams.Op.COUNT
      case Terminal.Min                     => Streams.Op.MIN
      case Terminal.Max                     => Streams.Op.MAX
      case Terminal.Average                 => Streams.Op.AVERAGE
      case Terminal.ToArray if chain.sorted => Streams.Op.SORTED
      case Terminal.ToArray                 => Streams.Op.TO_ARRAY
      case Terminal.Matches(Match.Any, _)   => Streams.Op.ANY
      case Terminal.Matches(Match.All, _)   => Streams.Op.ALL
      case Terminal.Matches(Match.None, _)  => Streams.Op.NONE
      case Terminal.ForEach(_)              => Streams.Op.FOR_EACH
      case Terminal.ReduceId(_, _)          => Streams.Op.REDUCE_ID
      case Terminal.Reduce(_)               => Streams.Op.REDUCE
      case Terminal.SetAll(_)               => Streams.Op.SET_ALL
      case Terminal.Prefix(_)               => Streams.Op.PREFIX
      case Terminal.SortInPlace             => Streams.Op.SORT
    }
    val source = chain.source match {
      case Source.Array(_, _)          => 'A'
      case Source.Range(closed = true) => 'C'
      case Source.Range(_, _, _, _)    => 'R'
    }
    Streams.shape(op, source, chain.source.kind, chain.elem, pattern(chain), chain.contextFirst)
  }

  case class Identity(key: String, shape: String)

  def identify(chain: Chain): Identity = {
    val shape = shapeOf(chain)
    Identity(provisionalKey(chain, shape), shape)
  }

  private def digest(text: String): String = {
    val sha = MessageDigest.getInstance("SHA-1").digest(text.getBytes(StandardCharsets.UTF_8))
    sha.take(8).map(b => f"${b & 0xff}%02x").mkString
  }

  private def provisionalKey(chain: Chain, shape: String): String = {
    val lams = chain.lams.map(l => s"${l.impl.owner.flatName}.${l.impl.name}:${l.impl.`type`}").mkString(",")
    digest(s"$shape|$lams")
  }

  private def finalKey(provisional: String, lowered: List[(Step, List[LaunchSites.StepArg])]): String =
    digest(
      (provisional :: lowered.map { (step, args) =>
        val parts = args.map {
          case LaunchSites.StepArg.Scalar(p) => s"P${p.code}"
          case LaunchSites.StepArg.Array(p)  => s"A${p.code}"
          case LaunchSites.StepArg.Fn(c) =>
            s"${c.function}|${c.dependencies}|${c.environment}|${c.defs}|${c.result}"
        }
        s"${step.role}:${step.identity}:${step.result}:${parts.mkString(",")}"
      }).mkString("\n")
    )

  case class Compiled(key: String, shape: String, classes: List[(String, Offloader.Executable)])

  def compile(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      chain: Chain,
      home: JSymbol.ClassSymbol
  ): Result[Compiled] =
    sources.env match {
      case javac: JavacProcessingEnvironment =>
        val synth       = Synth(javac)
        val ops         = (name: String) => synth.method(OpsClass, name)
        val frame       = Frame(synth, home, chain, ops)
        val shape       = shapeOf(chain)
        val provisional = provisionalKey(chain, shape)
        for {
          steps <- plan(chain)
          lowered <- steps.traverse { step =>
            val fqcn      = Streams.siteName(provisional, step.role)
            val entryName = LaunchCore.normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
            step.args.zipWithIndex
              .traverse {
                case (Arg.Scalar(kind), _) => StepArg.Scalar(prim(kind)).success
                case (Arg.Array(kind), _)  => StepArg.Array(prim(kind)).success
                case (Arg.Forward(name), i) =>
                  for {
                    impl     <- ops(name)
                    tree     <- LaunchSites.treeOf(sources, impl)
                    callable <- LaunchSites.lowerCallable(sources, config, entryName, i, tree, impl, 0, Mode.Site)
                  } yield StepArg.Fn(callable)
                case (Arg.Fn(spec), i) =>
                  for {
                    built <- program(frame, s"stream_${step.role}_$i", spec)
                    (decl, msym, captured) = built
                    callable <- LaunchSites.lowerCallable(
                      sources,
                      config,
                      entryName,
                      i,
                      decl,
                      msym,
                      captured,
                      Mode.Site
                    )
                  } yield StepArg.Fn(callable)
              }
              .map(step -> _)
          }
          key = finalKey(provisional, lowered)
          classes <- lowered.traverse { (step, args) =>
            val fqcn = Streams.siteName(key, step.role)
            LaunchSites.compileStep(config, fqcn, step.identity, args, step.result.map(prim(_))).map(fqcn -> _)
          }
        } yield Compiled(key, shape, classes)
      case _ => "stream chains need the javac processing environment".fail
    }

  private def boxed(synth: Synth, tree: JCTree.JCExpression): Result[JCTree.JCExpression] =
    if (Option(tree.`type`).exists(_.isPrimitive)) {
      val box = synth.types.boxedClass(tree.`type`)
      synth
        .method(
          box.flatName.toString,
          "valueOf",
          m =>
            m.`type`.getParameterTypes.size == 1 && synth.types.isSameType(m.`type`.getParameterTypes.head, tree.`type`)
        )
        .map(valueOf => synth.call(valueOf, List(tree)))
    } else tree.success

  private def entryOf(chain: Chain): String = chain.terminal match {
    case Terminal.Sum | Terminal.ReduceId(_, _)           => chain.elem.toString.toLowerCase
    case Terminal.Count                                   => "j"
    case Terminal.Min | Terminal.Max | Terminal.Reduce(_) => "o" + chain.elem.toString.toLowerCase
    case Terminal.Average                                 => "od"
    case Terminal.ToArray                                 => chain.elem.toString.toLowerCase + "a"
    case Terminal.Matches(_, _)                           => "z"
    case _                                                => "v"
  }

  def rewrite(sources: TreeRemapper.Sources, site: Site, chain: Chain, compiled: Compiled): Result[Unit] =
    sources.env match {
      case javac: JavacProcessingEnvironment =>
        val synth = Synth(javac)
        import synth.make
        for {
          src <- chain.source match {
            case Source.Array(array, _)     => List(array).success
            case Source.Range(lo, hi, _, _) => List(lo, hi).traverse(boxed(synth, _))
          }
          extras <- chain.terminal match {
            case Terminal.ReduceId(identity, _) => boxed(synth, identity).map(List(_))
            case _                              => Nil.success
          }
          captured <- chain.lams.flatMap(_.args).traverse(boxed(synth, _))
          context <- chain.context.fold(
            synth
              .method("polyregion.jvm.runtime.Context", "current", _.`type`.getParameterTypes.isEmpty)
              .map(synth.call(_, Nil))
          )(_.success)
          values =
            if (chain.contextFirst) context :: src ::: extras ::: captured
            else src ::: context :: extras ::: captured
          entry <- synth.method(StreamsClass, entryOf(chain), _.`type`.getParameterTypes.size == 4)
          array = {
            val a = make.NewArray(
              make.Type(synth.syms.objectType),
              JList.nil[JCTree.JCExpression](),
              JList.from(values.asJava)
            )
            a.`type` = JType.ArrayType(synth.syms.objectType, synth.syms.arrayClass)
            a
          }
        } yield {
          chain.root.meth = make.QualIdent(entry)
          chain.root.typeargs = JList.nil[JCTree.JCExpression]()
          chain.root.args = JList.of[JCTree.JCExpression](
            make.ClassLiteral(site.home),
            make.Literal(compiled.key),
            make.Literal(compiled.shape),
            array
          )
        }
      case _ => "stream chains need the javac processing environment".fail
    }
}
