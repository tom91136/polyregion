package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.Tree
import com.sun.tools.javac.code.{Flags, Symbol as JSymbol, Symtab, Type as JType, TypeTag, Types}
import com.sun.tools.javac.processing.JavacProcessingEnvironment
import com.sun.tools.javac.tree.{JCTree, TreeMaker, TreeScanner}
import com.sun.tools.javac.util.{List as JList, Names, Position}
import polyregion.ast.*
import polyregion.javalang.LaunchSites.{Mode, Prim, StepArg}
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
      ret: JType
  )(
      body: (
          JSymbol.MethodSymbol,
          List[JCTree.JCExpression],
          List[JCTree.JCExpression],
          List[JCTree.JCExpression]
      ) => Result[List[JCTree.JCStatement]]
  ): Result[(JCTree.JCMethodDecl, JSymbol.MethodSymbol, Int)] = {
    val synth  = frame.synth
    val prefix = originType :: neutralType :: frame.gTypes
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
          synth.tpe(neutral.getOrElse('I')),
          List(synth.tpe(kind), synth.tpe(kind)),
          synth.tpe(ret)
        ) { (_, _, g, args) =>
          List(make.Return(applyLam(frame, lam, g, args)): JCTree.JCStatement).success
        }
      case Program.Element(kind, stages, leaf, ret, neutral) =>
        build(frame, name, frame.originType, synth.tpe(neutral.getOrElse('I')), List(synth.tpe(kind)), synth.tpe(ret)) {
          (msym, sn, g, args) => body(frame, msym, sn, g, args.head, stages, leaf)
        }
      case Program.Index(index, origin, stages, leaf, ret, neutral) =>
        build(
          frame,
          name,
          frame.originType,
          synth.tpe(neutral.getOrElse('I')),
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
      case Leaf.Consume(_) => None
      case Leaf.True       => Some(make.Return(synth.literal('Z', 0)))
      case _               => Some(make.Return(neutral))
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

  private def prim(kind: Char): Prim = LaunchSites.Prims.values.find(_.code == kind).get

  private def combine(op: String, kind: Char): Arg = Arg.Forward(s"$op${suffix(kind)}")

  private def reduceStep(chain: Chain, role: String, w: Char, leaf: Leaf, combiner: Arg): Step = chain.source match {
    case Source.Array(_, k) =>
      Step(
        role,
        s"${Spectra}transform_reduce",
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
        s"${Core}transform_reduce",
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

  private def valuesStep(chain: Chain, in: Char): Step = {
    val maps = chain.stages.collect { case m: Stage.Map => m }
    Step(
      Streams.VALUES,
      s"${Spectra}transform",
      List(
        Arg.Array(in),
        Arg.Array(chain.elem),
        Arg.Scalar('I'),
        Arg.Fn(Program.Element(in, maps, Leaf.Identity, chain.elem))
      ),
      None
    )
  }

  private def materialise(chain: Chain): List[Step] = {
    val out = chain.elem
    chain.source match {
      case Source.Array(_, k) =>
        if (chain.filtered)
          Step(
            Streams.VALIDITY,
            s"${Spectra}copy_if",
            List(
              Arg.Array(k),
              Arg.Scalar('I'),
              Arg.Array(k),
              Arg.Fn(Program.Element(k, chain.stages, Leaf.True, 'Z'))
            ),
            Some('I')
          ) :: Option.when(chain.mapped)(valuesStep(chain, k)).toList
        else Option.when(chain.mapped)(valuesStep(chain, k)).toList
      case Source.Range(elem = k) =>
        if (chain.filtered)
          Step(
            Streams.INDICES,
            s"${Spectra}tabulate",
            List(
              Arg.Array(k),
              Arg.Scalar('I'),
              Arg.Fn(Program.Index('I', Origin.Counted(k), Nil, Leaf.Identity, k))
            ),
            None
          ) :: Step(
            Streams.VALIDITY,
            s"${Spectra}copy_if",
            List(
              Arg.Array(k),
              Arg.Scalar('I'),
              Arg.Array(k),
              Arg.Fn(Program.Element(k, chain.stages, Leaf.True, 'Z'))
            ),
            Some('I')
          ) :: Option.when(chain.mapped)(valuesStep(chain, k)).toList
        else
          List(
            Step(
              Streams.FILL,
              s"${Spectra}tabulate",
              List(
                Arg.Array(out),
                Arg.Scalar('I'),
                Arg.Fn(Program.Index('I', Origin.Counted(k), chain.stages, Leaf.Identity, out))
              ),
              None
            )
          )
    }
  }

  private def plan(chain: Chain): Result[List[Step]] = {
    val out      = chain.elem
    val filtered = chain.filtered
    chain.terminal match {
      case _ if chain.sorted && (chain.terminal != Terminal.ToArray || !chain.stages.lastOption.exists {
            case Stage.Sorted(_) => true
            case _               => false
          }) =>
        "`sorted` is only lowered as the last stage before `toArray`".fail
      case Terminal.Sum => List(reduceStep(chain, Streams.MAIN, out, Leaf.Identity, combine("sum", out))).success
      case Terminal.Count =>
        if (filtered) List(counting(chain, Streams.MAIN)).success else Nil.success
      case Terminal.Min | Terminal.Max =>
        val op = if (chain.terminal == Terminal.Min) "min" else "max"
        (reduceStep(chain, Streams.MAIN, out, Leaf.Identity, combine(op, out)) :: Option
          .when(filtered)(counting(chain, Streams.COUNT))
          .toList).success
      case Terminal.Average =>
        val wide = if (out == 'D') 'D' else 'J'
        val leaf = if (out == 'I') Leaf.Widen else Leaf.Identity
        (reduceStep(chain, Streams.MAIN, wide, leaf, combine("sum", wide)) :: Option
          .when(filtered)(counting(chain, Streams.COUNT))
          .toList).success
      case Terminal.Matches(kind, lam) =>
        val leaf = if (kind == Match.All) Leaf.NotBit(lam) else Leaf.Bit(lam)
        List(reduceStep(chain, Streams.MAIN, 'I', leaf, combine("sum", 'I'))).success
      case Terminal.ReduceId(_, lam) =>
        List(
          reduceStep(chain, Streams.MAIN, out, Leaf.Identity, Arg.Fn(Program.Binary(lam, out, out, Some(out))))
        ).success
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
      case Terminal.Reduce(lam) =>
        (materialise(chain) :+ Step(
          Streams.TAIL,
          s"${Spectra}reduce",
          List(Arg.Array(out), Arg.Scalar('I'), Arg.Scalar(out), Arg.Fn(Program.Binary(lam, out, out))),
          Some(out)
        )).success
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
      Streams.SORT,
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
    Streams.shape(op, source, chain.source.kind, chain.elem, chain.filtered, chain.mapped)
  }

  case class Identity(key: String, shape: String)

  def identify(chain: Chain): Identity = {
    val shape = shapeOf(chain)
    Identity(keyOf(chain, shape), shape)
  }

  private def keyOf(chain: Chain, shape: String): String = {
    val lams = chain.lams.map(l => s"${l.impl.owner.flatName}.${l.impl.name}:${l.impl.`type`}").mkString(",")
    val sha  = MessageDigest.getInstance("SHA-1").digest(s"$shape|$lams".getBytes(StandardCharsets.UTF_8))
    sha.take(8).map(b => f"${b & 0xff}%02x").mkString
  }

  case class Compiled(key: String, shape: String, classes: List[(String, Offloader.Executable)])

  def compile(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      chain: Chain,
      home: JSymbol.ClassSymbol
  ): Result[Compiled] =
    sources.env match {
      case javac: JavacProcessingEnvironment =>
        val synth = Synth(javac)
        val ops   = (name: String) => synth.method(OpsClass, name)
        val frame = Frame(synth, home, chain, ops)
        val shape = shapeOf(chain)
        val key   = keyOf(chain, shape)
        for {
          steps <- plan(chain)
          classes <- steps.traverse { step =>
            val fqcn      = Streams.siteName(key, step.role)
            val entryName = LaunchSites.normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
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
              .flatMap(args => LaunchSites.compileStep(config, fqcn, step.identity, args, step.result.map(prim(_))))
              .map(fqcn -> _)
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
          values = context :: src ::: extras ::: captured
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
