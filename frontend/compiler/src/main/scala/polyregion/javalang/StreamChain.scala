package polyregion.javalang

import com.sun.tools.javac.code.{Flags, Symbol as JSymbol}
import com.sun.tools.javac.tree.{JCTree, TreeInfo}
import polyregion.ast.*

import scala.jdk.CollectionConverters.*

object StreamChain {

  enum Via {
    case Jdk
    case Par
  }

  case class Lam(impl: JSymbol.MethodSymbol, args: List[JCTree.JCExpression])

  enum Source {
    case Array(array: JCTree.JCExpression, elem: Char)
    case Range(lo: JCTree.JCExpression, hi: JCTree.JCExpression, closed: Boolean, elem: Char)

    def kind: Char = this match {
      case Array(_, elem)     => elem
      case Range(elem = elem) => elem
    }
  }

  enum Stage {
    case Map(lam: Lam, from: Char, to: Char)
    case Filter(lam: Lam, kind: Char)
    case Sorted(kind: Char)
  }

  enum Match {
    case Any
    case All
    case None
  }

  enum Terminal {
    case Sum
    case Count
    case Min
    case Max
    case Average
    case ToArray
    case Matches(kind: Match, predicate: Lam)
    case ForEach(consumer: Lam)
    case ReduceId(identity: JCTree.JCExpression, op: Lam)
    case Reduce(op: Lam)
    case SetAll(lam: Lam)
    case Prefix(op: Lam)
    case SortInPlace
  }

  case class Chain(
      root: JCTree.JCMethodInvocation,
      via: Via,
      source: Source,
      stages: List[Stage],
      terminal: Terminal,
      context: Option[JCTree.JCExpression],
      parallel: Boolean
  ) {
    def lams: List[Lam] = stages.collect {
      case Stage.Map(lam, _, _) => lam
      case Stage.Filter(lam, _) => lam
    } ::: (terminal match {
      case Terminal.Matches(_, lam)  => List(lam)
      case Terminal.ForEach(lam)     => List(lam)
      case Terminal.ReduceId(_, lam) => List(lam)
      case Terminal.Reduce(lam)      => List(lam)
      case Terminal.SetAll(lam)      => List(lam)
      case Terminal.Prefix(lam)      => List(lam)
      case _                         => Nil
    })

    def filtered: Boolean = stages.exists { case Stage.Filter(_, _) => true; case _ => false }
    def mapped: Boolean   = stages.exists { case Stage.Map(_, _, _) => true; case _ => false }
    def sorted: Boolean   = stages.exists { case Stage.Sorted(_) => true; case _ => false }

    def elem: Char = stages.reverse.collectFirst { case Stage.Map(_, _, to) => to }.getOrElse(source.kind)
  }

  case class Decoded(root: JCTree.JCMethodInvocation, via: Via, chain: Result[Chain])

  private val JdkKinds = Map(
    "java.util.stream.IntStream"    -> 'I',
    "java.util.stream.LongStream"   -> 'J',
    "java.util.stream.DoubleStream" -> 'D'
  )

  private val ParKinds = Map(
    "polyregion.javalang.ParInt"    -> 'I',
    "polyregion.javalang.ParLong"   -> 'J',
    "polyregion.javalang.ParDouble" -> 'D'
  )

  private val ArrayKinds = Map("int[]" -> 'I', "long[]" -> 'J', "double[]" -> 'D')

  private val SortKinds = ArrayKinds + ("float[]" -> 'F')

  private val PrimitiveKinds = Map("int" -> 'I', "long" -> 'J', "double" -> 'D')

  private val ParFactory = Map("ints" -> 'I', "longs" -> 'J', "doubles" -> 'D')

  private val ParClass = "polyregion.javalang.Par"

  private val ContextType = "polyregion.jvm.runtime.Context"

  private val Terminals = Set(
    "sum",
    "count",
    "min",
    "max",
    "average",
    "toArray",
    "anyMatch",
    "allMatch",
    "noneMatch",
    "forEach",
    "forEachOrdered",
    "reduce"
  )

  private val ArrayCalls = Map(
    "java.util.Arrays" -> Set("parallelSetAll", "parallelPrefix", "parallelSort"),
    ParClass           -> Set("setAll", "prefix", "sort")
  )

  private def methodOf(tree: JCTree.JCExpression): Option[JSymbol.MethodSymbol] = tree match {
    case a: JCTree.JCMethodInvocation =>
      TreeInfo.symbol(a.meth) match {
        case m: JSymbol.MethodSymbol => Some(m)
        case _                       => None
      }
    case _ => None
  }

  private def receiver(a: JCTree.JCMethodInvocation): Option[JCTree.JCExpression] = a.meth match {
    case select: JCTree.JCFieldAccess => Some(TreeInfo.skipParens(select.selected))
    case _                            => None
  }

  private def ownerOf(m: JSymbol.MethodSymbol): String = m.owner.flatName.toString

  private def typeName(tree: JCTree.JCExpression): String = Option(tree.`type`).fold("")(_.toString)

  private def kindOfStream(tree: JCTree.JCExpression): Option[(Via, Char)] =
    Option(tree.`type`).map(_.tsym.flatName.toString).flatMap { name =>
      JdkKinds.get(name).map(Via.Jdk -> _).orElse(ParKinds.get(name).map(Via.Par -> _))
    }

  private def lam(tree: JCTree.JCExpression): Result[Lam] =
    LaunchSites.lambdaOf(tree) match {
      case Some((call: JCTree.JCMethodInvocation, impl)) =>
        if (impl.isStatic || (impl.flags & Flags.SYNTHETIC) != 0) Right(Lam(impl, call.args.asScala.toList))
        else s"a stream lambda cannot bind a receiver, `${impl.owner}.${impl.name}` is an instance method".fail
      case _ => "a stream operation must take a lambda expression or a static method reference".fail
    }

  private case class Spine(
      via: Via,
      source: Source,
      stages: List[Stage],
      context: Option[JCTree.JCExpression],
      parallel: Option[Boolean]
  )

  private def source(via: Via, source: Source): Result[Spine] = Right(Spine(via, source, Nil, None, None))

  private def spine(tree: JCTree.JCExpression): Result[Spine] = {
    val expr = TreeInfo.skipParens(tree)
    (expr, methodOf(expr)) match {
      case (a: JCTree.JCMethodInvocation, Some(m)) =>
        val owner = ownerOf(m)
        val name  = m.name.toString
        val args  = a.args.asScala.toList.map(TreeInfo.skipParens(_))
        (owner, name) match {
          case ("java.util.stream.IntStream", "range" | "rangeClosed") =>
            source(Via.Jdk, Source.Range(args(0), args(1), name == "rangeClosed", 'I'))
          case ("java.util.stream.LongStream", "range" | "rangeClosed") =>
            source(Via.Jdk, Source.Range(args(0), args(1), name == "rangeClosed", 'J'))
          case ("java.util.Arrays", "stream") if args.size == 1 =>
            ArrayKinds
              .get(typeName(args.head))
              .toRight(CompilerException("Arrays.stream is only lowered for int[], long[] and double[]"))
              .flatMap(k => source(Via.Jdk, Source.Array(args.head, k)))
          case (ParClass, "ints" | "longs" | "doubles") =>
            source(Via.Par, Source.Array(args.head, ParFactory(name)))
          case (ParClass, "range" | "rangeClosed") =>
            val kind = PrimitiveKinds.getOrElse(m.`type`.getParameterTypes.head.toString, 'I')
            source(Via.Par, Source.Range(args(0), args(1), name == "rangeClosed", kind))
          case _ =>
            (kindOfStream(expr), receiver(a)) match {
              case (Some((via, k)), Some(inner)) =>
                spine(inner).flatMap { up =>
                  name match {
                    case "parallel"   => Right(up.copy(parallel = Some(true)))
                    case "sequential" => Right(up.copy(parallel = Some(false)))
                    case "unordered"  => Right(up)
                    case "on" if via == Via.Par =>
                      if (up.context.isDefined) "a .par chain selects its context once".fail
                      else Right(up.copy(context = Some(args.head)))
                    case "map"    => lam(args.head).map(l => up.copy(stages = up.stages :+ Stage.Map(l, k, k)))
                    case "filter" => lam(args.head).map(l => up.copy(stages = up.stages :+ Stage.Filter(l, k)))
                    case "mapToInt" | "mapToLong" | "mapToDouble" =>
                      val to = name match {
                        case "mapToInt"  => 'I'
                        case "mapToLong" => 'J'
                        case _           => 'D'
                      }
                      lam(args.head).map(l => up.copy(stages = up.stages :+ Stage.Map(l, k, to)))
                    case "sorted" if args.isEmpty => Right(up.copy(stages = up.stages :+ Stage.Sorted(k)))
                    case other                    => s"`$other` is not a lowered stream operation".fail
                  }
                }
              case _ => s"`$owner.$name` is not a lowered stream source".fail
            }
        }
      case _ => "a stream chain must be a single expression".fail
    }
  }

  def decode(root: JCTree.JCMethodInvocation): Option[Decoded] =
    methodOf(root).flatMap { m =>
      val owner = ownerOf(m)
      val name  = m.name.toString
      val args  = root.args.asScala.toList.map(TreeInfo.skipParens(_))
      if (ArrayCalls.get(owner).exists(_.contains(name)))
        Some(Decoded(root, if (owner == ParClass) Via.Par else Via.Jdk, arrays(root, m, args)))
      else if (JdkKinds.contains(owner) && Terminals.contains(name))
        Some(Decoded(root, Via.Jdk, terminal(root, m, args)))
      else if (ParKinds.contains(owner) && Terminals.contains(name))
        Some(Decoded(root, Via.Par, terminal(root, m, args)))
      else if (ParKinds.contains(owner) || owner == ParClass)
        Some(
          Decoded(root, Via.Par, s"`$owner.$name` is not part of a .par chain that ends in a terminal operation".fail)
        )
      else None
    }

  private def arrays(
      root: JCTree.JCMethodInvocation,
      m: JSymbol.MethodSymbol,
      args: List[JCTree.JCExpression]
  ): Result[Chain] = {
    val name    = m.name.toString
    val via     = if (ownerOf(m) == ParClass) Via.Par else Via.Jdk
    val context = args.headOption.filter(typeName(_) == ContextType)
    val rest    = if (context.isDefined) args.tail else args
    val sorting = name == "parallelSort" || name == "sort"
    for {
      array <- rest.headOption.toRight(CompilerException(s"`$name` has no array"))
      kind <- (if (sorting) SortKinds else ArrayKinds)
        .get(typeName(array))
        .toRight(
          CompilerException(
            s"`$name` is only lowered for int[], long[], double[]${if (sorting) " and float[]" else ""}"
          )
        )
      _ <-
        if (rest.size == (if (sorting) 1 else 2)) Right(())
        else s"`$name` with a range or a comparator is not lowered".fail
      terminal <-
        if (sorting) Right(Terminal.SortInPlace)
        else if (name.endsWith("SetAll") || name == "setAll") lam(rest(1)).map(Terminal.SetAll(_))
        else lam(rest(1)).map(Terminal.Prefix(_))
    } yield Chain(root, via, Source.Array(array, kind), Nil, terminal, context, parallel = true)
  }

  private def terminal(
      root: JCTree.JCMethodInvocation,
      m: JSymbol.MethodSymbol,
      args: List[JCTree.JCExpression]
  ): Result[Chain] = {
    val name = m.name.toString
    for {
      up <- receiver(root).toRight(CompilerException(s"`$name` has no receiver")).flatMap(spine(_))
      terminal <- (name, args) match {
        case ("sum", Nil)                            => Right(Terminal.Sum)
        case ("count", Nil)                          => Right(Terminal.Count)
        case ("min", Nil)                            => Right(Terminal.Min)
        case ("max", Nil)                            => Right(Terminal.Max)
        case ("average", Nil)                        => Right(Terminal.Average)
        case ("toArray", Nil)                        => Right(Terminal.ToArray)
        case ("anyMatch", List(p))                   => lam(p).map(Terminal.Matches(Match.Any, _))
        case ("allMatch", List(p))                   => lam(p).map(Terminal.Matches(Match.All, _))
        case ("noneMatch", List(p))                  => lam(p).map(Terminal.Matches(Match.None, _))
        case ("forEach" | "forEachOrdered", List(f)) => lam(f).map(Terminal.ForEach(_))
        case ("reduce", List(identity, op))          => lam(op).map(Terminal.ReduceId(identity, _))
        case ("reduce", List(op))                    => lam(op).map(Terminal.Reduce(_))
        case _                                       => s"`$name` is not a lowered stream terminal".fail
      }
    } yield Chain(root, up.via, up.source, up.stages, terminal, up.context, up.parallel.contains(true))
  }
}
