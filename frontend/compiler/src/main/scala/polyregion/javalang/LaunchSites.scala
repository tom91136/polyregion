package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.{ExpressionTree, MethodInvocationTree, MethodTree, ParenthesizedTree, Tree, TypeCastTree}
import com.sun.tools.javac.code.{Flags, Symbol as JSymbol, Type as JType}
import com.sun.tools.javac.processing.JavacProcessingEnvironment
import com.sun.tools.javac.util.{List as JList, Names, Position}
import com.sun.tools.javac.tree.{JCTree, TreeInfo, TreeMaker, TreeScanner}
import polyregion.ast.pass.*
import polyregion.ast.{PolyAST as p, *}
import polyregion.scalalang.LaunchConfig

import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, TypeElement}
import scala.annotation.tailrec
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object LaunchSites {

  import LaunchCore.*

  private val ContextType = "polyregion.jvm.runtime.Context"

  private val Boxed: Map[String, TypeKind] = Map(
    "java.lang.Integer"   -> TypeKind.INT,
    "java.lang.Long"      -> TypeKind.LONG,
    "java.lang.Float"     -> TypeKind.FLOAT,
    "java.lang.Double"    -> TypeKind.DOUBLE,
    "java.lang.Short"     -> TypeKind.SHORT,
    "java.lang.Byte"      -> TypeKind.BYTE,
    "java.lang.Character" -> TypeKind.CHAR,
    "java.lang.Boolean"   -> TypeKind.BOOLEAN
  )

  private enum Kind {
    case Context
    case Scalar(prim: Prim)
    case Array(prim: Prim)
    case Callable
  }

  case class Site(
      call: JCTree.JCMethodInvocation,
      stub: JSymbol.MethodSymbol,
      identity: String,
      within: Option[JSymbol.MethodSymbol]
  )

  case class SugarSite(
      call: JCTree.JCMethodInvocation,
      stub: JSymbol.MethodSymbol,
      identity: String,
      rank: Int,
      map: Int,
      within: Option[JSymbol.MethodSymbol]
  )

  private class StubScanner(
      within: Option[JSymbol.MethodSymbol],
      sites: mutable.ListBuffer[Site],
      references: mutable.ListBuffer[Site],
      sugars: mutable.ListBuffer[SugarSite]
  ) extends TreeScanner {
    override def visitApply(a: JCTree.JCMethodInvocation): Unit = {
      TreeInfo.symbol(a.meth) match {
        case stub: JSymbol.MethodSymbol =>
          InterfaceStub.identityOf(stub).foreach(id => sites += Site(a, stub, id, within))
          InterfaceStub.sugarOf(stub).foreach { (id, rank, map) =>
            sugars += SugarSite(a, stub, id, rank, map, within)
          }
        case _ =>
      }
      TreeRemapper.dynamicSym(a).flatMap(TreeRemapper.metafactoryImpl(_)).foreach { impl =>
        InterfaceStub.identityOf(impl).foreach(id => references += Site(a, impl, id, within))
      }
      super.visitApply(a)
    }
  }

  case class Collected(sites: List[Site], references: List[Site], sugars: List[SugarSite])

  def collect(scope: Tree): Collected = scope match {
    case cls: JCTree.JCClassDecl =>
      val sites      = mutable.ListBuffer.empty[Site]
      val references = mutable.ListBuffer.empty[Site]
      val sugars     = mutable.ListBuffer.empty[SugarSite]
      cls.defs.asScala.foreach {
        case m: JCTree.JCMethodDecl => m.accept(StubScanner(Option(m.sym), sites, references, sugars))
        case _: JCTree.JCClassDecl  =>
        case member                 => member.accept(StubScanner(None, sites, references, sugars))
      }
      Collected(sites.toList, references.toList, sugars.toList)
    case _ => Collected(Nil, Nil, Nil)
  }

  @tailrec
  private def peel(tree: ExpressionTree): ExpressionTree = tree match {
    case par: ParenthesizedTree => peel(par.getExpression)
    case cast: TypeCastTree     => peel(cast.getExpression)
    case other                  => other
  }

  def lambdaOf(tree: ExpressionTree): Option[(MethodInvocationTree, JSymbol.MethodSymbol)] =
    peel(tree) match {
      case mi: MethodInvocationTree =>
        TreeRemapper.dynamicSym(mi).flatMap(TreeRemapper.metafactoryImpl(_)).map(mi -> _)
      case _ => None
    }

  private def callablesIn(call: JCTree.JCMethodInvocation): List[JSymbol.MethodSymbol] =
    call.getArguments.asScala.toList.flatMap(lambdaOf(_).map(_._2))

  def callables(site: Site): List[JSymbol.MethodSymbol] = callablesIn(site.call)

  def callables(site: SugarSite): List[JSymbol.MethodSymbol] = callablesIn(site.call)

  private def classify(tpe: TypeMirror): Result[Kind] = {
    val unsupported = CompilerException(s"launch argument of type `$tpe` is not supported")
    tpe match {
      case primitive if Prims.contains(primitive.getKind) => Kind.Scalar(Prims(primitive.getKind)).success
      case array: ArrayType =>
        Prims.get(array.getComponentType.getKind).map(Kind.Array(_)).toRight(unsupported)
      case declared: DeclaredType =>
        declared.asElement match {
          case elem: TypeElement =>
            val name = elem.getQualifiedName.toString
            if (name == ContextType) Kind.Context.success
            else
              Boxed
                .get(name)
                .map(kind => Kind.Scalar(Prims(kind)))
                .orElse(Option.when(elem.getKind == ElementKind.INTERFACE)(Kind.Callable))
                .toRight(unsupported)
          case _ => Left(unsupported)
        }
      case _ => Left(unsupported)
    }
  }

  private def resultOf(tpe: TypeMirror): Result[(p.Type, Char)] = tpe.getKind match {
    case TypeKind.VOID                => (p.Type.Unit0: p.Type, 'V').success
    case kind if Prims.contains(kind) => (Prims(kind).tpe, Prims(kind).code).success
    case _                            => s"launch result of type `$tpe` is not supported".fail
  }

  private case class Argument(tree: ExpressionTree, kind: Kind)

  private def argumentsOf(site: Site): Result[List[Argument]] = {
    val params  = site.stub.getParameters.asScala.toList.map(_.asType)
    val actuals = site.call.getArguments.asScala.toList
    for {
      kinds <- params.traverse(classify(_))
      _ <-
        if (kinds.sizeCompare(actuals) == 0) ().success
        else s"`${site.stub.name}` takes ${kinds.size} arguments, the call passes ${actuals.size}".fail
    } yield actuals.zip(kinds).collect { case (tree, kind) if kind != Kind.Context => Argument(tree, kind) }
  }

  private def partOf(argument: Argument): Result[String] = argument match {
    case Argument(_, Kind.Scalar(prim)) => prim.code.toString.success
    case Argument(_, Kind.Array(prim))  => s"[${prim.code}".success
    case Argument(tree, Kind.Callable) =>
      lambdaOf(tree)
        .toRight(CompilerException("a launch callable must be a lambda expression or a static method reference"))
        .flatMap { (_, impl) =>
          if (impl.isStatic || (impl.flags & Flags.SYNTHETIC) != 0)
            s"@${impl.owner.flatName.toString.replace('.', '/')}.${impl.name}".success
          else s"a launch callable cannot bind a receiver, `${impl.owner}.${impl.name}` is an instance method".fail
        }
    case Argument(_, Kind.Context) => "".success
  }

  def nameOf(site: Site): Result[String] =
    for {
      arguments <- argumentsOf(site)
      parts     <- arguments.traverse(partOf(_))
    } yield LaunchSite.nameOf(site.identity, parts.asJava)

  private def kindOfCapture(tpe: p.Type, mode: Mode): Option[Char] = tpe match {
    case p.Type.Ptr(component, _) if mode == Mode.Region || PrimOfType.contains(component) => Some('a')
    case other => PrimOfType.get(other).map(_.code)
  }

  def lowerCallable(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      entryName: String,
      index: Int,
      tree: MethodTree,
      impl: JSymbol.MethodSymbol,
      captured: Int,
      mode: Mode
  ): Result[Callable] = {
    val symbol = s"${entryName}_callable$index"
    for {
      _ <-
        if (mode == Mode.Region || impl.isStatic) ().success
        else inexpressible("a launch callable cannot use `this`")
      region   <- TreeRemapper.compileRegion(sources, tree, byReference = mode == Mode.Region)
      raw      <- Offloader.withFieldStructs(region.program, sources.classTable, region.outerRefs)
      rawEntry <- raw.entry.failIfEmpty(s"launch callable `$symbol` has no entry")
      routed = if (mode == Mode.Region) Offloader.routeReturnThroughCapture(rawEntry) else rawEntry
      optimised <- Offloader.optimise(
        region.tags,
        region.dispatch,
        region.owners,
        raw,
        routed,
        symbol,
        PluginEntry.defaultLog
      )
      (monoToPoly, optimisedProgram) = optimised
      unique                         = withUniqueNames(optimisedProgram.entry.toList ::: optimisedProgram.functions)
      program = optimisedProgram.copy(
        entry = unique.headOption.filter(_ => optimisedProgram.entry.isDefined),
        functions = unique.drop(optimisedProgram.entry.toList.size)
      )
      resultCode <- rawEntry.rtn match {
        case p.Type.Unit0 => 'V'.success
        case other =>
          PrimOfType
            .get(other)
            .map(_.code)
            .toRight(Inexpressible(returnTypeError(other)))
      }
      entry <- program.entry.failIfEmpty(s"launch callable `$symbol` lost its entry")
      twinned <-
        if (mode == Mode.Region && program.defs.nonEmpty)
          Offloader.twinsOf(sources, program, monoToPoly, region.tags, entryName).map(Twinned(_, _))
        else Twinned(Nil, Nil).success
      (results, plain) = entry.decl.termCaptures.partition(_.named.symbol == Offloader.ResultCapture)
      origins <- plain.traverse { arg =>
        region.captures
          .find(_.name == arg.named.symbol)
          .map(arg -> _.capture)
          .toRight(CompilerException(s"launch callable parameter `${arg.named.symbol}` has no source"))
      }
      _ <-
        if (entry.moduleCaptures.isEmpty && entry.decl.args.isEmpty) ().success
        else s"launch callable `$symbol` captures modules".fail
      bound <- origins.traverse { (arg, capture) =>
        def env(kind: Option[Char], source: String): Result[Either[EnvField, Param]] =
          kind
            .map(k => Left(EnvField(arg.named, k, source)))
            .toRight(
              Inexpressible(
                s"launch callable capture `${arg.named.symbol}` of type `${arg.named.tpe.repr}` is not supported"
              )
            )
        capture match {
          case TreeRemapper.Capture.Arg(i) if mode == Mode.Region || i < captured =>
            env(kindOfCapture(arg.named.tpe, mode), i.toString)
          case TreeRemapper.Capture.Arg(i) => Right(Param(arg, i)).success
          case TreeRemapper.Capture.Length(i) if mode == Mode.Region || i < captured =>
            env(kindOfCapture(arg.named.tpe, mode).map(_ => 'l'), i.toString)
          case TreeRemapper.Capture.RowLengths(i) if mode == Mode.Region => env(Some('R'), i.toString)
          case TreeRemapper.Capture.StaticField(owner, name) if mode == Mode.Region =>
            env(Some('s'), s"$owner#$name")
          case other => inexpressible(s"launch callable `${impl.name}` uses an unsupported capture $other")
        }
      }
      split = Split(bound.collect { case Left(field) => field }, bound.collect { case Right(param) => param })
      callable <- callableOf(
        config,
        entryName,
        index,
        symbol,
        mode,
        entry,
        program,
        resultCode,
        twinned,
        split,
        results
      )
    } yield callable
  }

  def compile(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      site: Site,
      fqcn: String
  ): Result[Offloader.Executable] = {
    val entryName = normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
    for {
      identity  <- Identity.parse(site.identity)
      arguments <- argumentsOf(site)
      returned  <- resultOf(site.stub.getReturnType)
      (rtn, rtnCode) = returned
      lowered <- arguments.zipWithIndex.collect { case (Argument(tree, Kind.Callable), i) => i -> tree }.traverse {
        (i, tree) =>
          lambdaOf(tree)
            .toRight(CompilerException("a launch callable must be a lambda expression"))
            .flatMap { (call, impl) =>
              treeOf(sources, impl)
                .flatMap(tree =>
                  lowerCallable(sources, config, entryName, i, tree, impl, call.getArguments.size, Mode.Site)
                    .map(i -> _)
                )
            }
      }
      byIndex = lowered.toMap
      specs = arguments.zipWithIndex.map {
        case (Argument(_, Kind.Scalar(prim)), _) => s"P${prim.code}"
        case (Argument(_, Kind.Array(prim)), _)  => s"A${prim.code}"
        case (_, i)                              => specOf(byIndex.get(i).flatMap(_.environment))
      }
      argumentTypes = arguments.zipWithIndex.map {
        case (Argument(_, Kind.Scalar(prim)), _) => prim.tpe
        case (Argument(_, Kind.Array(prim)), _)  => p.Type.Ptr(prim.tpe, p.Type.Space.Global)
        case (_, i)                              => callableType(byIndex(i))
      }
      sizes = arguments.collect {
        case Argument(_, Kind.Scalar(prim)) => p.Program.TypeSize(prim.tpe, prim.size)
        case Argument(_, Kind.Array(prim))  => p.Program.TypeSize(prim.tpe, prim.size)
      }
      executable <- assemble(
        Assembly(config, fqcn, identity, rtn, rtnCode, argumentTypes, sizes, specs, lowered.map(_._2), 'V'),
        classOf[LaunchSite]
      )
    } yield executable
  }

  private def wrapperOf(
      sources: TreeRemapper.Sources,
      callable: ExpressionTree,
      rank: Int
  ): Result[(MethodTree, JSymbol.MethodSymbol, Int)] =
    sources.env match {
      case javac: JavacProcessingEnvironment =>
        Either.catchNonFatal {
          val context      = javac.getContext
          val make         = TreeMaker.instance(context).at(Position.NOPOS)
          val names        = Names.instance(context)
          val (site, impl) = lambdaOf(callable).get
          if (!impl.isStatic) throw new IllegalArgumentException("a launch callable cannot use `this`")
          val captured = site.getArguments.size
          val longTpe  = javac.getTypeUtils.getPrimitiveType(TypeKind.LONG).asInstanceOf[JType]
          val paramTypes = impl.`type`.getParameterTypes.asScala.toList.take(captured) ++
            List.fill(rank)(longTpe) ++ List(longTpe)
          val resultType = impl.getReturnType
          val wrapperType = new JType.MethodType(
            JList.from(paramTypes.asJava),
            resultType,
            JList.nil[JType](),
            impl.owner.enclClass()
          )
          val wrapper = JSymbol.MethodSymbol(
            Flags.STATIC | Flags.SYNTHETIC,
            names.fromString(impl.name.toString),
            wrapperType,
            impl.owner
          )
          val params = paramTypes.zipWithIndex.map((tpe, i) => make.Param(names.fromString(s"p$i"), tpe, wrapper))
          wrapper.params = JList.from(params.map(_.sym).asJava)
          def ident(at: Int): JCTree.JCExpression = make.Ident(params(at).sym): JCTree.JCExpression
          def binary(tag: JCTree.Tag, left: JCTree.JCExpression, right: JCTree.JCExpression): JCTree.JCExpression = {
            val tree = make.Binary(tag, left, right)
            tree.`type` = longTpe
            tree
          }
          val index = ident(params.size - 1)
          val indices = RegionIndices.axes(rank).map { axis =>
            val quot = axis.quotient match {
              case Nil => index
              case extents =>
                val divisor = extents
                  .map(extent => ident(captured + extent))
                  .reduceLeft((left, right) => binary(JCTree.Tag.MUL, left, right))
                binary(JCTree.Tag.DIV, index, divisor)
            }
            axis.modulus.fold(quot)(extent => binary(JCTree.Tag.MOD, quot, ident(captured + extent)))
          }
          val arguments = params.indices.take(captured).map(ident).toList ++ indices
          val call      = make.App(make.QualIdent(impl), JList.from(arguments.asJava))
          val stmt =
            if (resultType.getKind == TypeKind.VOID) make.Exec(call) else make.Return(call)
          val tree = make.MethodDef(wrapper, make.Block(0, JList.of[JCTree.JCStatement](stmt))): MethodTree
          (tree, wrapper, captured + rank)
        }
      case _ => "linearised region sugar needs the javac processing environment".fail
    }

  def compileSugar(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      site: SugarSite
  ): Result[(String, Offloader.Executable)] = {
    val parameters = site.stub.getParameters.asScala.toList
    val contextFirst =
      parameters.headOption.exists(param => classify(param.asType).toOption.contains(Kind.Context))
    val offset  = site.rank + Option.when(contextFirst)(1).getOrElse(0)
    val actuals = site.call.getArguments.asScala.toList
    val rest    = actuals.drop(offset)
    val kinds   = parameters.drop(offset).map(_.asType).traverse(classify(_))
    for {
      identity                        <- Identity.parse(site.identity)
      (rtn, rtnCode)                  <- resultOf(site.stub.getReturnType)
      ks                              <- kinds
      (wrapper, wrapperSym, captured) <- wrapperOf(sources, rest(site.map), site.rank)
      parts <- ks.zipWithIndex.traverse {
        case (Kind.Scalar(prim), _) => prim.code.toString.success
        case (Kind.Array(prim), _)  => s"[${prim.code}".success
        case (Kind.Callable, i) if i == site.map =>
          s"@${wrapperSym.owner.flatName.toString.replace('.', '/')}.${wrapperSym.name}".success
        case (Kind.Callable, i) =>
          lambdaOf(rest(i))
            .map(_._2)
            .map(impl => s"@${impl.owner.flatName.toString.replace('.', '/')}.${impl.name}".success)
            .getOrElse("a launch callable must be a lambda expression".fail)
        case (Kind.Context, _) => "".success
      }
      name      = LaunchSite.nameOf(site.identity, ("J" :: parts).asJava)
      entryName = normalise(name.substring(name.lastIndexOf('.') + 1))
      lowered <- rest.indices.toList
        .filter { i =>
          ks(i) == Kind.Callable
        }
        .traverse { i =>
          val expanded = i + 1
          if (i == site.map)
            lowerCallable(sources, config, entryName, expanded, wrapper, wrapperSym, captured, Mode.Site)
              .map(expanded -> _)
          else
            lambdaOf(rest(i))
              .toRight(CompilerException("a launch callable must be a lambda expression or a static method reference"))
              .flatMap { (call, impl) =>
                treeOf(sources, impl).flatMap(tree =>
                  lowerCallable(sources, config, entryName, expanded, tree, impl, call.getArguments.size, Mode.Site)
                    .map(expanded -> _)
                )
              }
        }
      byIndex = lowered.toMap
      argumentTypes = (p.Type.IntS64: p.Type) :: ks.zipWithIndex.map {
        case (Kind.Scalar(prim), _) => prim.tpe: p.Type
        case (Kind.Array(prim), _)  => p.Type.Ptr(prim.tpe, p.Type.Space.Global): p.Type
        case (_, i)                 => callableType(byIndex(i + 1))
      }
      sizes = p.Program.TypeSize(p.Type.IntS64, 8) :: ks.collect {
        case Kind.Scalar(prim) => p.Program.TypeSize(prim.tpe, prim.size)
        case Kind.Array(prim)  => p.Program.TypeSize(prim.tpe, prim.size)
      }
      specs = "PJ" :: ks.zipWithIndex.map {
        case (Kind.Scalar(prim), _) => s"P${prim.code}"
        case (Kind.Array(prim), _)  => s"A${prim.code}"
        case (Kind.Callable, i)     => specOf(byIndex(i + 1).environment)
        case (Kind.Context, _)      => ""
      }
      executable <- assemble(
        Assembly(config, name, identity, rtn, rtnCode, argumentTypes, sizes, specs, lowered.map(_._2), 'V'),
        classOf[LaunchSite]
      )
    } yield (name, executable)
  }

  enum StepArg {
    case Scalar(prim: Prim)
    case Array(prim: Prim)
    case Fn(callable: Callable)
  }

  def compileStep(
      config: LaunchConfig,
      fqcn: String,
      identity: String,
      args: List[StepArg],
      result: Option[Prim]
  ): Result[Offloader.Executable] =
    for {
      parsed <- Identity.parse(identity)
      (rtn, rtnCode) = result.fold((p.Type.Unit0: p.Type, 'V'))(prim => (prim.tpe, prim.code))
      specs = args.map {
        case StepArg.Scalar(prim) => s"P${prim.code}"
        case StepArg.Array(prim)  => s"A${prim.code}"
        case StepArg.Fn(callable) => specOf(callable.environment)
      }
      argumentTypes = args.map {
        case StepArg.Scalar(prim) => prim.tpe
        case StepArg.Array(prim)  => p.Type.Ptr(prim.tpe, p.Type.Space.Global)
        case StepArg.Fn(callable) => callableType(callable)
      }
      sizes = args.collect {
        case StepArg.Scalar(prim) => p.Program.TypeSize(prim.tpe, prim.size)
        case StepArg.Array(prim)  => p.Program.TypeSize(prim.tpe, prim.size)
      }
      executable <- assemble(
        Assembly(
          config,
          fqcn,
          parsed,
          rtn,
          rtnCode,
          argumentTypes,
          sizes,
          specs,
          args.collect { case StepArg.Fn(callable) => callable },
          'V'
        ),
        classOf[LaunchSite]
      )
    } yield executable

  def treeOf(sources: TreeRemapper.Sources, impl: JSymbol.MethodSymbol): Result[MethodTree] =
    sources.localTrees.get(impl).fold(forwarderOf(sources, impl))(_.success)

  private def forwarderOf(sources: TreeRemapper.Sources, impl: JSymbol.MethodSymbol): Result[MethodTree] =
    sources.env match {
      case javac: JavacProcessingEnvironment =>
        Either.catchNonFatal {
          val context = javac.getContext
          val make    = TreeMaker.instance(context).at(Position.NOPOS)
          val names   = Names.instance(context)
          val forwarder =
            JSymbol.MethodSymbol(
              Flags.STATIC | Flags.SYNTHETIC,
              names.fromString(s"${impl.name}$$ref"),
              impl.`type`,
              impl.owner
            )
          val params = impl.`type`.getParameterTypes.asScala.toList.zipWithIndex.map((tpe, i) =>
            make.Param(names.fromString(s"p$i"), tpe, forwarder)
          )
          forwarder.params = JList.from(params.map(_.sym).asJava)
          val args = params.map(param => make.Ident(param.sym): JCTree.JCExpression)
          val call = make.App(make.QualIdent(impl), JList.from(args.asJava))
          val stmt = if (impl.getReturnType.getKind == TypeKind.VOID) make.Exec(call) else make.Return(call)
          val body = make.Block(0, JList.of[JCTree.JCStatement](stmt))
          make.MethodDef(forwarder, body): MethodTree
        }
      case _ => s"launch callable `${impl.name}` has no source: write it as a lambda expression".fail
    }

  def compileRegion(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      entry: MethodTree,
      fqcn: String
  ): Result[Offloader.Executable] = {
    val entryName = normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
    entry match {
      case decl: JCTree.JCMethodDecl =>
        val impl = decl.sym
        for {
          _        <- if (config.roots.nonEmpty) ().success else "polyregion.library.path is not set".fail
          core     <- Identity.parse(CoreIdentity)
          callable <- lowerCallable(sources, config, entryName, 0, entry, impl, impl.params.size, Mode.Region)
          executable <- assemble(
            Assembly(
              config,
              fqcn,
              core,
              p.Type.Unit0,
              'V',
              List(p.Type.IntS64, callableType(callable)),
              List(p.Program.TypeSize(p.Type.IntS64, 8)),
              List("LJ1", specOf(callable.environment)),
              List(callable),
              callable.result
            ),
            classOf[LaunchRegion]
          )
        } yield executable
      case _ => s"region `$fqcn` is not a method".fail
    }
  }
}
