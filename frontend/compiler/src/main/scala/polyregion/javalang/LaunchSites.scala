package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.{ExpressionTree, MethodInvocationTree, MethodTree, ParenthesizedTree, Tree, TypeCastTree}
import com.sun.tools.javac.code.{Flags, Symbol as JSymbol}
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

  private class StubScanner(
      within: Option[JSymbol.MethodSymbol],
      sites: mutable.ListBuffer[Site],
      references: mutable.ListBuffer[Site]
  ) extends TreeScanner {
    override def visitApply(a: JCTree.JCMethodInvocation): Unit = {
      TreeInfo.symbol(a.meth) match {
        case stub: JSymbol.MethodSymbol =>
          InterfaceStub.identityOf(stub).foreach(id => sites += Site(a, stub, id, within))
        case _ =>
      }
      TreeRemapper.dynamicSym(a).flatMap(TreeRemapper.metafactoryImpl(_)).foreach { impl =>
        InterfaceStub.identityOf(impl).foreach(id => references += Site(a, impl, id, within))
      }
      super.visitApply(a)
    }
  }

  case class Collected(sites: List[Site], references: List[Site])

  def collect(scope: Tree): Collected = scope match {
    case cls: JCTree.JCClassDecl =>
      val sites      = mutable.ListBuffer.empty[Site]
      val references = mutable.ListBuffer.empty[Site]
      cls.defs.asScala.foreach {
        case m: JCTree.JCMethodDecl => m.accept(StubScanner(Option(m.sym), sites, references))
        case _: JCTree.JCClassDecl  =>
        case member                 => member.accept(StubScanner(None, sites, references))
      }
      Collected(sites.toList, references.toList)
    case _ => Collected(Nil, Nil)
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

  def callables(site: Site): List[JSymbol.MethodSymbol] =
    site.call.getArguments.asScala.toList.flatMap(lambdaOf(_).map(_._2))

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
            .toRight(
              Inexpressible(
                other match {
                  case p.Type.Ptr(_, _) => s"result of type `${other.repr}` escapes the region"
                  case _                => s"result of type `${other.repr}` cannot be returned from the region"
                }
              )
            )
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
        config,
        fqcn,
        identity,
        rtn,
        rtnCode,
        argumentTypes,
        sizes,
        specs,
        lowered.map(_._2),
        'V',
        classOf[LaunchSite]
      )
    } yield executable
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
        config,
        fqcn,
        parsed,
        rtn,
        rtnCode,
        argumentTypes,
        sizes,
        specs,
        args.collect { case StepArg.Fn(callable) => callable },
        'V',
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
    val hostOnly  = config.copy(targets = LaunchConfig.Default.targets)
    entry match {
      case decl: JCTree.JCMethodDecl =>
        val impl = decl.sym
        for {
          _        <- if (config.roots.nonEmpty) ().success else "polyregion.library.path is not set".fail
          core     <- Identity.parse(CoreIdentity)
          callable <- lowerCallable(sources, hostOnly, entryName, 0, entry, impl, impl.params.size, Mode.Region)
          executable <- assemble(
            hostOnly,
            fqcn,
            core,
            p.Type.Unit0,
            'V',
            List(p.Type.IntS64, callableType(callable)),
            List(p.Program.TypeSize(p.Type.IntS64, 8)),
            List("LJ1", specOf(callable.environment)),
            List(callable),
            callable.result,
            classOf[LaunchRegion]
          )
        } yield executable
      case _ => s"region `$fqcn` is not a method".fail
    }
  }
}
