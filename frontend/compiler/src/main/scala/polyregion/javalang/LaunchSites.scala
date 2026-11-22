package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.{ExpressionTree, MethodInvocationTree, MethodTree, ParenthesizedTree, Tree, TypeCastTree}
import com.sun.tools.javac.code.{Flags, Symbol as JSymbol}
import com.sun.tools.javac.processing.JavacProcessingEnvironment
import com.sun.tools.javac.util.{List as JList, Names, Position}
import com.sun.tools.javac.tree.{JCTree, TreeInfo, TreeMaker, TreeScanner}
import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.pass.*
import polyregion.intrinsic.IntrinsicTable
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p, given, *}
import polyregion.ast.Traversal.*
import polyregion.jvm.compiler as ct
import polyregion.jvm.twin.Layout
import polyregion.scalalang.LaunchConfig

import java.io.{ByteArrayOutputStream, DataOutputStream}
import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, TypeElement}
import scala.annotation.tailrec
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object LaunchSites {

  private val CoreIdentity = "polyregion.core:polyregion.core.for_each"
  private val ContextType  = "polyregion.jvm.runtime.Context"

  case class Prim(code: Char, tpe: p.Type, size: Int)

  val Prims: Map[TypeKind, Prim] = Map(
    TypeKind.INT     -> Prim('I', p.Type.IntS32, 4),
    TypeKind.LONG    -> Prim('J', p.Type.IntS64, 8),
    TypeKind.FLOAT   -> Prim('F', p.Type.Float32, 4),
    TypeKind.DOUBLE  -> Prim('D', p.Type.Float64, 8),
    TypeKind.SHORT   -> Prim('S', p.Type.IntS16, 2),
    TypeKind.BYTE    -> Prim('B', p.Type.IntS8, 1),
    TypeKind.CHAR    -> Prim('C', p.Type.IntU16, 2),
    TypeKind.BOOLEAN -> Prim('Z', p.Type.Bool1, 1)
  )

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

  val PrimOfType: Map[p.Type, Prim] = Prims.values.map(prim => prim.tpe -> prim).toMap

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

  def normalise(s: String): String = s.map(c => if (c.isLetterOrDigit || c == '_') c else '_')

  class Inexpressible(message: String) extends Exception(message)

  private def inexpressible(message: String): Result[Nothing] = Left(Inexpressible(message))

  enum Mode {
    case Site
    case Region
  }

  case class Callable(
      function: p.Function,
      dependencies: List[p.Function],
      environment: Option[Environment],
      defs: List[p.StructDef],
      result: Char,
      layouts: List[Layout],
      twins: List[TwinEmitter.Emitted]
  )

  case class Field(offset: Int, kind: Char, source: String)

  case class Environment(definition: p.StructDef, size: Int, fields: List[Field])

  private def kindOfCapture(tpe: p.Type, mode: Mode): Option[Char] = tpe match {
    case p.Type.Ptr(component, _) if mode == Mode.Region || PrimOfType.contains(component) => Some('a')
    case other => PrimOfType.get(other).map(_.code)
  }

  private def layoutOf(definition: p.StructDef, others: List[p.StructDef], config: LaunchConfig): Result[ct.Layout] = {
    val compiler = ct.Compiler.create()
    try
      for {
        encoded <- Either.catchNonFatal(MsgPack.encode(CodeGen.structDefsVersioned(others :+ definition)))
        layouts <- (LaunchConfig.Default.targets ::: config.targets).distinct.traverse { target =>
          Either.catchNonFatal(
            s"${target.name}@${target.arch}" ->
              compiler
                .layoutOf(encoded, ct.Options.of(compiler.targetByName(target.name), target.arch))
                .find(_.name == definition.name.repr)
                .get
          )
        }
        shape = (layout: ct.Layout) =>
          (
            layout.sizeInBytes,
            layout.alignment,
            layout.members.map(m => (m.name, m.offsetInBytes, m.sizeInBytes)).toList
          )
        agreed <-
          if (layouts.map(entry => shape(entry._2)).distinct.size == 1) layouts.head._2.success
          else
            s"the layout of `${definition.name.repr}` differs between the targets ${layouts.map(_._1).mkString(", ")}".fail
      } yield agreed
    finally compiler.close()
  }

  private def stored(named: p.Named): p.Named = named.tpe match {
    case p.Type.Ptr(referent @ p.Type.Struct(_, _), p.Type.Space.Private) =>
      named.copy(tpe = p.Type.Ptr(referent, p.Type.Space.Global))
    case _ => named
  }

  private def withUniqueNames(functions: List[p.Function]): List[p.Function] = {
    val overloaded = functions.groupBy(_.name).filter(_._2.map(_.decl).distinct.sizeIs > 1)
    if (overloaded.isEmpty) functions
    else {
      val renamed = overloaded.map { (name, group) =>
        name -> group.distinctBy(_.decl).zipWithIndex.map((fn, i) => fn.decl -> (name :+ s"$$$i"))
      }
      def choose(name: p.Sym, args: List[p.Type], rtn: p.Type): p.Sym = {
        val candidates = overloaded(name).distinctBy(_.decl)
        val arity      = args.size
        val found = candidates
          .find(fn => fn.args.map(_.named.tpe) == args && fn.rtn == rtn)
          .orElse(candidates.find(fn => fn.args.sizeIs == arity && fn.rtn == rtn))
          .getOrElse(candidates.head)
        renamed(name).find(_._1 == found.decl).get._2
      }
      functions.map { fn =>
        val rewritten = fn.modifyAll[p.Expr] {
          case invoke @ p.Expr.Invoke(p.Type.FnRef(name), _, _, args, rtn) if overloaded.contains(name) =>
            invoke.copy(callee = p.Type.FnRef(choose(name, args.map(_.tpe), rtn)))
          case other => other
        }
        if (overloaded.contains(fn.name))
          rewritten.modifyDecl(_.copy(name = renamed(fn.name).find(_._1 == fn.decl).get._2))
        else rewritten
      }
    }
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
      optimised <- Offloader.optimise(region, raw, routed, symbol, PluginEntry.defaultLog)
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
          Offloader.twinsOf(sources, program, monoToPoly, region.tags, entryName)
        else (List.empty[Layout], List.empty[TwinEmitter.Emitted]).success
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
      split <- origins.foldLeft[Result[(List[(p.Named, Char, String)], List[(p.Arg, Int)])]]((Nil, Nil).success) {
        case (acc, (arg, capture)) =>
          def env(kind: Option[Char], source: String) =
            kind
              .map(k => (arg.named, k, source))
              .toRight(
                Inexpressible(
                  s"launch callable capture `${arg.named.symbol}` of type `${arg.named.tpe.repr}` is not supported"
                )
              )
          acc.flatMap { (envs, params) =>
            capture match {
              case TreeRemapper.Capture.Arg(i) if mode == Mode.Region || i < captured =>
                env(kindOfCapture(arg.named.tpe, mode), i.toString).map(e => (envs :+ e, params))
              case TreeRemapper.Capture.Arg(i) => (envs, params :+ (arg, i)).success
              case TreeRemapper.Capture.Length(i) if mode == Mode.Region || i < captured =>
                env(kindOfCapture(arg.named.tpe, mode).map(_ => 'l'), i.toString).map(e => (envs :+ e, params))
              case TreeRemapper.Capture.RowLengths(i) if mode == Mode.Region =>
                env(Some('R'), i.toString).map(e => (envs :+ e, params))
              case TreeRemapper.Capture.StaticField(owner, name) if mode == Mode.Region =>
                env(Some('s'), s"$owner#$name").map(e => (envs :+ e, params))
              case other => inexpressible(s"launch callable `${impl.name}` uses an unsupported capture $other")
            }
          }
      }
      (envFields, paramArgs) = split
      receiver               = entry.decl.receiver.map(r => (r.named, 'a', "0")).toList
      fields                 = receiver ++ envFields ++ results.map(arg => (arg.named, 'r', "0"))
      environment <-
        if (fields.isEmpty) None.success
        else {
          val definition = p.StructDef(p.Sym(s"${entryName}_env$index"), Nil, fields.map(f => stored(f._1)), Nil)
          layoutOf(definition, program.defs, config).map { layout =>
            val offsets = layout.members.map(m => m.name -> m.offsetInBytes.toInt).toMap
            Some(
              Environment(
                definition,
                layout.sizeInBytes.toInt,
                fields.map((named, kind, i) => Field(offsets(named.symbol), kind, i))
              )
            )
          }
        }
      self = environment.map(e =>
        p.Named("#env", p.Type.Ptr(p.Type.Struct(e.definition.name, Nil), p.Type.Space.Global))
      )
      loads = self.toList.flatMap { s =>
        fields.map { (named, _, _) =>
          val steps = List(p.PathStep.Deref, p.PathStep.Field(named.symbol))
          val field = p.Term.Select(s, steps, stored(named).tpe)
          p.Stmt.Var(
            named,
            Some(named.tpe match {
              case p.Type.Ptr(referent @ p.Type.Struct(_, _), p.Type.Space.Private) =>
                p.Expr.RefTo(
                  p.Term.Select(s, steps :+ p.PathStep.Deref, referent),
                  None,
                  referent,
                  p.Type.Space.Private,
                  p.Region.Rooted(s)
                )
              case _ => p.Expr.Alias(field)
            })
          )
        }
      }
    } yield Callable(
      entry.copy(
        decl = entry.decl.copy(
          name = p.Sym(symbol),
          receiver = self.map(p.Arg(_)),
          args = mode match {
            case Mode.Site   => paramArgs.sortBy(_._2).map(_._1)
            case Mode.Region => List(p.Arg(p.Named("#i", p.Type.IntS64)))
          },
          termCaptures = Nil,
          affinity = p.Function.Affinity.Host
        ),
        body = loads ::: entry.body,
        visibility = p.Function.Visibility.Internal,
        convention = p.CallConvention.RegularCall
      ),
      program.functions,
      environment,
      program.defs,
      resultCode,
      twinned._1,
      twinned._2
    )
  }

  private def importRoot(entryName: String, declaration: p.Sym, args: List[p.Type], rtn: p.Type): p.Function =
    p.Function(
      p.FunctionDecl(
        p.Sym(entryName),
        Nil,
        None,
        args.zipWithIndex.map((tpe, index) => p.Arg(p.Named(s"arg$index", tpe))),
        Nil,
        Nil,
        rtn,
        p.Function.Affinity.Host
      ),
      Nil,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall,
      Some(declaration)
    )

  private def compileBundles(
      packageName: String,
      declaration: String,
      entryName: String,
      request: Array[Byte],
      config: LaunchConfig
  ): Result[List[(String, p.Compile.Bundle)]] = {
    val compiler = ct.Compiler.create()
    try
      for {
        raw <- Either.catchNonFatal(
          compiler.compileInterface(
            packageName,
            config.roots,
            declaration,
            entryName,
            request,
            config.capabilityList,
            config.targetList,
            config.opt,
            0
          )
        )
        bundles <- raw.toList.traverse(bytes => MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](bytes).map(_.t))
        usable = bundles.filter(_.unfit.isEmpty)
        _ <-
          if (usable.isEmpty)
            s"`$declaration` cannot be linked for the configured targets: ${bundles.flatMap(_.unfit).distinct.mkString("; ")}".fail
          else ().success
      } yield usable.zipWithIndex.map((bundle, group) => (s"$entryName@g$group", bundle))
    finally compiler.close()
  }

  private def encodeGroups(groups: List[(String, p.Compile.Bundle)]): Array[Byte] = {
    val bytes = ByteArrayOutputStream()
    val out   = DataOutputStream(bytes)
    out.writeInt(groups.size)
    groups.foreach { (program, bundle) =>
      out.writeUTF(program)
      out.writeInt(bundle.hostObject.size)
      out.write(bundle.hostObject.toArray)
      out.writeInt(bundle.remoteModules.size)
      bundle.remoteModules.foreach { module =>
        out.writeUTF(module.moduleName)
        out.writeByte(module.target)
        out.writeInt(module.features.size)
        module.features.foreach(out.writeUTF)
        out.writeInt(module.image.size)
        out.write(module.image.toArray)
      }
    }
    out.flush()
    bytes.toByteArray
  }

  def compile(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      site: Site,
      fqcn: String
  ): Result[Offloader.Executable] = {
    val entryName = normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
    for {
      split <- site.identity.split(":", 2).toList match {
        case a :: b :: Nil => (a, b).success
        case _             => s"malformed launch identity `${site.identity}`".fail
      }
      (packageName, declaration) = split
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
        (packageName, declaration),
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
      split <- identity.split(":", 2).toList match {
        case a :: b :: Nil => (a, b).success
        case _             => s"malformed launch identity `$identity`".fail
      }
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
        split,
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

  private def specOf(environment: Option[Environment]): String =
    environment.fold("C0:") { e =>
      s"C${e.size}:${e.fields.map(f => s"${f.offset}${f.kind}${f.source}").mkString(",")}"
    }

  private def callableType(callable: Callable): p.Type = callable.environment match {
    case None              => p.Type.FnRef(callable.function.name)
    case Some(environment) => p.Type.Struct(environment.definition.name, Nil)
  }

  private def assemble(
      config: LaunchConfig,
      fqcn: String,
      identity: (String, String),
      rtn: p.Type,
      rtnCode: Char,
      argumentTypes: List[p.Type],
      argumentSizes: List[p.Program.TypeSize],
      specs: List[String],
      callables: List[Callable],
      regionResult: Char,
      parent: Class[?]
  ): Result[Offloader.Executable] = {
    val entryName                  = normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
    val (packageName, declaration) = identity
    val environments               = callables.flatMap(_.environment)
    val typeSizes = environments.map(e => p.Program.TypeSize(p.Type.Struct(e.definition.name, Nil), e.size)) ++
      argumentSizes ++ PrimOfType.get(rtn).map(prim => p.Program.TypeSize(prim.tpe, prim.size))
    val functions = IntrinsicTable.dedupe(
      importRoot(entryName, p.Sym(declaration), argumentTypes, rtn) :: callables.flatMap(c =>
        c.function :: c.dependencies
      )
    )
    val defs    = (environments.map(_.definition) ++ callables.flatMap(_.defs)).distinct
    val request = p.Program.LinkRequest(Nil, p.Program(None, functions, defs), Nil, typeSizes.distinct)
    for {
      encoded <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request)))
      groups  <- compileBundles(packageName, declaration, entryName, encoded, config)
      twins   = callables.flatMap(_.twins)
      layouts = Layout.encodeAll(callables.flatMap(_.layouts).asJava)
      descriptor =
        s"$entryName|$rtnCode|${specs.mkString(";")}|$regionResult|$layouts|${twins.map(_.fqcn).mkString(",")}"
      bytes <- Offloader.emitClass(fqcn, encodeGroups(groups), descriptor, parent)
    } yield Offloader.Executable(fqcn, bytes, twins)
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
          callable <- lowerCallable(sources, hostOnly, entryName, 0, entry, impl, impl.params.size, Mode.Region)
          executable <- assemble(
            hostOnly,
            fqcn,
            CoreIdentity.split(":", 2) match { case Array(a, b) => (a, b) },
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
