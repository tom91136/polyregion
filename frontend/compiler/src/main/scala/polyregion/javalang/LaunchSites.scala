package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.{ExpressionTree, MethodInvocationTree, ParenthesizedTree, Tree, TypeCastTree}
import com.sun.tools.javac.code.Symbol as JSymbol
import com.sun.tools.javac.tree.{JCTree, TreeInfo, TreeScanner}
import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.pass.*
import polyregion.intrinsic.IntrinsicTable
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p, *}
import polyregion.jvm.compiler as ct
import polyregion.scalalang.LaunchConfig

import java.io.{ByteArrayOutputStream, DataOutputStream}
import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, TypeElement}
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object LaunchSites {

  final val InterfaceAnnotation = "polyregion.javalang.Interface"
  private val ContextType       = "polyregion.jvm.runtime.Context"

  private final case class Prim(code: Char, tpe: p.Type, size: Int)

  private val Prims: Map[TypeKind, Prim] = Map(
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

  private val PrimOfType: Map[p.Type, Prim] = Prims.values.map(prim => prim.tpe -> prim).toMap

  private enum Kind {
    case Context
    case Scalar(prim: Prim)
    case Array(prim: Prim)
    case Callable
  }

  final case class Site(
      call: JCTree.JCMethodInvocation,
      stub: JSymbol.MethodSymbol,
      identity: String,
      within: Option[JSymbol.MethodSymbol]
  )

  def identityOf(sym: JSymbol): Option[String] =
    sym.getAnnotationMirrors.asScala.collectFirst {
      case annotation if annotation.getAnnotationType.toString == InterfaceAnnotation =>
        annotation.getElementValues.asScala.values.headOption.map(_.getValue.toString)
    }.flatten

  def collect(scope: Tree): List[Site] = scope match {
    case tree: JCTree =>
      val found                                 = mutable.ListBuffer.empty[Site]
      var current: Option[JSymbol.MethodSymbol] = None
      tree.accept(new TreeScanner {
        override def visitMethodDef(m: JCTree.JCMethodDecl): Unit = {
          val outer = current
          current = Option(m.sym)
          super.visitMethodDef(m)
          current = outer
        }
        override def visitApply(a: JCTree.JCMethodInvocation): Unit = {
          TreeInfo.symbol(a.meth) match {
            case stub: JSymbol.MethodSymbol => identityOf(stub).foreach(id => found += Site(a, stub, id, current))
            case _                          =>
          }
          super.visitApply(a)
        }
      })
      found.toList
    case _ => Nil
  }

  private def peel(tree: ExpressionTree): ExpressionTree = tree match {
    case par: ParenthesizedTree => peel(par.getExpression)
    case cast: TypeCastTree     => peel(cast.getExpression)
    case other                  => other
  }

  private def lambdaOf(tree: ExpressionTree): Option[(MethodInvocationTree, JSymbol.MethodSymbol)] =
    peel(tree) match {
      case mi: MethodInvocationTree =>
        TreeRemapper.dynamicSym(mi).flatMap(TreeRemapper.metafactoryImpl(_)).map(mi -> _)
      case _ => None
    }

  def callables(site: Site): List[JSymbol.MethodSymbol] =
    site.call.getArguments.asScala.toList.flatMap(lambdaOf(_).map(_._2))

  private def classify(tpe: TypeMirror): Result[Kind] = tpe.getKind match {
    case kind if Prims.contains(kind) => Kind.Scalar(Prims(kind)).success
    case TypeKind.ARRAY =>
      Prims
        .get(tpe.asInstanceOf[ArrayType].getComponentType.getKind)
        .map(Kind.Array(_))
        .toRight(CompilerException(s"launch argument of type `$tpe` is not supported"))
    case TypeKind.DECLARED =>
      tpe.asInstanceOf[DeclaredType].asElement match {
        case elem: TypeElement =>
          val name = elem.getQualifiedName.toString
          if (name == ContextType) Kind.Context.success
          else
            Boxed
              .get(name)
              .map(kind => Kind.Scalar(Prims(kind)))
              .orElse(Option.when(elem.getKind == ElementKind.INTERFACE)(Kind.Callable))
              .toRight(CompilerException(s"launch argument of type `$tpe` is not supported"))
        case _ => s"launch argument of type `$tpe` is not supported".fail
      }
    case _ => s"launch argument of type `$tpe` is not supported".fail
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
      kinds <- params.traverse(classify)
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
        .map((_, impl) => s"@${impl.owner.flatName.toString.replace('.', '/')}.${impl.name}")
        .toRight(CompilerException("a launch callable must be a lambda expression"))
    case Argument(_, Kind.Context) => "".success
  }

  def nameOf(site: Site): Result[String] =
    for {
      arguments <- argumentsOf(site)
      parts     <- arguments.traverse(partOf)
    } yield LaunchSite.nameOf(site.identity, parts.asJava)

  private def normalise(s: String): String = s.map(c => if (c.isLetterOrDigit || c == '_') c else '_')

  private case class Callable(
      function: p.Function,
      dependencies: List[p.Function],
      environment: Option[Environment],
      defs: List[p.StructDef]
  )

  private case class Field(offset: Int, kind: Char, capture: Int)

  private case class Environment(definition: p.StructDef, size: Int, fields: List[Field])

  private def kindOfCapture(tpe: p.Type): Option[Char] = tpe match {
    case p.Type.Ptr(component, _) if PrimOfType.contains(component) => Some('a')
    case other                                                      => PrimOfType.get(other).map(_.code)
  }

  private def layoutOf(definition: p.StructDef, config: LaunchConfig): Result[ct.Layout] = {
    val compiler = ct.Compiler.create()
    try
      for {
        encoded <- Either.catchNonFatal(MsgPack.encode(CodeGen.structDefsVersioned(List(definition))))
        layouts <- (LaunchConfig.Default.targets ::: config.targets).distinct.traverse { target =>
          Either.catchNonFatal(
            s"${target.name}@${target.arch}" ->
              compiler.layoutOf(encoded, ct.Options.of(compiler.targetByName(target.name), target.arch)).head
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

  private def lowerCallable(
      sources: TreeRemapper.Sources,
      config: LaunchConfig,
      entryName: String,
      index: Int,
      call: MethodInvocationTree,
      impl: JSymbol.MethodSymbol
  ): Result[Callable] = {
    val symbol = s"${entryName}_callable$index"
    for {
      _ <-
        if (impl.isStatic) ().success
        else "a launch callable cannot use `this`".fail
      tree <- sources.localTrees
        .get(impl)
        .toRight(CompilerException(s"launch callable `${impl.name}` has no source: write it as a lambda expression"))
      captured = call.getArguments.size
      region    <- TreeRemapper.compileRegion(sources, tree)
      raw       <- Offloader.withFieldStructs(region.program, sources.classTable, region.outerRefs)
      rawEntry  <- raw.entry.failIfEmpty(s"launch callable `$symbol` has no entry")
      optimised <- Offloader.optimise(region, raw, rawEntry, symbol, PluginEntry.defaultLog)
      (_, program) = optimised
      entry <- program.entry.failIfEmpty(s"launch callable `$symbol` lost its entry")
      origins <- entry.decl.termCaptures.traverse { arg =>
        region.captures
          .find(_.name == arg.named.symbol)
          .map(arg -> _.capture)
          .toRight(CompilerException(s"launch callable parameter `${arg.named.symbol}` has no source"))
      }
      _ <-
        if (entry.moduleCaptures.isEmpty && entry.decl.args.isEmpty) ().success
        else s"launch callable `$symbol` captures modules".fail
      split <- origins.foldLeft[Result[(List[(p.Arg, Int, Boolean)], List[(p.Arg, Int)])]]((Nil, Nil).success) {
        case (acc, (arg, capture)) =>
          acc.flatMap { (envs, params) =>
            capture match {
              case TreeRemapper.Capture.Arg(i) if i < captured    => (envs :+ (arg, i, false), params).success
              case TreeRemapper.Capture.Arg(i)                    => (envs, params :+ (arg, i)).success
              case TreeRemapper.Capture.Length(i) if i < captured => (envs :+ (arg, i, true), params).success
              case other => s"launch callable `${impl.name}` uses an unsupported capture $other".fail
            }
          }
      }
      (envArgs, paramArgs) = split
      fields <- envArgs.traverse { (arg, i, isLength) =>
        kindOfCapture(arg.named.tpe)
          .map(kind => (arg.named, if (isLength) 'l' else kind, i))
          .toRight(
            CompilerException(
              s"launch callable capture `${arg.named.symbol}` of type `${arg.named.tpe.repr}` is not supported"
            )
          )
      }
      environment <-
        if (fields.isEmpty) None.success
        else {
          val definition = p.StructDef(p.Sym(s"${entryName}_env$index"), Nil, fields.map(_._1), Nil)
          layoutOf(definition, config).map { layout =>
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
        p.Named("this", p.Type.Ptr(p.Type.Struct(e.definition.name, Nil), p.Type.Space.Global))
      )
      loads = self.toList.flatMap { s =>
        fields.map((named, _, _) =>
          p.Stmt.Var(
            named,
            Some(p.Expr.Alias(p.Term.Select(s, List(p.PathStep.Deref, p.PathStep.Field(named.symbol)), named.tpe)))
          )
        )
      }
    } yield Callable(
      entry.copy(
        decl = entry.decl.copy(
          name = p.Sym(symbol),
          receiver = self.map(p.Arg(_)),
          args = paramArgs.sortBy(_._2).map(_._1),
          termCaptures = Nil,
          affinity = p.Function.Affinity.Host
        ),
        body = loads ::: entry.body,
        visibility = p.Function.Visibility.Internal,
        convention = p.CallConvention.RegularCall
      ),
      program.functions,
      environment,
      program.defs
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
    import polyregion.ast.MsgPack.Codec.given
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
            .flatMap((call, impl) => lowerCallable(sources, config, entryName, i, call, impl).map(i -> _))
      }
      byIndex = lowered.toMap
      argumentTypes = arguments.zipWithIndex.map {
        case (Argument(_, Kind.Scalar(prim)), _) => prim.tpe
        case (Argument(_, Kind.Array(prim)), _)  => p.Type.Ptr(prim.tpe, p.Type.Space.Global)
        case (_, i) =>
          byIndex(i).environment match {
            case None              => p.Type.FnRef(byIndex(i).function.name)
            case Some(environment) => p.Type.Struct(environment.definition.name, Nil)
          }
      }
      environments = lowered.flatMap(_._2.environment)
      typeSizes = environments.map(e => p.Program.TypeSize(p.Type.Struct(e.definition.name, Nil), e.size)) ++
        arguments.collect {
          case Argument(_, Kind.Scalar(prim)) => p.Program.TypeSize(prim.tpe, prim.size)
          case Argument(_, Kind.Array(prim))  => p.Program.TypeSize(prim.tpe, prim.size)
        } ++ PrimOfType.get(rtn).map(prim => p.Program.TypeSize(prim.tpe, prim.size))
      functions = IntrinsicTable.dedupe(
        importRoot(entryName, p.Sym(declaration), argumentTypes, rtn) ::
          lowered.flatMap(c => c._2.function :: c._2.dependencies)
      )
      defs    = (environments.map(_.definition) ++ lowered.flatMap(_._2.defs)).distinct
      request = p.Program.LinkRequest(Nil, p.Program(None, functions, defs), Nil, typeSizes.distinct)
      encoded <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request)))
      groups  <- compileBundles(packageName, declaration, entryName, encoded, config)
      descriptor = {
        val specs = arguments.zipWithIndex.map {
          case (Argument(_, Kind.Scalar(prim)), _) => s"P${prim.code}"
          case (Argument(_, Kind.Array(prim)), _)  => s"A${prim.code}"
          case (_, i) =>
            byIndex(i).environment.fold("C0:") { e =>
              s"C${e.size}:${e.fields.map(f => s"${f.offset}${f.kind}${f.capture}").mkString(",")}"
            }
        }
        s"$entryName|$rtnCode|${specs.mkString(";")}"
      }
      bytes <- Offloader.emitClass(fqcn, encodeGroups(groups), descriptor, classOf[LaunchSite])
    } yield Offloader.Executable(fqcn, bytes)
  }
}
