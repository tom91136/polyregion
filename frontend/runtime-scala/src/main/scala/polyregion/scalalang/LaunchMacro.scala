package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.{MsgPack, PolyAST as p, *}
import polyregion.jvm.twin.{Layout, Marshal}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.prism.StdLib

import java.util.concurrent.atomic.AtomicLong
import scala.quoted.*
import scala.util.chaining.*

object LaunchMacro {

  private val Counter = AtomicLong(0)

  private case class Argument(expr: Expr[Any], kind: Kind)

  private case class Environment(definition: p.StructDef, size: Int, fields: List[(Int, Kind, Expr[Any])])

  private case class Callable(function: p.Function, environment: Option[Environment])

  private enum Kind {
    case Callable
    case Prim(tpe: p.Type, size: Int, name: String)
    case Arr(component: p.Type, size: Int, name: String)
  }

  private def prims(using q: Quotes): List[(q.reflect.TypeRepr, p.Type, Int, String)] = {
    import q.reflect.*
    List(
      (TypeRepr.of[Int], p.Type.IntS32, 4, "int"),
      (TypeRepr.of[Long], p.Type.IntS64, 8, "long"),
      (TypeRepr.of[Float], p.Type.Float32, 4, "float"),
      (TypeRepr.of[Double], p.Type.Float64, 8, "double"),
      (TypeRepr.of[Short], p.Type.IntS16, 2, "short"),
      (TypeRepr.of[Byte], p.Type.IntS8, 1, "byte"),
      (TypeRepr.of[Char], p.Type.IntU16, 2, "char"),
      (TypeRepr.of[Boolean], p.Type.Bool1, 1, "boolean")
    )
  }

  private def classify(using q: Quotes)(expr: Expr[Any]): Result[Argument] = {
    import q.reflect.*
    val tpe = expr.asTerm.tpe.widen.dealias.simplified
    def array(of: TypeRepr) = prims.collectFirst {
      case (repr, component, size, name) if TypeRepr.of[Array].appliedTo(repr) =:= of => Kind.Arr(component, size, name)
    }
    if (tpe.typeSymbol.fullName.startsWith("scala.Function")) Argument(expr, Kind.Callable).success
    else
      prims
        .collectFirst { case (repr, t, size, name) if tpe <:< repr => Kind.Prim(t, size, name) }
        .orElse(array(tpe))
        .map(Argument(expr, _))
        .toRight(CompilerException(s"launch argument of type ${tpe.show} is not supported"))
  }

  def generate[R: Type](identity: Expr[String], context: Expr[rt.Context], args: Expr[Seq[Any]])(using
      q: Quotes
  ): Expr[R] = {
    val Q = Quoted(q)
    import q.reflect.*
    val result = for {
      qualified <- identity.value.toRight(CompilerException("launch identity is not a constant"))
      (packageName, declaration) <- qualified.split(":", 2).toList match {
        case a :: b :: Nil => (a, b).success
        case _             => s"malformed launch identity `$qualified`".fail
      }
      exprs <- args match {
        case Varargs(xs) => xs.toList.success
        case _           => "launch arguments are not a literal sequence".fail
      }
      arguments <- exprs.traverse(expr => classify(expr))
      config    <- LaunchConfig.fromSettings(LaunchConfig.macroSettings).leftMap(CompilerException(_))
      generated <- build[R](using Q)(packageName, declaration, context, arguments, config)
    } yield generated
    result.fold(
      error => report.errorAndAbort(Option(error.getMessage).getOrElse(error.toString)),
      generated => generated
    )
  }

  private def build[R: Type](using
      q: Quoted
  )(
      packageName: String,
      declaration: String,
      context: Expr[rt.Context],
      arguments: List[Argument],
      config: LaunchConfig
  ): Result[Expr[R]] = {
    given Quotes  = q.underlying
    val id        = Counter.getAndIncrement()
    val entryName = s"__polyregion_scala_launch_${id}_${System.nanoTime().toHexString}"
    val log       = RenderedLog("")
    for {
      callables <- arguments.zipWithIndex
        .collect { case (Argument(expr, Kind.Callable), index) => index -> expr }
        .traverse { case (index, expr) => compileCallable(log, config, entryName, index, expr).map(index -> _) }
      callableByIndex = callables.toMap
      rtn <- returnType(Type.of[R])
      argumentTypes = arguments.zipWithIndex.map {
        case (Argument(_, Kind.Callable), index) =>
          callableByIndex(index) match {
            case Callable(function, None)       => p.Type.FnRef(function.name)
            case Callable(_, Some(environment)) => p.Type.Struct(environment.definition.name, Nil)
          }
        case (Argument(_, Kind.Prim(tpe, _, _)), _) => tpe
        case (Argument(_, Kind.Arr(component, _, _)), _) =>
          p.Type.Ptr(component, p.Type.Space.Global)
      }
      environments = callables.flatMap(_._2.environment)
      typeSizes = environments.map(e => p.Program.TypeSize(p.Type.Struct(e.definition.name, Nil), e.size)) ++ arguments
        .collect {
          case Argument(_, Kind.Prim(tpe, size, _))      => p.Program.TypeSize(tpe, size)
          case Argument(_, Kind.Arr(component, size, _)) => p.Program.TypeSize(component, size)
        } ++ rtn.collect { case (tpe, size, _) => p.Program.TypeSize(tpe, size) }
      root = importRoot(entryName, p.Sym(declaration), argumentTypes, rtn.fold(p.Type.Unit0: p.Type)(_._1))
      request = p.Program.LinkRequest(
        Nil,
        p.Program(None, root :: callables.map(_._2.function), environments.map(_.definition)),
        Nil,
        typeSizes.distinct
      )
      bytes   <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request)))
      bundles <- compileBundles(packageName, declaration, entryName, bytes, config)
    } yield emit[R](
      context,
      entryName,
      arguments,
      callableByIndex,
      Layout.encodeAll(java.util.List.of()),
      bundles,
      rtn.map(_._3)
    )
  }

  private def returnType(using q: Quotes)(tpe: Type[?]): Result[Option[(p.Type, Int, String)]] = {
    import q.reflect.*
    val repr = TypeRepr.of(using tpe).widen.dealias
    if (repr =:= TypeRepr.of[Unit]) None.success
    else
      prims
        .collectFirst { case (r, t, size, name) if repr =:= r => (t, size, name) }
        .map(Some(_))
        .toRight(CompilerException(s"launch result type ${repr.show} is not supported"))
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

  private def compileCallable(using
      q: Quoted
  )(
      log: Log,
      config: LaunchConfig,
      entryName: String,
      index: Int,
      expr: Expr[Any]
  ): Result[Callable] =
    Compiler.compileExpr(log, expr, StdLib.Tables, config, callable = true).flatMap { (captures, _, _, programs) =>
      for {
        program <-
          if (programs.map(_._2).distinct.size == 1) programs.head._2.success
          else "a launch callable resolves interface calls differently per target, which is not supported".fail
        entry <- program.entry.toRight(CompilerException("callable has no entry"))
        _ <-
          if (entry.moduleCaptures.nonEmpty)
            s"callables capturing modules are not supported: ${entry.moduleCaptures.map(_.named.repr)}".fail
          else ().success
        named = captures.map(_._1)
        terms = captures.map(_._2)
        values <- terms.traverse[Result, (Kind, Expr[Any])] { term =>
          classify(using q.underlying)(term.asExpr).flatMap {
            case Argument(value, kind @ (Kind.Prim(_, _, _) | Kind.Arr(_, _, _))) => (kind, value).success
            case _ => s"callable capture of type ${term.tpe.show} is not supported".fail
          }
        }
        name = p.Sym(s"${entryName}_callable$index")
        environment <-
          if (named.isEmpty) None.success
          else {
            val definition = p.StructDef(p.Sym(s"${entryName}_env$index"), Nil, named, Nil)
            layoutOf(definition, config).map { layout =>
              val offsets = layout.members.map(m => m.name -> m.offsetInBytes.toInt).toMap
              Some(
                Environment(
                  definition,
                  layout.sizeInBytes.toInt,
                  named.zip(values).map { case (capture, (kind, value)) => (offsets(capture.symbol), kind, value) }
                )
              )
            }
          }
        receiver = environment
          .map(e => p.Arg(p.Named("this", p.Type.Ptr(p.Type.Struct(e.definition.name, Nil), p.Type.Space.Global))))
        loads = environment.toList.flatMap { e =>
          val self = p.Named("this", p.Type.Ptr(p.Type.Struct(e.definition.name, Nil), p.Type.Space.Global))
          named.map(capture =>
            p.Stmt.Var(
              capture,
              Some(
                p.Expr.Alias(p.Term.Select(self, List(p.PathStep.Deref, p.PathStep.Field(capture.symbol)), capture.tpe))
              )
            )
          )
        }
      } yield Callable(
        entry.copy(
          decl =
            entry.decl.copy(name = name, receiver = receiver, termCaptures = Nil, affinity = p.Function.Affinity.Host),
          body = loads ::: entry.body,
          visibility = p.Function.Visibility.Internal,
          convention = p.CallConvention.RegularCall
        ),
        environment
      )
    }

  private[scalalang] def layoutOf(definition: p.StructDef, config: LaunchConfig): Result[ct.Layout] = {
    val compiler = ct.Compiler.create()
    try
      for {
        encoded <- Either.catchNonFatal(MsgPack.encode(CodeGen.structDefsVersioned(List(definition))))
        layouts <- (LaunchConfig.Default.targets ::: config.targets).distinct.traverse {
          case LaunchConfig.DeviceTarget(name, arch) =>
            Either.catchNonFatal {
              val layout = compiler.layoutOf(encoded, ct.Options.of(compiler.targetByName(name), arch)).head
              s"$name@$arch" -> layout
            }
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

  private def compileBundles(
      packageName: String,
      declaration: String,
      entryName: String,
      request: Array[Byte],
      config: LaunchConfig
  ): Result[List[(String, Array[Byte], p.Compile.Bundle)]] = {
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
        bundles <- raw.toList.traverse(bytes =>
          MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](bytes).map(versioned => bytes -> versioned.t)
        )
        usable = bundles.filter(_._2.unfit.isEmpty)
        _ <-
          if (usable.isEmpty)
            s"`$declaration` cannot be linked for the configured targets: ${bundles.flatMap(_._2.unfit).distinct.mkString("; ")}".fail
          else ().success
      } yield usable.zipWithIndex.map { case ((bytes, bundle), group) => (s"$entryName@g$group", bytes, bundle) }
    finally compiler.close()
  }

  private def emit[R: Type](using
      Quotes
  )(
      context: Expr[rt.Context],
      entryName: String,
      arguments: List[Argument],
      callables: Map[Int, Callable],
      layouts: String,
      groups: List[(String, Array[Byte], p.Compile.Bundle)],
      rtn: Option[String]
  ): Expr[R] = {
    val entry = Expr(entryName)
    val groupExprs = groups.map { (program, bytes, bundle) =>
      val chunks = java.util.Base64.getEncoder.encodeToString(bytes).grouped(30000).toList.map(Expr(_))
      '{
        LaunchRuntime.Group(
          ${ Expr(program) },
          ${ Expr.ofList(chunks) }.toArray,
          ${ Expr(bundle.remoteModules.isEmpty) }
        )
      }
    }
    def staged(marshal: Expr[Marshal]): List[Expr[LaunchRuntime.Arg]] = arguments.zipWithIndex.collect {
      case (Argument(expr, Kind.Prim(_, _, name)), _) => primitive(marshal, expr, name)
      case (Argument(expr, Kind.Arr(_, _, _)), _) => '{ LaunchRuntime.array($marshal, ${ expr.asExprOf[Array[?]] }) }
      case (Argument(_, Kind.Callable), index) if callables(index).environment.nonEmpty =>
        environment(marshal, callables(index).environment.get)
    }
    def resultCell(marshal: Expr[Marshal]): Option[Expr[LaunchRuntime.Arg]] = rtn.map { name =>
      val size = prims.collectFirst { case (_, _, size, n) if n == name => size }.get
      '{ LaunchRuntime.result($marshal, ${ Expr(size) }) }
    }
    '{
      val ctx     = $context
      val site    = LaunchRuntime.site($entry, ${ Expr.ofList(groupExprs) }.toArray, ${ Expr(layouts) })
      val marshal = LaunchRuntime.marshal(site)
      try {
        val args = ${ Expr.ofList(staged('marshal)) }
        ${
          resultCell('marshal) match {
            case None =>
              '{
                LaunchRuntime.invoke(ctx, site, marshal, args.toArray)
                ().asInstanceOf[R]
              }
            case Some(cell) =>
              '{
                val cellArg = $cell
                LaunchRuntime.invoke(ctx, site, marshal, (args :+ cellArg).toArray)
                ${ readResult(rtn.get, 'cellArg) }.asInstanceOf[R]
              }
          }
        }
      } finally marshal.close()
    }
  }

  private def environment(using Quotes)(marshal: Expr[Marshal], env: Environment): Expr[LaunchRuntime.Arg] =
    env.fields
      .foldLeft('{ LaunchRuntime.environment($marshal, ${ Expr(env.size) }) }) {
        case (builder, (offset, kind, value)) =>
          kind match {
            case Kind.Arr(_, _, _)     => '{ $builder.array(${ Expr(offset) }, ${ value.asExprOf[Array[?]] }) }
            case Kind.Prim(_, _, name) => putPrimitive(builder, Expr(offset), value, name)
            case Kind.Callable         => builder
          }
      }
      .pipe(builder => '{ $builder.done })

  private def putPrimitive(using
      Quotes
  )(
      builder: Expr[LaunchRuntime.EnvironmentBuilder],
      offset: Expr[Int],
      value: Expr[Any],
      name: String
  ): Expr[LaunchRuntime.EnvironmentBuilder] =
    name match {
      case "int"     => '{ $builder.int($offset, ${ value.asExprOf[Int] }) }
      case "long"    => '{ $builder.long($offset, ${ value.asExprOf[Long] }) }
      case "float"   => '{ $builder.float($offset, ${ value.asExprOf[Float] }) }
      case "double"  => '{ $builder.double($offset, ${ value.asExprOf[Double] }) }
      case "short"   => '{ $builder.short($offset, ${ value.asExprOf[Short] }) }
      case "byte"    => '{ $builder.byte($offset, ${ value.asExprOf[Byte] }) }
      case "char"    => '{ $builder.char($offset, ${ value.asExprOf[Char] }) }
      case "boolean" => '{ $builder.boolean($offset, ${ value.asExprOf[Boolean] }) }
    }

  private def primitive(using
      Quotes
  )(marshal: Expr[Marshal], expr: Expr[Any], name: String): Expr[LaunchRuntime.Arg] = name match {
    case "int"     => '{ LaunchRuntime.int($marshal, ${ expr.asExprOf[Int] }) }
    case "long"    => '{ LaunchRuntime.long($marshal, ${ expr.asExprOf[Long] }) }
    case "float"   => '{ LaunchRuntime.float($marshal, ${ expr.asExprOf[Float] }) }
    case "double"  => '{ LaunchRuntime.double($marshal, ${ expr.asExprOf[Double] }) }
    case "short"   => '{ LaunchRuntime.short($marshal, ${ expr.asExprOf[Short] }) }
    case "byte"    => '{ LaunchRuntime.byte($marshal, ${ expr.asExprOf[Byte] }) }
    case "char"    => '{ LaunchRuntime.char($marshal, ${ expr.asExprOf[Char] }) }
    case "boolean" => '{ LaunchRuntime.boolean($marshal, ${ expr.asExprOf[Boolean] }) }
  }

  private def readResult(using Quotes)(name: String, cell: Expr[LaunchRuntime.Arg]): Expr[Any] = name match {
    case "int"     => '{ LaunchRuntime.resultInt($cell) }
    case "long"    => '{ LaunchRuntime.resultLong($cell) }
    case "float"   => '{ LaunchRuntime.resultFloat($cell) }
    case "double"  => '{ LaunchRuntime.resultDouble($cell) }
    case "short"   => '{ LaunchRuntime.resultShort($cell) }
    case "byte"    => '{ LaunchRuntime.resultByte($cell) }
    case "char"    => '{ LaunchRuntime.resultChar($cell) }
    case "boolean" => '{ LaunchRuntime.resultBoolean($cell) }
  }
}
