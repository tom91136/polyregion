package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{MsgPack, PolyAST as p, ProgramLinker, calleeName, given}
import polyregion.scalalang.generated.PolyASTWireSchema

private[scalalang] object InterfaceResolver {

  final case class Import(packageName: String, declaration: String, target: p.Expr.Invoke)

  private def configured(name: String, property: String): Option[String] =
    Option(System.getenv(name)).orElse(Option(System.getProperty(property))).map(_.trim).filter(_.nonEmpty)

  def configuredCapabilityList: String =
    configured("POLYFRONT_LIBRARY_CAPABILITIES", "polyregion.library.capabilities").getOrElse("")

  def configuredCapabilities: Set[String] =
    configuredCapabilityList.split(',').view.map(_.trim).filter(_.nonEmpty).toSet

  private lazy val compiler = polyregion.jvm.compiler.Compiler.create()

  def configuredRoots: String = configured("POLYFRONT_LIBRARY_PATH", "polyregion.library.path").getOrElse("")

  def loadPackage(
      packageName: String,
      roots: String = configuredRoots,
      capabilities: Set[String] = configuredCapabilities
  ): Either[List[String], p.Package] =
    for {
      bytes <-
        try Right(compiler.loadPackage(packageName, roots, capabilities.mkString(",")))
        catch {
          case error: polyregion.jvm.compiler.PolyregionCompilerException =>
            Left(error.getMessage.linesIterator.toList)
        }
      versioned <- MsgPack
        .decode[MsgPack.Versioned[p.Package]](bytes)
        .left
        .map(error => List(s"cannot decode package `$packageName`: ${error.getMessage}"))
      _ <- Either.cond(
        versioned.hash == PolyASTWireSchema.PackageHash,
        (),
        List(s"package `$packageName` has schema ${versioned.hash}, expected ${PolyASTWireSchema.PackageHash}")
      )
    } yield versioned.t

  private def sizeOf(tpe: p.Type): Option[Int] = tpe match {
    case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8     => Some(1)
    case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16 => Some(2)
    case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32 => Some(4)
    case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 => Some(8)
    case p.Type.Unit0                                   => Some(0)
    case p.Type.Ptr(_, _) =>
      Some(System.getProperty("sun.arch.data.model", "64").toIntOption.filter(_ > 0).getOrElse(64) / 8)
    case _ => None
  }

  private def layoutsOf(tpe: p.Type): List[(p.Type, Int)] =
    sizeOf(tpe).map(tpe -> _).toList ::: (tpe match {
      case p.Type.Ptr(component, _) => layoutsOf(component)
      case _                        => Nil
    })

  private def root(declaration: String, target: p.Expr.Invoke): p.Function =
    p.Function(
      p.FunctionDecl(
        target.calleeName,
        Nil,
        target.receiver.map(value => p.Arg(p.Named("this", value.tpe))),
        target.args.zipWithIndex.map((value, index) => p.Arg(p.Named(s"arg$index", value.tpe))),
        Nil,
        Nil,
        target.rtn,
        p.Function.Affinity.Host
      ),
      Nil,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall,
      Some(p.Sym(declaration))
    )

  def importPackages(
      packages: List[p.Package],
      imports: List[Import],
      callers: List[p.Function],
      capabilities: Set[String],
      typeSizes: Map[p.Type, Int]
  ): Either[List[String], (List[p.Function], Set[p.StructDef])] = {
    val layouts =
      imports.flatMap(value => (value.target.args.map(_.tpe) :+ value.target.rtn).flatMap(layoutsOf)).toMap ++ typeSizes
    val roots = imports.map(value => root(value.declaration, value.target))
    val signatures = imports.map { value =>
      value.target.calleeName -> ProgramLinker.CallSignature(
        p.Sym(value.declaration),
        value.target.tpeArgs,
        value.target.receiver.map(_.tpe),
        value.target.args.map(_.tpe),
        value.target.rtn
      )
    }.toMap
    val request = p.Program.LinkRequest(
      packages,
      p.Program(None, roots ::: callers, Nil),
      capabilities.toList.sorted,
      layouts.toList.filter(_._2 > 0).sortBy(_._1.canonicalName).map((tpe, size) => p.Program.TypeSize(tpe, size))
    )
    val callerNames = callers.map(_.name).toSet
    ProgramLinker
      .importProgram(request, signatures)
      .map(program => program.functions.filterNot(fn => callerNames(fn.name)) -> program.defs.toSet)
  }

  def importAll(
      imports: List[Import],
      callers: List[p.Function],
      capabilities: Set[String] = configuredCapabilities,
      typeSizes: Map[p.Type, Int] = Map.empty,
      target: Option[String] = LaunchConfig.targetNames.headOption
  ): Either[List[String], (List[p.Function], Set[p.StructDef])] =
    if (imports.isEmpty) Right(Nil -> Set.empty)
    else {
      val available = target.fold(capabilities)(LaunchConfig.capabilitiesFor(capabilities, _))
      imports.map(_.packageName).distinct.sorted.traverse(loadPackage(_, capabilities = capabilities)).flatMap {
        packages =>
          importPackages(
            packages,
            imports,
            callers,
            available,
            typeSizes
          )
      }
    }
}
