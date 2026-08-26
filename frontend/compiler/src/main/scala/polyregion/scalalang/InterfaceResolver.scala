package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{MsgPack, PolyAST as p, ProgramLinker, calleeName, canonicalName, given}
import polyregion.scalalang.generated.PolyASTWireSchema

import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.control.NonFatal

private[scalalang] object InterfaceResolver {

  final case class Import(packageName: String, declaration: String, target: p.Expr.Invoke)

  private def configured(name: String, property: String): Option[String] =
    Option(System.getenv(name)).orElse(Option(System.getProperty(property))).map(_.trim).filter(_.nonEmpty)

  def configuredCapabilities: Set[String] =
    configured("POLYFRONT_LIBRARY_CAPABILITIES", "polyregion.library.capabilities")
      .fold(Set.empty[String])(_.split(',').iterator.map(_.trim).filter(_.nonEmpty).toSet)

  private def safePathComponent(value: String): Boolean = {
    val invalid = "\\/<>:\"|?*"
    val base    = value.takeWhile(_ != '.').toUpperCase(Locale.ROOT)
    value.nonEmpty && value != "." && value != ".." && !value.endsWith(".") && !value.endsWith(" ") &&
    value.forall(ch => ch >= ' ' && ch != 127 && !invalid.contains(ch)) &&
    !Set("CON", "PRN", "AUX", "NUL")(base) &&
    !(base.length == 4 && (base.startsWith("COM") || base.startsWith("LPT")) && base.last >= '1' && base.last <= '9')
  }

  private def decode[A: MsgPack.Codec](path: Path, expectedHash: String): Either[List[String], A] =
    try
      MsgPack
        .decode[MsgPack.Versioned[A]](Files.readAllBytes(path))
        .left
        .map(error => List(s"cannot decode package file `$path`: ${error.getMessage}"))
        .flatMap(versioned =>
          Either.cond(
            versioned.hash == expectedHash,
            versioned.t,
            List(s"package file `$path` has schema ${versioned.hash}, expected $expectedHash")
          )
        )
    catch {
      case NonFatal(error) => Left(List(s"cannot read package file `$path`: ${error.getMessage}"))
    }

  def loadPackage(packageName: String): Either[List[String], p.Package] = {
    if (!safePathComponent(packageName)) return Left(List(s"invalid package identity `$packageName`"))
    val roots = configured("POLYFRONT_LIBRARY_PATH", "polyregion.library.path").toList
      .flatMap(_.split(java.io.File.pathSeparator).toList)
      .filter(_.nonEmpty)
    val matches = roots
      .map(Paths.get(_).resolve(packageName).resolve("lib.polyast"))
      .filter(Files.isRegularFile(_))
    matches match {
      case Nil         => Left(List(s"no library package is available for interface `$packageName`"))
      case _ :: _ :: _ => Left(List(s"interface `$packageName` is ambiguous across ${matches.size} package roots"))
      case path :: Nil =>
        for {
          pkg <- decode[p.Package](path, PolyASTWireSchema.PackageHash)
          _ <- Either.cond(
            pkg.interface.name.fqn.mkString(".") == packageName,
            (),
            List(
              s"package identity differs: expected `$packageName`, got `${pkg.interface.name.fqn.mkString(".")}`"
            )
          )
        } yield pkg
    }
  }

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
        target.receiver.map(value => p.Arg(p.Named("this", value.tpe), None)),
        target.args.zipWithIndex.map((value, index) => p.Arg(p.Named(s"arg$index", value.tpe), None)),
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
      callerDecls: List[p.FunctionDecl],
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
    val callers = callerDecls.map(decl =>
      p.Function(decl, Nil, p.Function.Visibility.Internal, p.Function.FpMode.Relaxed, p.CallConvention.RegularCall)
    )
    val request = p.Program.LinkRequest(
      packages,
      p.Program(None, roots ::: callers, Nil),
      capabilities.toList.sorted,
      layouts.toList.filter(_._2 > 0).sortBy(_._1.canonicalName).map((tpe, size) => p.Program.TypeSize(tpe, size))
    )
    val callerNames = callerDecls.map(_.name).toSet
    ProgramLinker
      .importProgram(request, signatures)
      .map(program => program.functions.filterNot(fn => callerNames(fn.name)) -> program.defs.toSet)
  }

  def importAll(
      imports: List[Import],
      callerDecls: List[p.FunctionDecl],
      capabilities: Set[String] = configuredCapabilities,
      typeSizes: Map[p.Type, Int] = Map.empty
  ): Either[List[String], (List[p.Function], Set[p.StructDef])] =
    if (imports.isEmpty) Right(Nil -> Set.empty)
    else
      imports.map(_.packageName).distinct.sorted.traverse(loadPackage).flatMap { packages =>
        importPackages(
          packages,
          imports,
          callerDecls,
          capabilities,
          typeSizes
        )
      }
}
