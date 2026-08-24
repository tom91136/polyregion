package polyregion.scalalang

import polyregion.ast.{MsgPack, PackageSymResolver, PolyAST as p, given}
import polyregion.scalalang.generated.PolyASTWireSchema

import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.control.NonFatal

private[scalalang] object InterfaceResolver {

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

  def resolve(
      pkg: p.Package,
      declaration: String,
      target: p.Expr.Invoke,
      callerDecls: List[p.FunctionDecl] = Nil,
      capabilities: Set[String] = configuredCapabilities,
      typeSizes: Map[p.Type, Int] = Map.empty
  ): Either[List[String], (List[p.Function], Set[p.StructDef])] = {
    val layouts = (target.args.map(_.tpe) :+ target.rtn).flatMap(layoutsOf).toMap ++ typeSizes
    PackageSymResolver
      .resolveImplementation(pkg, declaration, target, callerDecls, capabilities, layouts)
      .map(resolved => resolved.functions -> resolved.definitions)
  }
}
