package polyregion.spectra

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

import polyregion.ast.InterfaceCodeGen

import scala.jdk.CollectionConverters.*
import scala.util.Using

object SpectraCodeGen {

  private val fortran = InterfaceCodeGen.FortranConfig("spectra_api")
  private val scala   = InterfaceCodeGen.ScalaConfig("polyregion", "spectra")

  val cppHeader: String     = InterfaceCodeGen.cppHeader(Spectra.interfaceDef)
  val fortranModule: String = InterfaceCodeGen.fortranModule(Spectra.interfaceDef, fortran)
  val scalaObject: String   = InterfaceCodeGen.scalaObject(Spectra.interfaceDef, scala)

  private def outputs(root: Path): List[(Path, String)] = List(
    root.resolve("generated/cpp/include/polyregion/spectra_api.hpp") -> cppHeader,
    root.resolve("generated/fortran/spectra_api.f90")                -> fortranModule,
    root.resolve("generated/scala/polyregion/spectra.scala")         -> scalaObject
  )

  def checkGenerated(root: Path): List[String] = {
    val normalized = root.toAbsolutePath.normalize
    val expected   = outputs(normalized)
    val current = expected.flatMap { case (path, source) =>
      InterfaceCodeGen.checkCurrent(path, source).left.toOption
    }
    val generated     = normalized.resolve("generated")
    val expectedPaths = expected.map(_._1).toSet
    val unexpected =
      if (!Files.exists(generated)) Nil
      else
        Using
          .resource(Files.walk(generated))(
            _.iterator.asScala.filter(Files.isRegularFile(_)).filterNot(expectedPaths).toList
          )
          .map(path => s"unexpected generated output: $path")
    current ++ unexpected
  }

  def main(args: Array[String]): Unit = outputs(Paths.get(args.head).toAbsolutePath.normalize).foreach {
    case (path, source) =>
      Files.createDirectories(path.getParent)
      Files.writeString(path, source, StandardCharsets.UTF_8)
  }
}
