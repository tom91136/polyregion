package polyregion.spectra

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.{InterfaceCodeGen, MsgPack}

import scala.jdk.CollectionConverters.*
import scala.util.Using

object SpectraCodeGen {

  private val fortran = InterfaceCodeGen.FortranConfig("spectra_api")
  private val scala   = InterfaceCodeGen.ScalaConfig("polyregion", "spectra")

  private[spectra] val cppHeader: String = InterfaceCodeGen.cppHeader(Spectra.interfaceDef)
  private val fortranModule: String      = InterfaceCodeGen.fortranModule(Spectra.interfaceDef, fortran)
  private val scalaObject: String        = InterfaceCodeGen.scalaObject(Spectra.interfaceDef, scala)

  private def textOutputs(root: Path): List[(Path, String)] =
    List(
      root.resolve("generated/cpp/include/polyregion/spectra_api.hpp") -> cppHeader,
      root.resolve("generated/fortran/spectra_api.f90")                -> fortranModule,
      root.resolve("generated/scala/polyregion/spectra.scala")         -> scalaObject
    ) ++ SpectraTestCodeGen.textOutputs(root)

  private def interfaceBytes: Array[Byte] =
    MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.PackageHash, Spectra.interfaceDef))

  private def binaryOutputs(root: Path): List[(Path, Array[Byte])] =
    List(root.resolve("generated/package/spectra.interface.polyast") -> interfaceBytes)

  def checkGenerated(root: Path): List[String] = {
    val normalized     = root.toAbsolutePath.normalize
    val textExpected   = textOutputs(normalized)
    val binaryExpected = binaryOutputs(normalized)
    val current = textExpected.flatMap { case (path, source) =>
      InterfaceCodeGen.checkCurrent(path, source).left.toOption
    }
    val generated     = normalized.resolve("generated")
    val expectedPaths = (textExpected.map(_._1) ++ binaryExpected.map(_._1)).toSet
    val unexpectedGenerated =
      if (!Files.exists(generated)) Nil
      else
        Using
          .resource(Files.walk(generated))(
            _.iterator.asScala.filter(Files.isRegularFile(_)).filterNot(expectedPaths).toList
          )
          .map(path => s"unexpected generated output: $path")
    val testRoot = normalized.resolve("test/generated")
    val unexpectedTests =
      if (!Files.exists(testRoot)) Nil
      else
        Using
          .resource(Files.list(testRoot))(
            _.iterator.asScala.filter(Files.isRegularFile(_)).filterNot(expectedPaths).toList
          )
          .map(path => s"unexpected generated output: $path")
    current ++ unexpectedGenerated ++ unexpectedTests
  }

  def main(args: Array[String]): Unit = {
    val (root, interfaceOnly) = args.toList match {
      case root :: Nil                  => (root, false)
      case root :: "--interface" :: Nil => (root, true)
      case _                            => sys.error("usage: SpectraCodeGen <root> [--interface]")
    }
    val normalized = Paths.get(root).toAbsolutePath.normalize
    if (!interfaceOnly) textOutputs(normalized).foreach { case (path, source) =>
      Files.createDirectories(path.getParent)
      Files.writeString(path, source, StandardCharsets.UTF_8)
    }
    binaryOutputs(normalized).foreach { case (path, bytes) =>
      Files.createDirectories(path.getParent)
      Files.write(path, bytes)
    }
  }
}
