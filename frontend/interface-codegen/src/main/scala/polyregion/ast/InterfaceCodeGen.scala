package polyregion.ast

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import polyregion.ast.PolyAST as p

object InterfaceCodeGen {

  def cppHeader(interfaceDef: p.Interface): String =
    CppInterfaceCodeGen(interfaceDef)

  def fortranModule(interfaceDef: p.Interface, config: FortranConfig): String =
    FortranInterfaceCodeGen(interfaceDef, config)

  def scalaObject(interfaceDef: p.Interface, config: ScalaConfig): String =
    ScalaInterfaceCodeGen(interfaceDef, config)

  def checkCurrent(path: Path, expected: String): Either[String, Unit] =
    if (!Files.exists(path)) Left(s"generated output is missing: $path")
    else {
      val actual = Files.readString(path, StandardCharsets.UTF_8)
      Either.cond(actual == expected, (), s"generated output is stale: $path")
    }
}
