package polyregion.ast

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import polyregion.ast.PolyAST as p

object InterfaceCodeGen {

  def cppHeader(interfaceDef: p.Interface, contextOverloads: Boolean = false): String =
    CppInterfaceCodeGen(interfaceDef, contextOverloads)

  def cppRegionHeader(interfaceDef: p.Interface, config: CppRegionConfig): String =
    CppRegionCodeGen(interfaceDef, config)

  def fortranModule(interfaceDef: p.Interface, config: FortranConfig): String =
    FortranInterfaceCodeGen(interfaceDef, config)

  def scalaObject(interfaceDef: p.Interface, config: ScalaConfig): String =
    ScalaInterfaceCodeGen(interfaceDef, config)

  def scalaSurface(interfaceDef: p.Interface, config: ScalaConfig, objectName: String): String =
    ScalaInterfaceCodeGen.surface(interfaceDef, config, objectName)

  def javaClass(interfaceDef: p.Interface, config: JavaConfig): String =
    JavaInterfaceCodeGen(interfaceDef, config)

  def javaSurface(interfaceDef: p.Interface, config: JavaConfig, className: String): String =
    JavaInterfaceCodeGen.surface(interfaceDef, config, className)

  def javaOverloadCount(decl: p.FunctionDecl): Int =
    JavaInterfaceCodeGen.overloadCount(decl)

  def checkCurrent(path: Path, expected: String): Either[String, Unit] =
    if (!Files.exists(path)) Left(s"generated output is missing: $path")
    else {
      val actual = Files.readString(path, StandardCharsets.UTF_8)
      Either.cond(actual == expected, (), s"generated output is stale: $path")
    }
}
