package polyregion.ast

import polyregion.ast.PolyAST.Enums
import polyregion.ast.PolyAST.Enums.Variant

object EnumCodeGen {

  private val Header = "// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT."

  private def byEnum: List[List[Variant]] = {
    val grouped = Enums.All.groupBy(_.cppName)
    Enums.All.map(_.cppName).distinct.map(grouped)
  }

  private def cppEnum(variants: List[Variant]): String = {
    val head      = variants.head
    val exportTag = if (head.cppExport) "POLYREGION_EXPORT " else ""
    val body      = variants.map(v => s"  ${v.name} = ${v.value},").mkString("\n")
    s"""enum class ${exportTag}${head.cppName} : uint8_t {
       |$body
       |};""".stripMargin
  }

  def cppHeader: String = {
    val namespaces = Enums.All.map(_.namespace).distinct
    val blocks = namespaces.map { ns =>
      val enums = byEnum.filter(_.head.namespace == ns).map(cppEnum).mkString("\n\n")
      s"""namespace polyregion::$ns {
         |
         |$enums
         |
         |} // namespace polyregion::$ns""".stripMargin
    }
    s"""$Header
       |#pragma once
       |
       |#include <cstdint>
       |
       |#include "polyregion/export.h"
       |
       |${blocks.mkString("\n\n")}
       |""".stripMargin
  }

  private def javaEnum(variants: List[Variant]): (String, String) = {
    val m       = variants.head.javaMirror.get
    val members = variants.filter(_.java.isDefined)
    val sized   = members.exists(_.javaSize.isDefined)
    val (ctorArgs, fields, ctorAssign, sizeAccessor) =
      if (sized)
        (
          "byte value, int sizeInBytes",
          "  final byte value;\n  final int sizeInBytes;",
          "    this.value = value;\n    this.sizeInBytes = sizeInBytes;",
          "\n\n  public int sizeInBytes() {\n    return sizeInBytes;\n  }"
        )
      else
        ("byte value", "  final byte value;", "    this.value = value;", "")
    val body = members
      .map { v =>
        val args = if (sized) s"(byte) ${v.value}, ${v.javaSize.get}" else s"(byte) ${v.value}"
        s"  ${v.java.get}($args)"
      }
      .mkString(",\n") + ";"
    val source =
      s"""$Header
         |package polyregion.jvm.${m.pkg};
         |
         |import polyregion.jvm.ByteEnum;
         |
         |@SuppressWarnings("unused")
         |public enum ${m.name} implements ByteEnum {
         |$body
         |
         |  public static final ${m.name}[] VALUES = values();
         |
         |$fields
         |
         |  ${m.name}($ctorArgs) {
         |$ctorAssign
         |  }
         |
         |  @Override
         |  public byte value() {
         |    return value;
         |  }$sizeAccessor
         |}
         |""".stripMargin
    (s"${m.pkg}/${m.name}.java", source)
  }

  def javaMirrors: List[(String, String)] = byEnum.filter(_.head.javaMirror.isDefined).map(javaEnum)
}
