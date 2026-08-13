package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object ScalaLibraryCodeGen {
  import LibraryCodeGen.Support.*

  private val Keywords = Set(
    "abstract",
    "case",
    "catch",
    "class",
    "def",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "final",
    "finally",
    "for",
    "forSome",
    "given",
    "if",
    "implicit",
    "import",
    "lazy",
    "match",
    "new",
    "null",
    "object",
    "opaque",
    "open",
    "override",
    "package",
    "private",
    "protected",
    "return",
    "sealed",
    "super",
    "then",
    "this",
    "throw",
    "trait",
    "transparent",
    "true",
    "try",
    "type",
    "val",
    "var",
    "while",
    "with",
    "yield",
    "_"
  )

  private def identifier(value: String, what: String): String =
    targetIdentifier(value, s"Scala $what", Keywords)

  private def tpe(value: p.Type): String = value match {
    case p.Type.Float32             => "Float"
    case p.Type.Float64             => "Double"
    case p.Type.IntS8               => "Byte"
    case p.Type.IntS16              => "Short"
    case p.Type.IntS32              => "Int"
    case p.Type.IntS64              => "Long"
    case p.Type.Bool1               => "Boolean"
    case p.Type.Unit0               => "Unit"
    case p.Type.Var(name)           => name
    case p.Type.Struct(name, Nil)   => s"_root_.${name.fqn.map(identifier(_, "type")).mkString(".")}"
    case p.Type.Ptr(comp, _)        => s"Array[${tpe(comp)}]"
    case p.Type.Arr(comp, _, _)     => s"Array[${tpe(comp)}]"
    case p.Type.Exec(Nil, Nil, rtn) => s"() => ${tpe(rtn)}"
    case p.Type.Exec(Nil, List(arg), rtn) =>
      val parameter = arg match {
        case _: p.Type.Exec => s"(${tpe(arg)})"
        case _              => tpe(arg)
      }
      s"$parameter => ${tpe(rtn)}"
    case p.Type.Exec(Nil, args, rtn) => s"(${args.map(tpe).mkString(", ")}) => ${tpe(rtn)}"
    case p.Type.Exec(_, _, _)        => fail("generic callable types are not supported by the Scala projection yet")
    case other                       => fail(s"type has no exact Scala library projection: $other")
  }

  private def declaration(
      library: p.LibraryDef,
      decl: p.FunctionDecl,
      config: LibraryCodeGen.ScalaConfig
  ): String = {
    val tpeVars    = if (decl.tpeVars.nonEmpty) decl.tpeVars.mkString("[", ", ", "]") else ""
    val parameters = decl.args.map(arg => s"${arg.named.symbol}: ${tpe(arg.named.tpe)}")
    val prefix     = s"  def ${decl.name.last}$tpeVars("
    val suffix     = s"): ${tpe(decl.rtn)} ="
    val signature = {
      val inline = s"$prefix${parameters.mkString(", ")}$suffix"
      if (inline.length <= 120) inline
      else s"$prefix\n${parameters.map("      " + _).mkString(",\n")}\n  $suffix"
    }
    s"""  @${config.traitName}.PolyregionImport("${library.name.fqn.mkString(".")}", "${decl.name.fqn.mkString(".")}")
       |$signature
       |    throw ${config.traitName}.PolyregionImportFailure("compiler did not replace ${decl.name.fqn.mkString(
        "."
      )}")""".stripMargin
  }

  def apply(library: p.LibraryDef, config: LibraryCodeGen.ScalaConfig): String = {
    if (config.packageName.isEmpty) fail("Scala package name is empty")
    config.packageName.split('.').foreach(identifier(_, "package"))
    identifier(config.traitName, "trait")
    val decls = declarations(library)
    validatePortableOverloads(decls, "Scala")
    decls.foreach { decl =>
      identifier(decl.name.last, "method")
      decl.tpeVars.foreach { name =>
        identifier(name, "type variable")
        if (Set("Array", "Boolean", "Byte", "Double", "Float", "Int", "Long", "Short", "Unit")(name))
          fail(s"Scala type variable `$name` shadows a projected built-in type")
      }
      decl.args.foreach(arg => identifier(arg.named.symbol, "parameter"))
    }
    val body = decls.map(declaration(library, _, config)).mkString("\n\n")
    s"""package ${config.packageName}
       |
       |import scala.annotation.StaticAnnotation
       |
       |object ${config.traitName} {
       |  final class PolyregionImport(val library: String, val declaration: String) extends StaticAnnotation
       |
       |  final class PolyregionImportFailure(message: String) extends RuntimeException(message)
       |}
       |
       |trait ${config.traitName} {
       |
       |$body
       |}
       |""".stripMargin
  }
}
