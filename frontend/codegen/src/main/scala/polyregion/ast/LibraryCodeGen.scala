package polyregion.ast

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import polyregion.ast.PolyAST as p

object LibraryCodeGen {

  case class FortranVariant(suffix: String, tpeVars: Map[String, String], imports: List[String])
  case class FortranConfig(moduleName: String, variants: List[FortranVariant])
  case class ScalaConfig(packageName: String, traitName: String)

  def cppHeader(library: p.LibraryDef): String =
    CppLibraryCodeGen(library)

  def fortranModule(library: p.LibraryDef, config: FortranConfig): String =
    FortranLibraryCodeGen(library, config)

  def scalaTrait(library: p.LibraryDef, config: ScalaConfig): String =
    ScalaLibraryCodeGen(library, config)

  def checkCurrent(path: Path, expected: String): Either[String, Unit] =
    if (!Files.exists(path)) Left(s"generated output is missing: $path")
    else {
      val actual = Files.readString(path, StandardCharsets.UTF_8)
      Either.cond(actual == expected, (), s"generated output is stale: $path")
    }

  private[ast] object Support {

    private val Identifier = "[A-Za-z_][A-Za-z0-9_]*".r

    def fail(message: String): Nothing = throw IllegalArgumentException(message)

    def identifier(value: String, what: String): String =
      if (Identifier.matches(value)) value else fail(s"invalid $what identifier `$value`")

    def targetIdentifier(value: String, what: String, keywords: Set[String]): String = {
      identifier(value, what)
      if (keywords(value)) fail(s"$what identifier `$value` is reserved") else value
    }

    def validatePortableOverloads(decls: List[p.FunctionDecl], target: String): Unit =
      decls.groupBy(_.name).values.foreach { overloads =>
        overloads.groupBy(_.args.size).collectFirst { case (arity, clashes) if clashes.size > 1 => arity }.foreach {
          arity =>
            fail(
              s"$target cannot portably distinguish overloads `${overloads.head.name.fqn.mkString(".")}` with $arity arguments"
            )
        }
      }

    def declarations(library: p.LibraryDef): List[p.FunctionDecl] = {
      if (library.name.fqn.isEmpty) fail("library name is empty")
      library.name.fqn.foreach(identifier(_, "library"))
      if (library.decls.isEmpty) fail(s"library `${library.name.fqn.mkString(".")}` has no declarations")
      val sorted = library.decls.sortBy(decl => (decl.name.fqn.mkString("."), decl.toString))
      sorted.foreach { decl =>
        decl.name.fqn.foreach(identifier(_, "declaration"))
        if (decl.name.fqn.dropRight(1) != library.name.fqn)
          fail(
            s"declaration `${decl.name.fqn.mkString(".")}` is not in library `${library.name.fqn.mkString(".")}`"
          )
        decl.tpeVars.foreach(identifier(_, "type variable"))
        decl.args.foreach(arg => identifier(arg.named.symbol, "argument"))
        decl.classifyArguments.left.foreach(errors =>
          fail(s"invalid declaration `${decl.name.fqn.mkString(".")}`: ${errors.mkString("; ")}")
        )
        if (decl.receiver.nonEmpty || decl.moduleCaptures.nonEmpty || decl.termCaptures.nonEmpty)
          fail(s"declaration `${decl.name.fqn.mkString(".")}` has an unsupported receiver or explicit captures")
      }
      sorted
    }

    def marker(library: p.LibraryDef, decl: p.FunctionDecl, suffix: Option[String] = None): String =
      (List("polyregion_import", library.name.fqn.mkString("."), decl.name.fqn.mkString(".")) ::: suffix.toList)
        .mkString(":")
  }
}
