package polyregion.ast

import polyregion.ast.PolyAST as p

final case class FortranConfig(moduleName: String)
final case class ScalaConfig(packageName: String, objectName: String)

private[ast] object InterfaceCodeGenSupport {

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

  def declarations(interfaceDef: p.Interface): List[p.FunctionDecl] = {
    if (interfaceDef.name.fqn.isEmpty) fail("interface name is empty")
    interfaceDef.name.fqn.foreach(identifier(_, "interface"))
    if (interfaceDef.declarations.isEmpty)
      fail(s"interface `${interfaceDef.name.fqn.mkString(".")}` has no declarations")
    val sorted             = interfaceDef.declarations.sortBy(_.signatureKey)
    lazy val interfaceName = interfaceDef.name.fqn.mkString(".")
    sorted.foreach { decl =>
      decl.name.fqn.foreach(identifier(_, "declaration"))
      if (decl.name.fqn.dropRight(1) != interfaceDef.name.fqn)
        fail(
          s"declaration `${decl.name.fqn.mkString(".")}` is not in interface `$interfaceName`"
        )
      decl.tpeVars.foreach(variable => identifier(variable.name, "type variable"))
      decl.args.foreach(arg => identifier(arg.named.symbol, "argument"))
      val errors = decl.validateInterfaceDeclaration
      if (errors.nonEmpty)
        fail(s"invalid declaration `${decl.name.fqn.mkString(".")}`: ${errors.mkString("; ")}")
      if (decl.receiver.nonEmpty || decl.moduleCaptures.nonEmpty || decl.termCaptures.nonEmpty)
        fail(s"declaration `${decl.name.fqn.mkString(".")}` has an unsupported receiver or explicit captures")
    }
    sorted
  }

  val FallbackMarker = "polyregion_interface_fallback"

  def marker(interfaceDef: p.Interface, decl: p.FunctionDecl): String =
    List("polyregion_interface", interfaceDef.name.fqn.mkString("."), decl.name.fqn.mkString(".")).mkString(":")
}
