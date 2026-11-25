package polyregion.ast

import polyregion.ast.PolyAST as p

final case class FortranConfig(moduleName: String)
final case class RegionConfig(contextType: String)
final case class CppRegionConfig(namespaceName: String, includes: List[String])
final case class ScalaConfig(packageName: String, objectName: String, region: Option[RegionConfig] = None)
final case class JavaConfig(packageName: String, className: String, region: Option[RegionConfig] = None)

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

  val Ranks            = List(2, 3)
  val ContextParameter = "context"
  val LinearIndex      = "i"

  private def reject(decls: List[p.FunctionDecl], reserved: List[String]): Unit =
    decls.foreach { decl =>
      decl.args.map(_.named.symbol).find(reserved.contains).foreach { name =>
        fail(
          s"declaration `${decl.name.fqn.mkString(".")}` has a parameter named `$name`, which the projection reserves"
        )
      }
    }

  def validateContextName(decls: List[p.FunctionDecl]): Unit = reject(decls, List(ContextParameter))

  def validateRegionNames(decls: List[p.FunctionDecl]): Unit =
    reject(decls, ContextParameter :: LinearIndex :: extents(Ranks.max))

  def isIndexCallable(tpe: p.Type): Boolean = tpe match {
    case p.Type.Exec(Nil, List(p.Type.IntS64), _) => true
    case _                                        => false
  }

  def isIndexed(decl: p.FunctionDecl): Boolean =
    decl.args.headOption.exists(_.named.tpe == p.Type.IntS64) && decl.args.exists(arg => isIndexCallable(arg.named.tpe))

  def isTask(decl: p.FunctionDecl): Boolean = decl.rtn == p.Type.Unit0 && (decl.args.map(_.named.tpe) match {
    case List(p.Type.IntS64, callable) => isIndexCallable(callable)
    case _                             => false
  })

  def extents(rank: Int): List[String] = RegionIndices.extents(rank)

  def linearisedIndices(rank: Int, linear: String): List[String] =
    RegionIndices.axes(rank).map { axis =>
      val divisor = axis.quotient.map(index => s"n$index")
      val quot = divisor match {
        case Nil        => linear
        case List(only) => s"$linear / $only"
        case several    => s"$linear / (${several.mkString(" * ")})"
      }
      axis.modulus.fold(quot)(index => s"$quot % n$index")
    }

  val FallbackMarker = "polyregion_interface_fallback"
  val ContextMarker  = "polyregion_interface_context"

  def interfaceIdentity(interfaceDef: p.Interface, decl: p.FunctionDecl): String =
    List(interfaceDef.name.fqn.mkString("."), decl.name.fqn.mkString(".")).mkString(":")

  def marker(interfaceDef: p.Interface, decl: p.FunctionDecl): String =
    s"polyregion_interface:${interfaceIdentity(interfaceDef, decl)}"
}
