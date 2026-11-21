package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object ScalaInterfaceCodeGen {
  import InterfaceCodeGenSupport.*

  private val ProjectedBuiltinTypes = Set("Array", "Boolean", "Byte", "Double", "Float", "Int", "Long", "Short", "Unit")

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

  private[ast] def tpe(value: p.Type): String = value match {
    case p.Type.Float32             => "Float"
    case p.Type.Float64             => "Double"
    case p.Type.IntS8               => "Byte"
    case p.Type.IntS16              => "Short"
    case p.Type.IntS32              => "Int"
    case p.Type.IntS64              => "Long"
    case p.Type.Bool1               => "Boolean"
    case p.Type.Unit0               => "Unit"
    case p.Type.Var(name, _)        => name
    case p.Type.Struct(name, Nil)   => s"_root_.${name.fqn.map(identifier(_, "type")).mkString(".")}"
    case p.Type.Ptr(comp, _)        => s"Array[${tpe(comp)}]"
    case p.Type.Arr(comp, _, _)     => s"Array[${tpe(comp)}]"
    case p.Type.Exec(Nil, Nil, rtn) => s"() => ${tpe(rtn)}"
    case p.Type.Exec(Nil, List(arg), rtn) =>
      val parameter = arg match {
        case p.Type.Exec(_, _, _) => s"(${tpe(arg)})"
        case _                    => tpe(arg)
      }
      s"$parameter => ${tpe(rtn)}"
    case p.Type.Exec(Nil, args, rtn) => s"(${args.map(tpe).mkString(", ")}) => ${tpe(rtn)}"
    case p.Type.Exec(_, _, _)        => fail("generic callable types are not supported by the Scala projection yet")
    case other                       => fail(s"type has no exact Scala interface projection: $other")
  }

  private def render(prefix: String, parameters: List[String], suffix: String): String = {
    val inline = s"$prefix${parameters.mkString(", ")}$suffix"
    if (inline.length <= 120) inline
    else s"$prefix\n${parameters.map("      " + _).mkString(",\n")}\n  $suffix"
  }

  private val CurrentContext = "polyregion.jvm.runtime.Context.current()"

  private def declaration(interfaceDef: p.Interface, decl: p.FunctionDecl, context: Option[String]): String = {
    val tpeVars = if (decl.tpeVars.nonEmpty) decl.tpeVars.map(_.name).mkString("[", ", ", "]") else ""
    val parameters = context.map(tpe => s"$ContextParameter: $tpe").toList ++ decl.args.map { arg =>
      val modifier = arg.named.tpe match {
        case p.Type.Exec(_, _, _) => "inline "
        case _                    => ""
      }
      s"$modifier${arg.named.symbol}: ${tpe(arg.named.tpe)}"
    }
    val signature = render(s"  inline def ${decl.name.last}$tpeVars(", parameters, s"): ${tpe(decl.rtn)} =")
    val arguments = (if (context.isDefined) ContextParameter else CurrentContext) :: decl.args.map(_.named.symbol)
    s"""  @compileTimeOnly("${marker(interfaceDef, decl)}")
       |$signature
       |${launchCall(interfaceDef, decl, arguments)}""".stripMargin
  }

  private def launchCall(interfaceDef: p.Interface, decl: p.FunctionDecl, arguments: List[String]): String = {
    val prefix =
      s"    polyregion.scalalang.launch.call[${tpe(decl.rtn)}](\"${interfaceIdentity(interfaceDef, decl)}\", "
    val inline = s"$prefix${arguments.mkString(", ")})"
    if (inline.length <= 120) inline
    else
      s"    polyregion.scalalang.launch.call[${tpe(decl.rtn)}](\n      \"${interfaceIdentity(interfaceDef, decl)}\",\n${arguments
          .map("      " + _)
          .mkString(",\n")}\n    )"
  }

  private def sugar(decl: p.FunctionDecl, context: Option[String], rank: Int): String = {
    val tpeVars  = if (decl.tpeVars.nonEmpty) decl.tpeVars.map(_.name).mkString("[", ", ", "]") else ""
    val dims     = extents(rank)
    val total    = dims.tail.foldLeft(dims.head)((acc, dim) => s"Math.multiplyExact($acc, $dim)")
    val ctxParam = context.map(tpe => s"$ContextParameter: $tpe").toList
    val ctxArg   = Option.when(context.isDefined)(ContextParameter).toList
    val rest     = decl.args.tail
    val parameters = ctxParam ++ dims.map(d => s"$d: Long") ++ rest.map { arg =>
      arg.named.tpe match {
        case p.Type.Exec(Nil, List(p.Type.IntS64), rtn) =>
          s"inline ${arg.named.symbol}: (${List.fill(rank)("Long").mkString(", ")}) => ${tpe(rtn)}"
        case other @ p.Type.Exec(_, _, _) => s"inline ${arg.named.symbol}: ${tpe(other)}"
        case other                        => s"${arg.named.symbol}: ${tpe(other)}"
      }
    }
    val arguments = ctxArg ++ (total :: rest.map { arg =>
      arg.named.tpe match {
        case p.Type.Exec(Nil, List(p.Type.IntS64), _) =>
          s"($LinearIndex: Long) => ${arg.named.symbol}(${linearisedIndices(rank, LinearIndex).mkString(", ")})"
        case _ => arg.named.symbol
      }
    })
    val signature = render(s"  inline def ${decl.name.last}$tpeVars(", parameters, s"): ${tpe(decl.rtn)} =")
    s"""$signature
       |${bodyCall(decl.name.last, arguments)}""".stripMargin
  }

  private def bodyCall(name: String, arguments: List[String]): String = {
    val inline = s"    $name(${arguments.mkString(", ")})"
    if (inline.length <= 120) inline else s"    $name(\n${arguments.map("      " + _).mkString(",\n")}\n    )"
  }

  private def task(decl: p.FunctionDecl, context: Option[String]): String = {
    val parameters = context.map(tpe => s"$ContextParameter: $tpe").toList :+ "inline body: () => Unit"
    val arguments  = Option.when(context.isDefined)(ContextParameter).toList ++ List("1L", s"(_: Long) => body()")
    s"""${render("  inline def task(", parameters, "): Unit =")}
       |    ${decl.name.last}(${arguments.mkString(", ")})""".stripMargin
  }

  def apply(interfaceDef: p.Interface, config: ScalaConfig): String = {
    if (config.packageName.isEmpty) fail("Scala package name is empty")
    config.packageName.split('.').foreach(identifier(_, "package"))
    identifier(config.objectName, "object")
    val decls = declarations(interfaceDef)
    validatePortableOverloads(decls, "Scala")
    if (config.region.nonEmpty) validateRegionNames(decls)
    decls.foreach { decl =>
      identifier(decl.name.last, "method")
      decl.tpeVars.foreach { variable =>
        val name = variable.name
        identifier(name, "type variable")
        if (ProjectedBuiltinTypes(name))
          fail(s"Scala type variable `$name` shadows a projected built-in type")
      }
      decl.args.foreach(arg => identifier(arg.named.symbol, "parameter"))
    }
    val contexts = None :: config.region.map(region => Some(region.contextType)).toList
    val stubs    = decls.flatMap(decl => contexts.map(declaration(interfaceDef, decl, _)))
    val sugars = config.region.toList.flatMap { _ =>
      decls.filter(isIndexed).flatMap(decl => contexts.flatMap(context => Ranks.map(sugar(decl, context, _))))
        ++ decls.filter(isTask).flatMap(decl => contexts.map(task(decl, _)))
    }
    val body = (stubs ++ sugars).mkString("\n\n")
    s"""package ${config.packageName}
       |
       |import scala.annotation.compileTimeOnly
       |
       |object ${config.objectName} {
       |
       |$body
       |}
       |""".stripMargin
  }

  private val SurfaceProfiles = List(
    "Int"    -> "0",
    "Long"   -> "0L",
    "Float"  -> "0.0f",
    "Double" -> "0.0"
  )

  private def surfaceZero(tpe: p.Type, profile: Map[String, (String, String)]): String = tpe match {
    case p.Type.Var(name, _) => profile(name)._2
    case p.Type.Unit0        => "()"
    case p.Type.Bool1        => "false"
    case p.Type.IntS32       => "0"
    case p.Type.IntS64       => "0L"
    case p.Type.Float32      => "0.0f"
    case p.Type.Float64      => "0.0"
    case other               => fail(s"no Scala surface value for $other")
  }

  private def surfaceTypeName(tpe: p.Type, profile: Map[String, (String, String)]): String = tpe match {
    case p.Type.Var(name, _) => profile(name)._1
    case other               => ScalaInterfaceCodeGen.tpe(other)
  }

  private def surfaceCall(prefix: String, args: List[String]): String = {
    val inline = s"    $prefix(${args.mkString(", ")})"
    if (inline.length <= 120) inline else s"    $prefix(\n${args.map("      " + _).mkString(",\n")}\n    )"
  }

  private def surfaceArgument(tpe: p.Type, profile: Map[String, (String, String)]): String = tpe match {
    case p.Type.Exec(_, args, rtn) =>
      val parameters = args.zipWithIndex.map((arg, i) => s"arg$i: ${surfaceTypeName(arg, profile)}")
      s"(${parameters.mkString(", ")}) => ${surfaceZero(rtn, profile)}"
    case other => surfaceZero(other, profile)
  }

  def surface(interfaceDef: p.Interface, config: ScalaConfig, objectName: String): String = {
    val decls = declarations(interfaceDef)
    identifier(objectName, "object")
    val contexts = None :: config.region.map(region => Some(region.contextType)).toList
    val calls = for {
      decl <- decls
      profile <- decl.tpeVars match {
        case Nil => List(Map.empty[String, (String, String)])
        case vars =>
          vars
            .foldRight(List(List.empty[(String, (String, String))])) { (variable, tails) =>
              for {
                prim <- SurfaceProfiles
                tail <- tails
              } yield (variable.name -> prim) :: tail
            }
            .map(_.toMap)
      }
      context <- contexts
    } yield {
      val shown   = decl.tpeVars.map(variable => s"${variable.name}=${profile(variable.name)._1}")
      val label   = (shown ++ context.map(_ => ContextParameter)).mkString(": ")
      val comment = s"    // ${decl.name.last}${if (label.isEmpty) "" else s": $label"}"
      val args = Option.when(context.isDefined)(ContextParameter).toList ++ decl.args.map(arg =>
        surfaceArgument(arg.named.tpe, profile)
      )
      s"$comment\n${surfaceCall(s"${config.objectName}.${decl.name.last}", args)}"
    }
    val sugar = config.region.toList.flatMap { _ =>
      val indexedCalls = for {
        decl    <- decls.filter(isIndexed)
        rank    <- Ranks
        context <- contexts
      } yield {
        val profile = decl.tpeVars.map(variable => variable.name -> ("Int", "0")).toMap
        val args =
          Option.when(context.isDefined)(ContextParameter).toList ++ List.fill(rank)("0L") ++ decl.args.tail.map { arg =>
            arg.named.tpe match {
              case callable if isIndexCallable(callable) =>
                val rtn = callable match { case p.Type.Exec(_, _, rtn) => rtn; case other => other }
                s"(${List.tabulate(rank)(i => s"arg$i: Long").mkString(", ")}) => ${surfaceZero(rtn, profile)}"
              case other => surfaceArgument(other, profile)
            }
          }
        s"    // ${decl.name.last}: rank $rank\n${surfaceCall(s"${config.objectName}.${decl.name.last}", args)}"
      }
      val taskCalls = for {
        _       <- decls.filter(isTask)
        context <- contexts
      } yield {
        val args = Option.when(context.isDefined)(ContextParameter).toList :+ "() => ()"
        s"    // task\n    ${config.objectName}.task(${args.mkString(", ")})"
      }
      indexedCalls ++ taskCalls
    }
    val parameter = config.region.fold("")(region => s"$ContextParameter: ${region.contextType}")
    s"""package ${config.packageName}
       |
       |object $objectName {
       |  def compile($parameter): Unit = InterfaceSurfaceCheck.erase {
       |${(calls ++ sugar).mkString("\n\n")}
       |  }
       |}
       |""".stripMargin
  }
}
