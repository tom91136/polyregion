package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object FortranLibraryCodeGen {
  import LibraryCodeGen.Support.*

  private val Identifier = "[A-Za-z][A-Za-z0-9_]*".r
  private val IsoTypes =
    List("c_bool", "c_double", "c_float", "c_int8_t", "c_int16_t", "c_int32_t", "c_int64_t")

  private def identifier(value: String, what: String): String = {
    if (!Identifier.matches(value)) fail(s"invalid Fortran $what identifier `$value`")
    if (value.length > 63) fail(s"Fortran $what identifier exceeds 63 characters: `$value`")
    value
  }

  private def tpe(tpe: p.Type, variant: LibraryCodeGen.FortranVariant): String = tpe match {
    case p.Type.Float32 => "real(c_float)"
    case p.Type.Float64 => "real(c_double)"
    case p.Type.IntS8   => "integer(c_int8_t)"
    case p.Type.IntS16  => "integer(c_int16_t)"
    case p.Type.IntS32  => "integer(c_int32_t)"
    case p.Type.IntS64  => "integer(c_int64_t)"
    case p.Type.Bool1   => "logical(c_bool)"
    case p.Type.Var(name) =>
      variant.tpeVars.getOrElse(name, fail(s"Fortran variant `${variant.suffix}` does not bind type variable `$name`"))
    case other => fail(s"type has no exact Fortran library projection: $other")
  }

  private def intent(arg: p.Arg): String = arg.boundary.map(_.access) match {
    case Some(p.Arg.Access.Read)      => "in"
    case Some(p.Arg.Access.Write)     => "out"
    case Some(p.Arg.Access.ReadWrite) => "inout"
    case None                         => fail(s"pointer `${arg.named.symbol}` has no boundary")
  }

  private def imports(variant: LibraryCodeGen.FortranVariant): String =
    (variant.imports ::: IsoTypes).distinct.sorted.mkString(", ")

  private def value(tpe0: p.Type, name: String, variant: LibraryCodeGen.FortranVariant): String = tpe0 match {
    case p.Type.Ptr(comp, _)           => s"${tpe(comp, variant)}, intent(in) :: $name(*)"
    case p.Type.Arr(comp, length, _)   => s"${tpe(comp, variant)}, intent(in) :: $name($length)"
    case _: p.Type.Exec | p.Type.Unit0 => fail(s"unsupported Fortran value parameter `$name`: $tpe0")
    case other                         => s"${tpe(other, variant)}, intent(in), value :: $name"
  }

  private def callable(
      decl: p.FunctionDecl,
      arg: p.Arg,
      signature: p.Type.Exec,
      variant: LibraryCodeGen.FortranVariant,
      overload: String
  ): String = {
    if (signature.tpeVars.nonEmpty)
      fail(s"generic callable `${arg.named.symbol}` is not supported by the Fortran projection yet")
    val name = s"polyregion_${decl.name.last}${overload}_${arg.named.symbol}_${variant.suffix}"
    val args = signature.args.indices.map(index => s"arg$index").mkString(", ")
    val decls = signature.args.zipWithIndex
      .map((tpe0, index) => s"      ${value(tpe0, s"arg$index", variant)}")
      .mkString("\n")
    val imported = s"      import :: ${imports(variant)}"
    if (signature.rtn == p.Type.Unit0)
      s"""    subroutine $name($args)
         |$imported
         |$decls
         |    end subroutine $name""".stripMargin
    else
      s"""    function $name($args) result(r)
         |$imported
         |$decls
         |      ${tpe(signature.rtn, variant)} :: r
         |    end function $name""".stripMargin
  }

  private def parameter(
      decl: p.FunctionDecl,
      arg: p.Arg,
      variant: LibraryCodeGen.FortranVariant,
      overload: String
  ): String = arg.named.tpe match {
    case _: p.Type.Exec =>
      s"    procedure(polyregion_${decl.name.last}${overload}_${arg.named.symbol}_${variant.suffix}) :: ${arg.named.symbol}"
    case p.Type.Ptr(comp, _) =>
      s"    ${tpe(comp, variant)}, intent(${intent(arg)}) :: ${arg.named.symbol}(*)"
    case p.Type.Arr(comp, length, _) =>
      s"    ${tpe(comp, variant)}, intent(in) :: ${arg.named.symbol}($length)"
    case other => s"    ${tpe(other, variant)}, intent(in), value :: ${arg.named.symbol}"
  }

  private def signatureKey(decl: p.FunctionDecl, variant: LibraryCodeGen.FortranVariant): List[String] =
    decl.args.map(_.named.tpe match {
      case _: p.Type.Exec         => "procedure"
      case p.Type.Ptr(comp, _)    => s"array:${tpe(comp, variant).toLowerCase}"
      case p.Type.Arr(comp, _, _) => s"array:${tpe(comp, variant).toLowerCase}"
      case other                  => s"value:${tpe(other, variant).toLowerCase}"
    })

  private def procedure(
      library: p.LibraryDef,
      decl: p.FunctionDecl,
      variant: LibraryCodeGen.FortranVariant,
      overload: String
  ): String = {
    val name  = s"polyregion_${decl.name.last}${overload}_${variant.suffix}"
    val args  = decl.args.map(_.named.symbol).mkString(", ")
    val decls = decl.args.map(parameter(decl, _, variant, overload)).mkString("\n")
    val poison =
      s"""    call polyregion_import("${marker(library, decl, Some(variant.suffix))}")
         |    error stop 'compiler did not replace'""".stripMargin
    if (decl.rtn == p.Type.Unit0)
      s"""  subroutine $name($args)
         |$decls
         |$poison
         |  end subroutine $name""".stripMargin
    else
      s"""  function $name($args) result(r)
         |$decls
         |    ${tpe(decl.rtn, variant)} :: r
         |$poison
         |  end function $name""".stripMargin
  }

  def apply(library: p.LibraryDef, config: LibraryCodeGen.FortranConfig): String = {
    identifier(config.moduleName, "module")
    if (config.variants.isEmpty) fail("Fortran projection has no variants")
    config.variants.foreach { variant =>
      identifier(variant.suffix, "variant")
      variant.imports.foreach(identifier(_, "import"))
    }
    if (config.variants.map(_.suffix.toLowerCase).distinct.size != config.variants.size)
      fail("Fortran variant suffixes are not unique")
    val decls = declarations(library)
    validatePortableOverloads(decls, "Fortran")
    decls
      .groupBy(_.name.last.toLowerCase)
      .collectFirst { case (_, clashes) if clashes.map(_.name.last).distinct.size > 1 => clashes }
      .foreach(clashes =>
        fail(
          s"Fortran declaration names differ only by case: ${clashes.map(_.name.last).distinct.sorted.mkString(", ")}"
        )
      )
    decls.foreach { decl =>
      identifier(decl.name.last, "procedure")
      decl.args
        .groupBy(_.named.symbol.toLowerCase)
        .collectFirst { case (_, clashes) if clashes.size > 1 => clashes }
        .foreach { clashes =>
          fail(
            s"Fortran declaration `${decl.name.fqn
                .mkString(".")}` has arguments differing only by case: ${clashes.map(_.named.symbol).mkString(", ")}"
          )
        }
      decl.args.foreach(arg => identifier(arg.named.symbol, "argument"))
    }
    val indexedDecls = decls
      .groupBy(_.name)
      .toList
      .sortBy(_._1.fqn.mkString("."))
      .flatMap { case (_, overloads) =>
        val sorted = overloads.sortBy(_.toString)
        sorted.zipWithIndex.map { case (decl, index) =>
          decl -> Option.when(sorted.size > 1)(s"_o$index").getOrElse("")
        }
      }
    val variants = config.variants.sortBy(_.suffix)
    indexedDecls.foreach { case (decl, overload) =>
      identifier(s"polyregion_${decl.name.last}", "generic")
      variants.foreach { variant =>
        identifier(s"polyregion_${decl.name.last}${overload}_${variant.suffix}", "specific procedure")
        decl.args
          .collect { case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] => arg }
          .foreach(arg =>
            identifier(
              s"polyregion_${decl.name.last}${overload}_${arg.named.symbol}_${variant.suffix}",
              "callable interface"
            )
          )
      }
    }
    decls.foreach(decl => variants.foreach(variant => decl.tpeVars.foreach(name => tpe(p.Type.Var(name), variant))))
    indexedDecls.groupBy(_._1.name).values.foreach { overloads =>
      val projected = overloads.flatMap { case (decl, _) => variants.map(signatureKey(decl, _)) }
      projected
        .groupMapReduce(identity)(_ => 1)(_ + _)
        .collectFirst { case (signature, n) if n > 1 => signature }
        .foreach { signature =>
          fail(
            s"Fortran variants for `${overloads.head._1.name.fqn
                .mkString(".")}` are not distinguishable: ${signature.mkString("(", ", ", ")")}"
          )
        }
    }
    val imported = variants.flatMap(variant => variant.imports ::: IsoTypes).distinct.sorted.mkString(", ")
    val publics  = decls.map(_.name).distinct.map(name => s"  public :: polyregion_${name.last}").mkString("\n")
    val interfaces = indexedDecls
      .groupBy(_._1.name)
      .toList
      .sortBy(_._1.fqn.mkString("."))
      .map { case (name, overloads) =>
        val specifics = overloads
          .flatMap { case (decl, overload) =>
            variants.map(variant => s"    module procedure polyregion_${decl.name.last}${overload}_${variant.suffix}")
          }
          .mkString("\n")
        s"""  interface polyregion_${name.last}
         |$specifics
         |  end interface polyregion_${name.last}""".stripMargin
      }
      .mkString("\n\n")
    val callableInterfaces = indexedDecls.flatMap { case (decl, overload) =>
      decl.args.collect {
        case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] =>
          variants.map(variant => callable(decl, arg, arg.named.tpe.asInstanceOf[p.Type.Exec], variant, overload))
      }.flatten
    }
    val abstractInterfaces =
      if (callableInterfaces.isEmpty) ""
      else s"""  abstract interface
            |${callableInterfaces.mkString("\n\n")}
            |  end interface
            |""".stripMargin
    val procedures = indexedDecls
      .flatMap { case (decl, overload) => variants.map(procedure(library, decl, _, overload)) }
      .mkString("\n\n")
    val result = s"""module ${config.moduleName}
       |  use iso_c_binding, only: $imported
       |  implicit none
       |  private
       |
       |$publics
       |
       |  interface
       |    subroutine polyregion_import(identity)
       |      character(len=*), intent(in) :: identity
       |    end subroutine polyregion_import
       |  end interface
       |
       |$abstractInterfaces
       |$interfaces
       |
       |contains
       |
       |$procedures
       |
       |end module ${config.moduleName}
       |""".stripMargin
    result.linesIterator.zipWithIndex
      .collectFirst { case (line, index) if line.length > 132 => index + 1 }
      .foreach(line => fail(s"generated Fortran line $line exceeds 132 characters"))
    result
  }
}
