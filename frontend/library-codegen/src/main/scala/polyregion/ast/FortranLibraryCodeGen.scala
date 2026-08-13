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

  private def erased(tpe: p.Type): Boolean = tpe match {
    case _: p.Type.Var | _: p.Type.Struct => true
    case p.Type.Ptr(comp, _)              => erased(comp)
    case p.Type.Arr(comp, _, _)           => erased(comp)
    case p.Type.Exec(_, args, rtn)        => args.exists(erased) || erased(rtn)
    case _                                => false
  }

  private def tpe(tpe: p.Type): String = tpe match {
    case p.Type.Float32 => "real(c_float)"
    case p.Type.Float64 => "real(c_double)"
    case p.Type.IntS8   => "integer(c_int8_t)"
    case p.Type.IntS16  => "integer(c_int16_t)"
    case p.Type.IntS32  => "integer(c_int32_t)"
    case p.Type.IntS64  => "integer(c_int64_t)"
    case p.Type.Bool1   => "logical(c_bool)"
    case _: p.Type.Var  => "type(*)"
    case other          => fail(s"type has no exact Fortran library projection: $other")
  }

  private def defaultValue(tpe: p.Type): String = tpe match {
    case p.Type.Bool1 => ".false."
    case _            => "0"
  }

  private def intent(arg: p.Arg): String = arg.boundary.map(_.access) match {
    case Some(p.Arg.Access.Read)      => "in"
    case Some(p.Arg.Access.Write)     => "out"
    case Some(p.Arg.Access.ReadWrite) => "inout"
    case None                         => fail(s"pointer `${arg.named.symbol}` has no boundary")
  }

  private def erasedIntent(access: Option[String]): String = access match {
    case Some("out") => ", intent(inout)"
    case other       => s", intent(${other.getOrElse("in")})"
  }

  private def value(tpe0: p.Type, name: String, access: Option[String] = None): String = tpe0 match {
    case p.Type.Ptr(comp, _) if erased(comp) =>
      s"type(*), dimension(*)${erasedIntent(access)} :: $name"
    case p.Type.Ptr(comp, _) => s"${tpe(comp)}, intent(${access.getOrElse("in")}) :: $name(*)"
    case p.Type.Arr(comp, _, _) if erased(comp) =>
      s"type(*), dimension(*)${erasedIntent(access)} :: $name"
    case p.Type.Arr(comp, length, _)      => s"${tpe(comp)}, intent(${access.getOrElse("in")}) :: $name($length)"
    case _: p.Type.Var | _: p.Type.Struct => s"type(*)${erasedIntent(access)} :: $name"
    case _: p.Type.Exec | p.Type.Unit0    => fail(s"unsupported Fortran value parameter `$name`: $tpe0")
    case other                            => s"${tpe(other)}, intent(${access.getOrElse("in")}), value :: $name"
  }

  private def callableValue(tpe0: p.Type, name: String): String = tpe0 match {
    case p.Type.Ptr(comp, _)         => s"${tpe(comp)}, intent(in) :: $name(*)"
    case p.Type.Arr(comp, length, _) => s"${tpe(comp)}, intent(in) :: $name($length)"
    case other                       => s"${tpe(other)}, intent(in) :: $name"
  }

  private def callable(decl: p.FunctionDecl, arg: p.Arg, signature: p.Type.Exec, overload: String): String = {
    if (signature.tpeVars.nonEmpty || erased(signature))
      fail(s"generic callable `${arg.named.symbol}` must use an erased Fortran procedure declaration")
    val name = s"polyregion_${decl.name.last}${overload}_${arg.named.symbol}"
    val args = signature.args.indices.map(index => s"arg$index").mkString(", ")
    val decls =
      signature.args.zipWithIndex.map((tpe0, index) => s"      ${callableValue(tpe0, s"arg$index")}").mkString("\n")
    val imported = s"      import :: ${IsoTypes.mkString(", ")}"
    if (signature.rtn == p.Type.Unit0)
      s"""    subroutine $name($args)
         |$imported
         |$decls
         |    end subroutine $name""".stripMargin
    else
      s"""    function $name($args) result(r)
         |$imported
         |$decls
         |      ${tpe(signature.rtn)} :: r
         |    end function $name""".stripMargin
  }

  private def parameter(decl: p.FunctionDecl, arg: p.Arg, overload: String): String = arg.named.tpe match {
    case signature: p.Type.Exec if signature.tpeVars.nonEmpty || erased(signature) =>
      s"    procedure() :: ${arg.named.symbol}"
    case _: p.Type.Exec =>
      s"    procedure(polyregion_${decl.name.last}${overload}_${arg.named.symbol}) :: ${arg.named.symbol}"
    case p: p.Type.Ptr => s"    ${value(p, arg.named.symbol, Some(intent(arg)))}"
    case a: p.Type.Arr => s"    ${value(a, arg.named.symbol)}"
    case other         => s"    ${value(other, arg.named.symbol)}"
  }

  private def resultName(decl: p.FunctionDecl): String = {
    val used = decl.args.map(_.named.symbol.toLowerCase).toSet
    Iterator.iterate("polyregion_result")(_ + "_").find(name => !used(name.toLowerCase)).get
  }

  private def procedure(library: p.LibraryDef, decl: p.FunctionDecl, overload: String): String = {
    val name         = s"polyregion_${decl.name.last}$overload"
    val erasedResult = erased(decl.rtn)
    val outName      = resultName(decl)
    val logicalArgs  = decl.args.map(_.named.symbol)
    val args         = (logicalArgs ++ Option.when(erasedResult)(outName)).mkString(", ")
    val decls = decl.args.map(parameter(decl, _, overload)) ++ Option.when(erasedResult)(
      s"    type(*), intent(inout) :: $outName"
    )
    val poison =
      s"""    call polyregion_import("${marker(library, decl)}")
         |    error stop 'compiler did not replace'""".stripMargin
    if (decl.rtn == p.Type.Unit0 || erasedResult)
      s"""  subroutine $name($args)
         |${decls.mkString("\n")}
         |$poison
         |  end subroutine $name""".stripMargin
    else
      s"""  function $name($args) result(r)
         |${decls.mkString("\n")}
         |    ${tpe(decl.rtn)} :: r
         |    r = ${defaultValue(decl.rtn)}
         |$poison
         |  end function $name""".stripMargin
  }

  def apply(library: p.LibraryDef, config: LibraryCodeGen.FortranConfig): String = {
    identifier(config.moduleName, "module")
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
        .foreach(clashes =>
          fail(
            s"Fortran declaration `${decl.name.fqn
                .mkString(".")}` has arguments differing only by case: ${clashes.map(_.named.symbol).mkString(", ")}"
          )
        )
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
    indexedDecls.groupBy(_._1.name).values.foreach { overloads =>
      val physicalArities = overloads.map { case (decl, _) => decl.args.size + Option.when(erased(decl.rtn))(1).size }
      physicalArities
        .groupMapReduce(identity)(_ => 1)(_ + _)
        .collectFirst { case (arity, n) if n > 1 => arity }
        .foreach(arity =>
          fail(
            s"Fortran cannot portably distinguish overloads `${overloads.head._1.name.fqn.mkString(".")}` after result adaptation with $arity arguments"
          )
        )
      val procedureKinds = overloads.map { case (decl, _) => decl.rtn != p.Type.Unit0 && !erased(decl.rtn) }.distinct
      if (procedureKinds.size > 1)
        fail(
          s"Fortran cannot combine function and subroutine overloads `${overloads.head._1.name.fqn.mkString(".")}` after result adaptation"
        )
    }
    indexedDecls.foreach { case (decl, overload) =>
      identifier(s"polyregion_${decl.name.last}$overload", "procedure")
      decl.args.collect { case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] => arg }.foreach { arg =>
        val signature = arg.named.tpe.asInstanceOf[p.Type.Exec]
        if (signature.tpeVars.isEmpty && !erased(signature))
          identifier(s"polyregion_${decl.name.last}${overload}_${arg.named.symbol}", "callable interface")
      }
    }
    val grouped = indexedDecls.groupBy(_._1.name).toList.sortBy(_._1.fqn.mkString("."))
    val publics = grouped.map { case (name, _) => s"  public :: polyregion_${name.last}" }.mkString("\n")
    val interfaces = grouped
      .collect {
        case (name, overloads) if overloads.size > 1 =>
          val specifics = overloads
            .map { case (decl, overload) => s"    module procedure polyregion_${decl.name.last}$overload" }
            .mkString("\n")
          s"""  interface polyregion_${name.last}
           |$specifics
           |  end interface polyregion_${name.last}""".stripMargin
      }
      .mkString("\n\n")
    val callableInterfaces = indexedDecls.flatMap { case (decl, overload) =>
      decl.args.collect {
        case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] =>
          val signature = arg.named.tpe.asInstanceOf[p.Type.Exec]
          Option.when(signature.tpeVars.isEmpty && !erased(signature))(callable(decl, arg, signature, overload))
      }.flatten
    }
    val abstractInterfaces =
      if (callableInterfaces.isEmpty) ""
      else s"""  abstract interface
            |${callableInterfaces.mkString("\n\n")}
            |  end interface""".stripMargin
    val procedures = indexedDecls.map { case (decl, overload) => procedure(library, decl, overload) }.mkString("\n\n")
    val importInterface =
      """  interface
        |    subroutine polyregion_import(identity)
        |      character(len=*), intent(in) :: identity
        |    end subroutine polyregion_import
        |  end interface""".stripMargin
    val declarationSections = List(importInterface, abstractInterfaces, interfaces).filter(_.nonEmpty).mkString("\n\n")
    val result = s"""module ${config.moduleName}
       |  use iso_c_binding, only: ${IsoTypes.mkString(", ")}
       |  implicit none
       |  private
       |
       |$publics
       |
       |$declarationSections
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
