package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object CppLibraryCodeGen {
  import LibraryCodeGen.Support.*

  private val Keywords = Set(
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq"
  )

  private def identifier(value: String, what: String): String = {
    targetIdentifier(value, s"C++ $what", Keywords)
    if (value.startsWith("_")) fail(s"C++ $what identifier `$value` is reserved") else value
  }

  private def pascalCase(name: String): String =
    name.split('_').iterator.filter(_.nonEmpty).map(word => word.head.toUpper +: word.tail).mkString

  private def tpe(value: p.Type): String = value match {
    case p.Type.Float16           => fail("Float16 has no exact C++17 library projection")
    case p.Type.Float32           => "float"
    case p.Type.Float64           => "double"
    case p.Type.IntU8             => "std::uint8_t"
    case p.Type.IntU16            => "std::uint16_t"
    case p.Type.IntU32            => "std::uint32_t"
    case p.Type.IntU64            => "std::uint64_t"
    case p.Type.IntS8             => "std::int8_t"
    case p.Type.IntS16            => "std::int16_t"
    case p.Type.IntS32            => "std::int32_t"
    case p.Type.IntS64            => "std::int64_t"
    case p.Type.Bool1             => "bool"
    case p.Type.Unit0             => "void"
    case p.Type.Var(name)         => identifier(name, "type variable")
    case p.Type.Struct(name, Nil) => s"::${name.fqn.map(identifier(_, "type")).mkString("::")}"
    case p.Type.Struct(name, args) =>
      s"::${name.fqn.map(identifier(_, "type")).mkString("::")}<${args.map(tpe).mkString(", ")}>"
    case p.Type.Ptr(comp, _)  => s"${tpe(comp)} *"
    case p.Type.Arr(_, _, _)  => fail("C++ array types require a parameter declarator")
    case p.Type.Nothing       => fail("PolyAST Nothing has no C++ library projection")
    case p.Type.Exec(_, _, _) => fail("callable types are projected as named C++ template parameters")
    case p.Type.FnRef(name)   => fail(s"function reference `${name.fqn.mkString(".")}` has no C++ projection")
  }

  private def parameter(arg: p.Arg): String = arg.named.tpe match {
    case p.Type.Exec(_, _, _) => s"${pascalCase(arg.named.symbol)} ${arg.named.symbol}"
    case p.Type.Ptr(comp, _) =>
      val component = arg.boundary.map(_.access) match {
        case Some(p.Arg.Access.Read) => s"const ${tpe(comp)}"
        case _                       => tpe(comp)
      }
      s"$component *${arg.named.symbol}"
    case p.Type.Arr(comp, length, _) => s"${tpe(comp)} (&${arg.named.symbol})[$length]"
    case other                       => s"${tpe(other)} ${arg.named.symbol}"
  }

  private def callableCheck(arg: p.Arg): List[String] = arg.named.tpe match {
    case p.Type.Exec(tpeVars, _, _) if tpeVars.nonEmpty =>
      fail(s"generic callable argument `${arg.named.symbol}` is not supported yet")
    case p.Type.Exec(_, args, rtn) =>
      val invokeArgs =
        (s"${pascalCase(arg.named.symbol)} &" :: args.map(tpe(_)).map("const " + _ + " &")).mkString(", ")
      List(
        s"static_assert(std::is_same_v<std::invoke_result_t<$invokeArgs>, ${tpe(rtn)}>, \"callable signature mismatch\");"
      )
    case _ => Nil
  }

  private def declaration(library: p.LibraryDef, decl: p.FunctionDecl): String = {
    decl.rtn match {
      case p.Type.Ptr(_, _) | p.Type.Arr(_, _, _) => fail(s"library return type is not supported yet: ${decl.rtn}")
      case _                                      => ()
    }
    val callableTemplates = decl.args.collect {
      case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] => s"class ${pascalCase(arg.named.symbol)}"
    }
    val templates = decl.tpeVars.map(name => s"class $name") ++ callableTemplates
    if (templates.distinct.size != templates.size)
      fail(s"declaration `${decl.name.fqn.mkString(".")}` has colliding C++ template parameters")
    val template = if (templates.nonEmpty) s"template <${templates.mkString(", ")}>\n" else ""
    val body     = (decl.args.flatMap(callableCheck) :+ "__builtin_trap();").map(s => s"  $s").mkString("\n")
    s"""$template[[clang::annotate("${marker(library, decl)}")]] inline ${tpe(decl.rtn)} ${decl.name.last}(${decl.args
        .map(parameter)
        .mkString(", ")}) {
       |$body
       |}""".stripMargin
  }

  def apply(library: p.LibraryDef): String = {
    val decls = declarations(library)
    validatePortableOverloads(decls, "C++")
    library.name.fqn.foreach(identifier(_, "namespace"))
    decls.foreach { decl =>
      identifier(decl.name.last, "function")
      decl.tpeVars.foreach(identifier(_, "type variable"))
      decl.args.foreach(arg => identifier(arg.named.symbol, "parameter"))
      decl.args
        .collect { case arg if arg.named.tpe.isInstanceOf[p.Type.Exec] => pascalCase(arg.named.symbol) }
        .foreach(identifier(_, "callable template"))
    }
    val body = decls.map(declaration(library, _)).mkString("\n\n")
    s"""#pragma once
       |
       |#include <cstdint>
       |#include <type_traits>
       |
       |namespace ${library.name.fqn.mkString("::")} {
       |
       |$body
       |
       |} // namespace ${library.name.fqn.mkString("::")}
       |""".stripMargin
  }
}
