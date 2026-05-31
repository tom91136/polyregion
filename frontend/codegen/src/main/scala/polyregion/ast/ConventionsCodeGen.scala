package polyregion.ast

import scala.quoted.*

object ConventionsCodeGen {

  inline def conventionsHeader: String = ${ headerImpl }

  private case class Entry(name: String, value: String, isMacro: Boolean)

  def headerImpl(using Quotes): Expr[String] = Expr(render(extract))

  private def constStrings(using Quotes)(mod: quotes.reflect.Symbol): List[(String, String)] = {
    import quotes.reflect.*
    mod.declarations.filter(_.isValDef).flatMap { f =>
      f.tree match {
        case v: ValDef =>
          v.tpt.tpe match {
            case ConstantType(StringConstant(s)) => Some(f.name -> s)
            case _                               => None
          }
        case _ => None
      }
    }
  }

  private def extract(using Quotes): List[Entry] = {
    import quotes.reflect.*
    val conventions = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions")
    val macros      = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.Macros")
    constStrings(conventions).map((n, v) => Entry(n, v, isMacro = false)) ++
      constStrings(macros).map((n, v) => Entry(n, v, isMacro = true))
  }

  private def render(entries: List[Entry]): String = {
    val macros = entries
      .filter(_.isMacro)
      .map(e => s"""#define ${AbiMacros.pascalToSnake(e.name).toUpperCase} "${e.value}"""")
      .mkString("\n")
    val consts =
      entries.filterNot(_.isMacro).map(e => s"""inline constexpr auto ${e.name} = "${e.value}";""").mkString("\n")
    s"""|// AUTO-GENERATED from PolyAST.Conventions via polyregion.ast.CodeGen. DO NOT EDIT.
        |#pragma once
        |
        |$macros
        |
        |namespace polyregion::conventions {
        |
        |$consts
        |
        |} // namespace polyregion::conventions
        |""".stripMargin
  }
}
