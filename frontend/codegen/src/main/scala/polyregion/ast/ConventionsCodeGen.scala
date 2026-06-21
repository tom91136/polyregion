package polyregion.ast

import scala.quoted.*

object ConventionsCodeGen {

  inline def conventionsHeader: String = ${ headerImpl }

  private enum Kind { case Const, Macro, Abi, Reflect }
  private case class Entry(name: String, value: String, kind: Kind)

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

  private def constLiterals(using Quotes)(mod: quotes.reflect.Symbol): List[(String, String)] = {
    import quotes.reflect.*
    mod.declarations.filter(_.isValDef).flatMap { f =>
      f.tree match {
        case v: ValDef =>
          v.tpt.tpe match {
            case ConstantType(StringConstant(s)) => Some(f.name -> s"\"$s\"")
            case ConstantType(LongConstant(l))   => Some(f.name -> s"${l}LL")
            case ConstantType(IntConstant(i))    => Some(f.name -> i.toString)
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
    val runtimeAbi  = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.RuntimeAbi")
    val reflect     = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.Reflect")
    constLiterals(conventions).map((n, v) => Entry(n, v, Kind.Const)) ++
      constStrings(macros).map((n, v) => Entry(n, v, Kind.Macro)) ++
      constStrings(runtimeAbi).map((n, v) => Entry(n, v, Kind.Abi)) ++
      constStrings(reflect).map((n, v) => Entry(n, v, Kind.Reflect))
  }

  private def render(entries: List[Entry]): String = {
    val macros = entries
      .collect { case Entry(n, v, Kind.Macro) =>
        s"""#define ${AbiMacros.pascalToSnake(n).toUpperCase} "$v""""
      }
      .mkString("\n")
    val consts = entries.collect { case Entry(n, v, Kind.Const) => s"inline constexpr auto $n = $v;" }.mkString("\n")
    val reflect =
      entries.collect { case Entry(n, v, Kind.Reflect) => s"""inline constexpr auto $n = "$v";""" }.mkString("\n")
    val abiList = entries.collect { case Entry(n, v, Kind.Abi) => s"  X($n, $v)" }.mkString(" \\\n")
    s"""|// AUTO-GENERATED from PolyAST.Conventions via polyregion.ast.CodeGen. DO NOT EDIT.
        |#pragma once
        |
        |$macros
        |
        |#define POLYREGION_RUNTIME_ABI(X) \\
        |$abiList
        |
        |namespace polyregion::conventions {
        |
        |$consts
        |
        |namespace reflect {
        |$reflect
        |} // namespace reflect
        |
        |} // namespace polyregion::conventions
        |""".stripMargin
  }
}
