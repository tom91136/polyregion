package polyregion.ast

import scala.quoted.*

object ConventionsCodeGen {

  inline def conventionsHeader: String = ${ headerImpl }
  inline def hostAbiCmake: String      = ${ hostAbiCmakeImpl }

  private enum Kind { case Const, Macro, Abi, HostAbi, Reflect }
  private case class Entry(name: String, value: String, kind: Kind)

  def headerImpl(using Quotes): Expr[String] = Expr(render(extract))

  def hostAbiCmakeImpl(using Quotes): Expr[String] = Expr(renderCmake(extract))

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
    val conventions        = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions")
    val macros             = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.Macros")
    val runtimeAbi         = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.RuntimeAbi")
    val hostAbi            = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.HostAbi")
    val reflect            = Symbol.requiredModule("polyregion.ast.PolyAST.Conventions.Reflect")
    val assertMessageLimit = PolyAST.Conventions.assertMessageLimit
    constLiterals(conventions).map((n, v) => Entry(n, v, Kind.Const)) ++
      List(Entry("AssertMessageLimit", assertMessageLimit.toString, Kind.Const)) ++
      constStrings(macros).map((n, v) => Entry(n, v, Kind.Macro)) ++
      constStrings(runtimeAbi).map((n, v) => Entry(n, v, Kind.Abi)) ++
      constStrings(hostAbi).map((n, v) => Entry(n, v, Kind.HostAbi)) ++
      constStrings(reflect).map((n, v) => Entry(n, v, Kind.Reflect))
  }

  private def renderCmake(entries: List[Entry]): String = {
    val symbols = entries.collect { case Entry(_, v, Kind.HostAbi) => s"  $v" }.mkString("\n")
    s"""|# AUTO-GENERATED from PolyAST.Conventions.HostAbi via polyregion.ast.CodeGen. DO NOT EDIT.
        |set(POLYREGION_HOST_ABI_SYMBOLS
        |$symbols
        |)
        |""".stripMargin
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
    val abiList     = entries.collect { case Entry(n, v, Kind.Abi) => s"  X($n, $v)" }.mkString(" \\\n")
    val hostAbiList = entries.collect { case Entry(n, v, Kind.HostAbi) => s"  X($n, $v)" }.mkString(" \\\n")
    s"""|// AUTO-GENERATED from PolyAST.Conventions via polyregion.ast.CodeGen. DO NOT EDIT.
        |#pragma once
        |
        |$macros
        |
        |#define POLYREGION_RUNTIME_ABI(X) \\
        |$abiList
        |
        |#define POLYREGION_HOST_ABI(X) \\
        |$hostAbiList
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
