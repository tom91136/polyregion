package polyregion.ast

import scala.quoted.*

object CAbiCodeGen {

  inline def polyPassHeader: String        = ${ headerImpl }
  inline def polyPassSymbolsHeader: String = ${ symbolsImpl }
  inline def polyPassExportsList: String   = ${ exportsImpl }

  private case class Fn(cName: String, docs: Option[String], rtn: String, params: List[(String, String)])
  private case class Spec(version: Int, prefix: String, envPlugins: String, status: List[(String, Int)], fns: List[Fn])

  def headerImpl(using Quotes): Expr[String]  = Expr(renderHeader(extractSpec))
  def symbolsImpl(using Quotes): Expr[String] = Expr(renderSymbols(extractSpec))
  def exportsImpl(using Quotes): Expr[String] = Expr(renderExports(extractSpec))

  private def extractSpec(using Quotes): Spec = {
    import quotes.reflect.*
    val abi = Symbol.requiredModule("polyregion.ast.PolyAST.PolyPassAbi")

    val status = abi
      .declaredField("Status")
      .moduleClass
      .declarations
      .filter(_.isValDef)
      .flatMap(f => readConstInt(f).map(f.name -> _))

    val prefix = readConstString(abi.declaredField("Prefix"))
      .getOrElse(report.errorAndAbort("PolyPassAbi.Prefix: not a literal inline val"))

    val fns = abi.declarations
      .filter(s => s.isValDef && s.flags.is(Flags.Module) && s.moduleClass.declarations.exists(_.name == "Name"))
      .map(extractFn(_, prefix))

    Spec(
      version = readConstInt(abi.declaredField("Version")).getOrElse(report.errorAndAbort("PolyPassAbi.Version")),
      prefix = prefix,
      envPlugins =
        readConstString(abi.declaredField("EnvPlugins")).getOrElse(report.errorAndAbort("PolyPassAbi.EnvPlugins")),
      status = status,
      fns = fns
    )
  }

  private def extractFn(using Quotes)(sym: quotes.reflect.Symbol, prefix: String): Fn = {
    import quotes.reflect.*
    val applyDef = sym.moduleClass.tree
      .asInstanceOf[ClassDef]
      .body
      .collectFirst { case dd: DefDef if dd.name == "apply" => dd }
      .getOrElse(report.errorAndAbort(s"PolyPassAbi.${sym.name}: no `apply`"))
    val baseParams = applyDef.paramss match {
      case Nil                               => Nil
      case List(termParams: TermParamClause) => termParams.params.flatMap(p => expandParam(p.name, p.tpt.tpe))
      case _ => report.errorAndAbort(s"PolyPassAbi.${sym.name}: only a single term-param list supported")
    }
    val (rtn, extraParams) = expandReturn(applyDef.returnTpt.tpe)
    Fn(
      cName = prefix + AbiMacros.pascalToSnake(sym.name),
      docs = extractDocs(applyDef),
      rtn = rtn,
      params = baseParams ++ extraParams
    )
  }

  private def extractDocs(using Quotes)(d: quotes.reflect.DefDef): Option[String] = {
    import quotes.reflect.*
    d.rhs.flatMap {
      case Apply(fn, List(arg)) if fn.symbol.name == "docs" =>
        arg.asExprOf[String].value.orElse(report.errorAndAbort(s"docs() arg does not constant-fold: ${arg.show}"))
      case _ => None
    }
  }

  private def readConstString(using Quotes)(sym: quotes.reflect.Symbol): Option[String] = {
    import quotes.reflect.*
    sym.tree match {
      case v: ValDef => v.tpt.tpe match { case ConstantType(StringConstant(s)) => Some(s); case _ => None }
      case _         => None
    }
  }

  private def readConstInt(using Quotes)(sym: quotes.reflect.Symbol): Option[Int] = {
    import quotes.reflect.*
    sym.tree match {
      case v: ValDef => v.tpt.tpe match { case ConstantType(IntConstant(i)) => Some(i); case _ => None }
      case _         => None
    }
  }

  private val scalarCTypes = Map(
    "Int"    -> "uint32_t",
    "Long"   -> "size_t",
    "String" -> "const char *",
    "Unit"   -> "void",
    "Any"    -> "void *"
  )

  private def mapScalarToC(using Quotes)(tpe: quotes.reflect.TypeRepr): String = scalarCTypes.getOrElse(
    tpe.typeSymbol.name,
    quotes.reflect.report.errorAndAbort(s"unmapped Scala type: ${tpe.show}")
  )

  private def isArrayOfByte(using Quotes)(tpe: quotes.reflect.TypeRepr): Boolean = {
    import quotes.reflect.*
    tpe match {
      case AppliedType(tycon, List(arg)) => tycon.typeSymbol.name == "Array" && arg.typeSymbol.name == "Byte"
      case _                             => false
    }
  }

  private def isListOfString(using Quotes)(tpe: quotes.reflect.TypeRepr): Boolean = {
    import quotes.reflect.*
    tpe match {
      case AppliedType(tycon, List(arg)) => tycon.typeSymbol.name == "List" && arg.typeSymbol.name == "String"
      case _                             => false
    }
  }

  private def expandParam(using Quotes)(name: String, tpe: quotes.reflect.TypeRepr): List[(String, String)] =
    if (isArrayOfByte(tpe)) List(name -> "const uint8_t *", s"${name}Len" -> "size_t")
    else if (isListOfString(tpe)) List(name -> "const char *const *")
    else List(name                          -> mapScalarToC(tpe))

  private def expandReturn(using Quotes)(tpe: quotes.reflect.TypeRepr): (String, List[(String, String)]) =
    if (isArrayOfByte(tpe)) ("polypass_status_t", List("out" -> "uint8_t **", "outLen" -> "size_t *"))
    else (mapScalarToC(tpe), Nil)

  private def cParams(fn: Fn, withNames: Boolean): String =
    if (fn.params.isEmpty) "void"
    else fn.params.map { case (n, t) => if (withNames) s"$t $n" else t }.mkString(", ")

  private val Header = "// AUTO-GENERATED from PolyAST.PolyPassAbi via polyregion.ast.CodeGen. DO NOT EDIT."

  private def renderHeader(s: Spec): String = {
    val statusBody =
      s.status.map { case (n, v) => s"  POLYPASS_${AbiMacros.pascalToSnake(n).toUpperCase} = $v" }.mkString(",\n")
    val decls = s.fns
      .map { fn =>
        val doc = fn.docs.map(d => s"/**\n * $d\n */\n").getOrElse("")
        s"$doc${fn.rtn} ${fn.cName}(${cParams(fn, withNames = true)});"
      }
      .mkString("\n\n")
    val typedefs =
      s.fns.map(fn => s"typedef ${fn.rtn} (*${fn.cName}_fn)(${cParams(fn, withNames = false)});").mkString("\n")
    s"""$Header
       |
       |#ifndef POLYREGION_POLYPASS_H
       |#define POLYREGION_POLYPASS_H
       |
       |#include <stddef.h>
       |#include <stdint.h>
       |
       |#ifdef __cplusplus
       |extern "C" {
       |#endif
       |
       |#define POLYPASS_ABI_VERSION ${s.version}u
       |#define POLYPASS_ENV_PLUGINS "${s.envPlugins}"
       |
       |typedef enum polypass_status {
       |$statusBody
       |} polypass_status_t;
       |
       |$decls
       |
       |$typedefs
       |
       |#ifdef __cplusplus
       |} // extern "C"
       |#endif
       |
       |#endif // POLYREGION_POLYPASS_H
       |""".stripMargin
  }

  private def cppId(cName: String, prefix: String, suffix: String): String =
    cName.stripPrefix(prefix).split('_').filter(_.nonEmpty).map(s => s.head.toUpper +: s.tail).mkString + suffix

  private def renderSymbols(s: Spec): String = {
    val consts = s.fns
      .map { fn =>
        val doc = fn.docs.map(d => s"// $d\n").getOrElse("")
        s"""${doc}inline constexpr auto ${cppId(fn.cName, s.prefix, "")} = "${fn.cName}";"""
      }
      .mkString("\n\n")
    val aliases = s.fns.map(fn => s"using ${cppId(fn.cName, s.prefix, "Fn")} = ${fn.cName}_fn;").mkString("\n")
    s"""$Header
       |#pragma once
       |
       |#include "polyregion/polypass.h"
       |
       |namespace polyregion::polypass::abi {
       |
       |inline constexpr auto EnvPlugins = "${s.envPlugins}";
       |
       |$consts
       |
       |$aliases
       |
       |} // namespace polyregion::polypass::abi
       |""".stripMargin
  }

  private def renderExports(s: Spec): String = s.fns.map(_.cName).mkString("", "\n", "\n")
}
