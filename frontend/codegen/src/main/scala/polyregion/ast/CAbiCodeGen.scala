package polyregion.ast

import scala.quoted.*

object CAbiCodeGen {

  inline def polyPassHeader: String        = ${ headerImpl("polyregion.ast.PolyAST.PolyPassAbi") }
  inline def polyPassSymbolsHeader: String = ${ symbolsImpl("polyregion.ast.PolyAST.PolyPassAbi") }
  inline def polyPassExportsList: String   = ${ exportsImpl("polyregion.ast.PolyAST.PolyPassAbi") }

  inline def polyJitHeader: String        = ${ headerImpl("polyregion.ast.PolyAST.PolyJitAbi") }
  inline def polyJitSymbolsHeader: String = ${ symbolsImpl("polyregion.ast.PolyAST.PolyJitAbi") }
  inline def polyJitExportsList: String   = ${ exportsImpl("polyregion.ast.PolyAST.PolyJitAbi") }

  private case class Fn(cName: String, docs: Option[String], rtn: String, params: List[(String, String)])
  private case class CStruct(cName: String, members: List[(String, String)])
  private case class Spec(
      module: String,
      version: Int,
      prefix: String,
      envPlugins: Option[String],
      status: List[(String, Int)],
      structs: List[CStruct],
      fns: List[Fn]
  ) {
    val base      = prefix.stripSuffix("_")
    val guard     = s"POLYREGION_${base.toUpperCase}_H"
    val statusTpe = s"${prefix}status"
    val macroPfx  = prefix.toUpperCase
    val namespace = s"polyregion::${base}::abi"
    val include   = s"polyregion/${base}.h"
    val banner    = s"// AUTO-GENERATED from PolyAST.$module via polyregion.ast.CodeGen. DO NOT EDIT."
  }

  def headerImpl(fqn: String)(using Quotes): Expr[String]  = Expr(renderHeader(extractSpec(fqn)))
  def symbolsImpl(fqn: String)(using Quotes): Expr[String] = Expr(renderSymbols(extractSpec(fqn)))
  def exportsImpl(fqn: String)(using Quotes): Expr[String] = Expr(renderExports(extractSpec(fqn)))

  private def extractSpec(fqn: String)(using Quotes): Spec = {
    import quotes.reflect.*
    val abi = Symbol.requiredModule(fqn)

    val status = abi
      .declaredField("Status")
      .moduleClass
      .declarations
      .filter(_.isValDef)
      .flatMap(f => readConstInt(f).map(f.name -> _))

    val prefix = readConstString(abi.declaredField("Prefix"))
      .getOrElse(report.errorAndAbort(s"$fqn.Prefix: not a literal inline val"))

    val fns = abi.declarations
      .filter(s => s.isValDef && s.flags.is(Flags.Module) && s.moduleClass.declarations.exists(_.name == "Name"))
      .map(extractFn(_, fqn, prefix))

    Spec(
      module = fqn.split('.').last,
      version = readConstInt(abi.declaredField("Version")).getOrElse(report.errorAndAbort(s"$fqn.Version")),
      prefix = prefix,
      envPlugins = abi.declarations.find(_.name == "EnvPlugins").flatMap(readConstString),
      status = status,
      structs = fns.flatMap(_.structs).distinctBy(_.cName),
      fns = fns.map(_.fn)
    )
  }

  private case class ExtractedFn(fn: Fn, structs: List[CStruct])

  private def extractFn(using Quotes)(sym: quotes.reflect.Symbol, fqn: String, prefix: String): ExtractedFn = {
    import quotes.reflect.*
    val applyDef = sym.moduleClass.tree
      .asInstanceOf[ClassDef]
      .body
      .collectFirst { case dd: DefDef if dd.name == "apply" => dd }
      .getOrElse(report.errorAndAbort(s"$fqn.${sym.name}: no `apply`"))
    val params = applyDef.paramss match {
      case Nil                               => Nil
      case List(termParams: TermParamClause) => termParams.params
      case _ => report.errorAndAbort(s"$fqn.${sym.name}: only a single term-param list supported")
    }
    val (rtn, extraParams) = expandReturn(applyDef.returnTpt.tpe, prefix)
    ExtractedFn(
      Fn(
        cName = prefix + AbiMacros.pascalToSnake(sym.name),
        docs = extractDocs(applyDef),
        rtn = rtn,
        params = params.flatMap(p => expandParam(p.name, p.tpt.tpe, prefix)) ++ extraParams
      ),
      params.flatMap(p => listOfCaseClass(p.tpt.tpe).map(cStructOf(_, prefix)))
    )
  }

  private def listOfCaseClass(using Quotes)(tpe: quotes.reflect.TypeRepr): Option[quotes.reflect.Symbol] = {
    import quotes.reflect.*
    tpe match {
      case AppliedType(l, List(arg)) if l.typeSymbol.name == "List" && arg.typeSymbol.flags.is(Flags.Case) =>
        Some(arg.typeSymbol)
      case _ => None
    }
  }

  private def cStructOf(using Quotes)(sym: quotes.reflect.Symbol, prefix: String): CStruct = {
    import quotes.reflect.*
    CStruct(
      prefix + AbiMacros.pascalToSnake(sym.name),
      sym.caseFields.flatMap(f => expandParam(f.name, f.tree.asInstanceOf[ValDef].tpt.tpe, prefix))
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

  private def expandParam(using
      Quotes
  )(name: String, tpe: quotes.reflect.TypeRepr, prefix: String): List[(String, String)] =
    if (isArrayOfByte(tpe)) List(name -> "const uint8_t *", s"${name}Len" -> "size_t")
    else if (isListOfString(tpe)) List(name -> "const char *const *")
    else
      listOfCaseClass(tpe) match {
        case Some(sym) =>
          List(name -> s"const ${prefix}${AbiMacros.pascalToSnake(sym.name)}_t *", s"${name}Len" -> "size_t")
        case None => List(name -> mapScalarToC(tpe))
      }

  private def expandReturn(using
      Quotes
  )(tpe: quotes.reflect.TypeRepr, prefix: String): (String, List[(String, String)]) =
    if (isArrayOfByte(tpe)) (s"${prefix}status_t", List("out" -> "uint8_t **", "outLen" -> "size_t *"))
    else (mapScalarToC(tpe), Nil)

  private def cParams(fn: Fn, withNames: Boolean): String =
    if (fn.params.isEmpty) "void"
    else fn.params.map { case (n, t) => if (withNames) s"$t $n" else t }.mkString(", ")

  private def renderHeader(s: Spec): String = {
    val statusBody =
      s.status.map { case (n, v) => s"  ${s.macroPfx}${AbiMacros.pascalToSnake(n).toUpperCase} = $v" }.mkString(",\n")
    val decls = s.fns
      .map { fn =>
        val doc = fn.docs.map(d => s"/**\n * $d\n */\n").getOrElse("")
        s"$doc${s.macroPfx}EXPORT ${fn.rtn} ${fn.cName}(${cParams(fn, withNames = true)});"
      }
      .mkString("\n\n")
    val typedefs =
      s.fns.map(fn => s"typedef ${fn.rtn} (*${fn.cName}_fn)(${cParams(fn, withNames = false)});").mkString("\n")
    val structBlock =
      if (s.structs.isEmpty) ""
      else
        s.structs
          .map(st =>
            s"typedef struct ${st.cName} {\n${st.members.map((n, t) => s"  $t $n;").mkString("\n")}\n} ${st.cName}_t;"
          )
          .mkString("", "\n\n", "\n\n")
    val envDefine = s.envPlugins.map(e => s"""#define ${s.macroPfx}ENV_PLUGINS "$e"\n""").getOrElse("")
    s"""${s.banner}
       |
       |#ifndef ${s.guard}
       |#define ${s.guard}
       |
       |#include <stddef.h>
       |#include <stdint.h>
       |
       |#if defined(_WIN32) && defined(${s.macroPfx}BUILD)
       |  #define ${s.macroPfx}EXPORT __declspec(dllexport)
       |#elif defined(_WIN32)
       |  #define ${s.macroPfx}EXPORT
       |#else
       |  #define ${s.macroPfx}EXPORT __attribute__((visibility("default")))
       |#endif
       |
       |#ifdef __cplusplus
       |extern "C" {
       |#endif
       |
       |#define ${s.macroPfx}ABI_VERSION ${s.version}u
       |$envDefine
       |typedef enum ${s.statusTpe} {
       |$statusBody
       |} ${s.statusTpe}_t;
       |
       |$structBlock$decls
       |
       |$typedefs
       |
       |#ifdef __cplusplus
       |} // extern "C"
       |#endif
       |
       |#endif // ${s.guard}
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
    val aliases  = s.fns.map(fn => s"using ${cppId(fn.cName, s.prefix, "Fn")} = ${fn.cName}_fn;").mkString("\n")
    val envConst = s.envPlugins.map(e => s"""inline constexpr auto EnvPlugins = "$e";\n\n""").getOrElse("")
    s"""${s.banner}
       |#pragma once
       |
       |#include "${s.include}"
       |
       |namespace ${s.namespace} {
       |
       |$envConst$consts
       |
       |$aliases
       |
       |} // namespace ${s.namespace}
       |""".stripMargin
  }

  private def renderExports(s: Spec): String = s.fns.map(_.cName).mkString("", "\n", "\n")
}
