package polyregion.ast

import polyregion.data.Cpp._
import polyregion.data.MsgPack
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.lang.annotation.Target

import scala.collection.mutable.ArrayBuffer
import cats.conversions.variance
import java.nio.file.Path
import polyregion.PolyregionCompiler

object CppCodeGen {

  case class NlohmannJsonCodecSource(namespace: List[String], decls: List[String], impls: List[String])
  object NlohmannJsonCodecSource {
    def emit(s: StructNode): List[NlohmannJsonCodecSource] = {

      def fromJsonFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_from_json"
      def toJsonFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_to_json"
      def jsonAt(idx: Int)       = s"j.at($idx)"

      val fromJsonBody = //
        if (s.tpe.kind == CppType.Kind.Base) {
          s"size_t ord = ${jsonAt(0)}.get<size_t>();" ::
            s"const auto t = ${jsonAt(1)};" ::
            "switch (ord) {" ::
            s.variants.zipWithIndex.map((c, i) => s"case ${i}: return ${c.tpe.ns(fromJsonFn(c.tpe))}(t);") :::
            s"default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord));" ::
            "}" :: Nil
        } else {

          val ctorInvocation = s.members match {
            case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)";
            case xs               => s.members.map(_._1).mkString("{", ", ", "}")
          }

          s.members.zipWithIndex.flatMap { case ((name, tpe), idx) =>
            tpe.kind match {
              case CppType.Kind.StdLib =>
                (tpe.namespace, tpe.name, tpe.ctors) match {
                  case ("std" :: Nil, "vector", x :: Nil) if x.kind != CppType.Kind.StdLib =>
                    s"${tpe.ref(qualified = true)} $name;" ::
                      s"auto ${name}_json = ${jsonAt(idx)};" ::
                      s"std::transform(${name}_json.begin(), ${name}_json.end(), std::back_inserter($name), &${x.ns(fromJsonFn(x))});"
                      :: Nil
                  case _ => s"auto $name = ${jsonAt(idx)}.get<${tpe.ref(qualified = true)}>();" :: Nil
                }
              case _ => s"auto $name =  ${tpe.ns(fromJsonFn(tpe))}(${jsonAt(idx)});" :: Nil
            }
          } :::
            s"return ${ctorInvocation};" ::
            Nil
        }

        val toJsonBody = //
          if (s.tpe.kind == CppType.Kind.Base) {
            "return std::visit(overloaded{" ::
              s.variants.zipWithIndex.map((c, i) =>
                s"[](const ${c.tpe.ref(qualified = true)} &y) -> json { return {$i, ${c.tpe.ns(toJsonFn(c.tpe))}(y)}; },"
              ) ::: "[](const auto &x) -> json { throw std::out_of_range(\"Unimplemented type:\" + to_string(x) ); }" ::
              "}, *x);" :: Nil
          } else {
            s.members.flatMap { case (name, tpe) =>
              tpe.kind match {
                case CppType.Kind.StdLib =>
                  (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
                    case ("std" :: "vector" :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
                      s"std::vector<json> $name;" ::
                        s"std::transform(x.${name}.begin(), x.${name}.end(), std::back_inserter($name), &${x.ns(toJsonFn(x))});"
                        :: Nil
                    case _ => s"auto $name = x.${name};" :: Nil
                  }
                case _ => s"auto $name =  ${tpe.ns(toJsonFn(tpe))}(x.${name});" :: Nil
              }
            } :::
              s"return json::array(${s.members.map(_._1).mkString("{", ", ", "}")});" ::
              Nil
          }

      val fromJsonImpl = "" ::
        s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromJsonFn(s.tpe))}(const json& j) { " :: //
        fromJsonBody.map("  " + _) :::                                                        //
        "}" :: Nil                                                                            //

      val toJsonImpl = "" ::
        s"json ${s.tpe.ns(toJsonFn(s.tpe))}(const ${s.tpe.ref(qualified = true)}& x) { " :: //
        toJsonBody.map("  " + _) :::
        "}" :: Nil //

      val decls =
        s"[[nodiscard]] EXPORT ${s.tpe.ref(qualified = true)} ${fromJsonFn(s.tpe)}(const json &);" ::
          s"[[nodiscard]] EXPORT json ${toJsonFn(s.tpe)}(const ${s.tpe.ref(qualified = true)} &);" ::
          Nil

      s.variants.flatMap(s => emit(s)) :+ NlohmannJsonCodecSource(s.tpe.namespace, decls, fromJsonImpl ::: toJsonImpl)

    }

    def emitHeader(namespace: String, xs: List[NlohmannJsonCodecSource]) = {
      val lines = xs
        .groupMapReduce(_.namespace.mkString("::"))(_.decls)(_ ::: _)
        .toList
        .flatMap {
          case ("", lines) => lines
          case (ns, lines) => s"namespace $ns { " :: lines ::: s"} // namespace $ns" :: Nil

        }
      s"""|#pragma once
          |
          |#include "json.hpp"
          |#include "polyast.h"
          |#include "export.h"
          |
          |using json = nlohmann::json;
          |
          |namespace ${namespace} { 
          |${lines.mkString("\n")}
          |} // namespace $namespace
          |
          |""".stripMargin
    }

    def emitImpl(namespace: String, headerName: String, xs: List[NlohmannJsonCodecSource]) =
      s"""|#include "$headerName.h"
          |
          |template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
          |template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
          |
          |namespace $namespace { 
          |${xs.flatMap(_.impls).mkString("\n")}
          |} // namespace $namespace
          |""".stripMargin

  }

  @main def main(): Unit = {

    println("\n=========\n")

    import PolyAst._

    val ast: Stmt = Stmt.Cond(
      Expr.Alias(Term.BoolConst(true)),
      Stmt.Comment("a") :: Stmt.Return(Expr.Alias(Term.FloatConst(1.24f))) :: Nil,
      Stmt.Comment("b") :: Nil
    )

    // pprint.pprintln(MsgPack.encodeMsg(ast))
    println(MsgPack.encode(ast).length)

    // println(MsgPack.encodeMsg(ast))
    println(MsgPack.decode[Stmt](MsgPack.encode(ast)))

    println(MsgPack.decode[Stmt](MsgPack.encode(ast)).right.get == ast)

    Files.write(
      Paths.get("..").resolve("native/ast.msgpack").normalize.toAbsolutePath,
      MsgPack.encode(ast),
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    val structs = deriveStruct[Sym]() //
      :: deriveStruct[TypeKind]()
      :: deriveStruct[Type]()
      :: deriveStruct[Named]()
      :: deriveStruct[Position]()
      :: deriveStruct[Term]()
      :: deriveStruct[Expr]()
      :: deriveStruct[Stmt]()
      :: deriveStruct[Function]()
      :: deriveStruct[StructDef]()
      :: Nil

    val sources = structs.flatMap(_.emit)

    val jsonSources = structs.flatMap(NlohmannJsonCodecSource.emit(_))

    println("\n=========\n")

    val ns = "polyregion::polyast"

    val target = Paths.get(".").resolve("native/codegen/generated/").normalize.toAbsolutePath

    Files.createDirectories(target)
    println(s"Dest=${target}")
    println("\n=========\n")

    def overwrite(path: Path)(content: String) = Files.writeString(
      path,
      content,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    overwrite(target.resolve("polyast.h"))(StructSource.emitHeader(ns, sources))
    overwrite(target.resolve("polyast.cpp"))(StructSource.emitImpl(ns, "polyast", sources))

    overwrite(target.resolve("polyast_codec.h"))(NlohmannJsonCodecSource.emitHeader(ns, jsonSources))
    overwrite(target.resolve("polyast_codec.cpp"))(NlohmannJsonCodecSource.emitImpl(ns, "polyast_codec", jsonSources))

    println(MsgPack.encode(PolyAst.Expr.Alias(PolyAst.Term.IntConst(1))).map(_.toInt.toHexString).toList)
    // MsgPack.encodeMsg(PolyAst.Expr.Alias(PolyAst.Term.IntConst(1)))

    ()

    PolyregionCompiler.load()
    try {
      val c = PolyregionCompiler.compile(Array(), true, 0);
      println(c.program)
      println(c.disassembly)
      println(c.messages)
      println(c.elapsed)
    } catch {
      case e => e.printStackTrace();

    }
    println("Done")

  }

}
