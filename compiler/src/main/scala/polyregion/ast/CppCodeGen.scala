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

object CppCodeGen {

  case class NlohmannJsonCodecSource(namespace: List[String], decls: List[String], impls: List[String])
  object NlohmannJsonCodecSource {
    def emit(s: StructNode): List[NlohmannJsonCodecSource] = {

      def fnName(t: CppType) = t.ref(qualified = false).toLowerCase + "_json"
      def select(idx: Int)   = s"j.at($idx)"

      val body = //
        if (s.tpe.kind == CppType.Kind.Base) {
          s"size_t ord = ${select(0)}.get<size_t>();" ::
            s"const auto t = ${select(1)};" ::
            "switch (ord) {" ::
            s.variants.zipWithIndex.map((c, i) => s"case ${i}: return ${c.tpe.ns(fnName(c.tpe))}(t);") :::
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
                      s"auto ${name}_json = ${select(idx)};" ::
                      s"std::transform(${name}_json.begin(), ${name}_json.end(), std::back_inserter($name), &${x.ns(fnName(x))});"
                      :: Nil
                  case _ => s"auto $name = ${select(idx)}.get<${tpe.ref(qualified = true)}>();" :: Nil
                }
              case _ => s"auto $name =  ${tpe.ns(fnName(tpe))}(${select(idx)});" :: Nil
            }
          } :::
            s"return ${ctorInvocation};" ::
            Nil
        }

      val impls = "" ::
        s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fnName(s.tpe))}(const json& j) { " :: //
        body.map("  " + _) :::                                                            //
        "}" :: Nil                                                                        //

      val decls = s"[[nodiscard]] ${s.tpe.ref(qualified = true)} ${fnName(s.tpe)}(const json &);" :: Nil

      s.variants.flatMap(s => emit(s)) :+ NlohmannJsonCodecSource(s.tpe.namespace, decls, impls)

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
          |namespace $namespace { 
          |${xs.flatMap(_.impls).mkString("\n")}
          |} // namespace $namespace
          |""".stripMargin

  }

  @main def main(): Unit = {

    println("\n=========\n")

    import PolyAst._

    import MsgPack.Codec

    given MsgPack.Codec[Sym]       = Codec.derived
    given MsgPack.Codec[TypeKind]  = Codec.derived
    given MsgPack.Codec[Type]      = Codec.derived
    given MsgPack.Codec[Term]      = Codec.derived
    given MsgPack.Codec[Named]     = Codec.derived
    given MsgPack.Codec[Position]  = Codec.derived
    given MsgPack.Codec[Intr]      = Codec.derived
    given MsgPack.Codec[Expr]      = Codec.derived
    given MsgPack.Codec[Stmt]      = Codec.derived
    given MsgPack.Codec[Function]  = Codec.derived
    given MsgPack.Codec[StructDef] = Codec.derived

    val ast: Stmt = Stmt.Cond(
      Expr.Alias(Term.BoolConst(true)),
      Stmt.Comment("a") :: Stmt.Return(Expr.Alias(Term.FloatConst(1.24f))) :: Nil,
      Stmt.Comment("b") :: Nil
    )

    pprint.pprintln(MsgPack.encodeMsg(ast))
    println(MsgPack.encode(ast).length)

    // println(MsgPack.encodeMsg(ast))
    println(MsgPack.decode[Stmt](MsgPack.encode(ast)))

    println(MsgPack.decode[Stmt](MsgPack.encode(ast)).right.get == ast)

    Files.write(
      Paths.get(".").resolve("native/ast.msgpack").normalize.toAbsolutePath,
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

    val target = Paths.get(".").resolve("native/src/generated/").normalize.toAbsolutePath

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

    ()
  }

}
