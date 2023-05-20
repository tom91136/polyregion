package polyregion.ast.mirror

import cats.conversions.variance
import polyregion.ast.MsgPack
import polyregion.ast.mirror.CppStructGen.*
import polyregion.ast.mirror.CppNlohmannJsonCodecGen

import java.lang.annotation.Target
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.mutable.ArrayBuffer

private[polyregion] case class CppNlohmannJsonCodecGen(
    namespace: List[String],
    decls: List[String],
    impls: List[String]
)
private[polyregion] object CppNlohmannJsonCodecGen {

  def fromJsonFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_from_json"
  def toJsonFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_to_json"
  def jsonAt(idx: Int)       = s"j_.at($idx)"

  def fromJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    s"size_t ord_ = ${jsonAt(0)}.get<size_t>();" ::
      s"const auto t_ = ${jsonAt(1)};" ::
      "switch (ord_) {" ::
      s.variants.zipWithIndex.map((c, i) => s"case ${i}: return ${c.tpe.ns(fromJsonFn(c.tpe))}(t_);") :::
      s"default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
      "}" :: Nil
  } else {

    val ctorInvocation = s.members match {
      case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)";
      case xs               => s.members.map(_._1).mkString("{", ", ", "}")
    }

    s.members.zipWithIndex.flatMap { case ((name, tpe), idx) =>
      tpe.kind match {
        case CppType.Kind.StdLib =>
          (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case ("std" :: "optional" :: Nil, x :: Nil) =>
              val nested =
                if (x.kind != CppType.Kind.StdLib)
                  s"${x.ns(fromJsonFn(x))}(${jsonAt(idx)})"
                else
                  s"${jsonAt(idx)}.get<${x.ref(qualified = true)}>()"

              s"auto $name = ${jsonAt(idx)}.is_null() ? std::nullopt : std::make_optional($nested);" :: Nil
            case ("std" :: "vector" :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
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

  def toJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    "return std::visit(overloaded{" ::
      s.variants.zipWithIndex.map((c, i) =>
        s"[](const ${c.tpe.ref(qualified = true)} &y_) -> json { return {$i, ${c.tpe.ns(toJsonFn(c.tpe))}(y_)}; },"
      ) ::: "[](const auto &x_) -> json { throw std::out_of_range(\"Unimplemented type:\" + to_string(x_) ); }" ::
      "}, *x_);" :: Nil
  } else {
    s.members.flatMap { case (name, tpe) =>
      tpe.kind match {
        case CppType.Kind.StdLib =>
          (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case ("std" :: "optional" :: Nil, x :: Nil) =>
              val deref =
                if (x.kind != CppType.Kind.StdLib) s"${x.ns(toJsonFn(x))}(*x_.${name})"
                else s"json{*x_.${name}}"
              s"auto $name = x_.${name} ? $deref : json{};" :: Nil

            case ("std" :: "vector" :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
              s"std::vector<json> $name;" ::
                s"std::transform(x_.${name}.begin(), x_.${name}.end(), std::back_inserter($name), &${x.ns(toJsonFn(x))});"
                :: Nil
            case _ => s"auto $name = x_.${name};" :: Nil
          }
        case _ => s"auto $name =  ${tpe.ns(toJsonFn(tpe))}(x_.${name});" :: Nil
      }
    } :::
      s"return json::array(${s.members.map(_._1).mkString("{", ", ", "}")});" ::
      Nil
  }

  def emit(s: StructNode): List[CppNlohmannJsonCodecGen] = {
    val fromJsonImpl = "" ::
      s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromJsonFn(s.tpe))}(const json& j_) { " :: //
      fromJsonBody(s).map("  " + _) :::                                                      //
      "}" :: Nil                                                                             //

    val toJsonImpl = "" ::
      s"json ${s.tpe.ns(toJsonFn(s.tpe))}(const ${s.tpe.ref(qualified = true)}& x_) { " :: //
      toJsonBody(s).map("  " + _) :::
      "}" :: Nil //

    val decls =
      s"[[nodiscard]] EXPORT ${s.tpe.ref(qualified = true)} ${fromJsonFn(s.tpe)}(const json &);" ::
        s"[[nodiscard]] EXPORT json ${toJsonFn(s.tpe)}(const ${s.tpe.ref(qualified = true)} &);" ::
        Nil

    s.variants.flatMap(s => emit(s)) :+
      CppNlohmannJsonCodecGen(s.tpe.namespace, decls, fromJsonImpl ::: toJsonImpl)
  }

  def emitHeader(namespace: String, xs: List[CppNlohmannJsonCodecGen]) = {
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
          |[[nodiscard]] EXPORT json hashed_to_json(const json&);
          |[[nodiscard]] EXPORT json hashed_from_json(const json&);
          |} // namespace $namespace
          |
          |""".stripMargin
  }

  def emitImpl(namespace: String, headerName: String, hash: String, xs: List[CppNlohmannJsonCodecGen]) =
    s"""|#include "$headerName.h"
          |
          |template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
          |template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
          |
          |namespace $namespace { 
          |${xs.flatMap(_.impls).mkString("\n")}
          |json hashed_from_json(const json& j_) { 
          |  auto hash_ = j_.at(0).get<std::string>();
          |  auto data_ = j_.at(1);
          |  if(hash_ != "$hash") {
          |   throw std::runtime_error("Expecting ADT hash to be ${hash}, but was " + hash_);
          |  }
          |  return data_;
          |}
          |
          |json hashed_to_json(const json& x_) { 
          |  return json::array({"$hash", x_});
          |}
          |} // namespace $namespace
          |""".stripMargin

}
