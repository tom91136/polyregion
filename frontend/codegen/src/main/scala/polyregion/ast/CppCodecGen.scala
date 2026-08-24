package polyregion.ast

private[polyregion] object CppCodecGen {

  private def namespaced(lines: List[(List[String], List[String])]): String =
    lines
      .groupMapReduce(_._1.mkString("::"))(_._2)(_ ::: _)
      .toList
      .flatMap {
        case ("", body) => body
        case (ns, body) => s"namespace $ns {" :: body ::: s"} // namespace $ns" :: Nil
      }
      .mkString("\n")

  def emitHeader(
      namespace: String,
      json: List[CppNlohmannJsonCodecGen],
      roots: List[CppMsgPackCodecGen.Root]
  ): String = {
    val jsonDecls    = namespaced(json.map(x => x.namespace -> x.decls))
    val msgpackDecls = roots.flatMap(CppMsgPackCodecGen.rootDecls).mkString("\n")
    s"""|#pragma once
        |
        |#include <cstdint>
        |#include <vector>
        |
        |#include "nlohmann/json.hpp"
        |#include "polyast.h"
        |#include "polyregion/export.h"
        |
        |using json = nlohmann::json;
        |
        |namespace $namespace {
        |$jsonDecls
        |[[nodiscard]] POLYREGION_EXPORT json hashed_to_json(const json&);
        |[[nodiscard]] POLYREGION_EXPORT json hashed_from_json(const json&);
        |
        |$msgpackDecls
        |} // namespace $namespace
        |""".stripMargin
  }

  def emitImpl(
      namespace: String,
      headerName: String,
      hash: String,
      programHash: String,
      packageHash: String,
      packageWireHash: String,
      json: List[CppNlohmannJsonCodecGen],
      msgpack: List[CppMsgPackCodecGen],
      roots: List[CppMsgPackCodecGen.Root]
  ): String = {
    val msgpackForwardDecls = namespaced(msgpack.map(x => x.namespace -> x.forwardDecls))
    val msgpackImpls        = msgpack.flatMap(_.impls).mkString("\n")
    val msgpackRoots        = roots.flatMap(CppMsgPackCodecGen.rootImpl).mkString("\n")
    s"""|#include "$headerName.h"
        |#include "msgpack.hpp"
        |
        |template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
        |template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
        |
        |namespace $namespace {
        |constexpr auto AdtHash = "$hash";
        |constexpr auto ProgramHash = "$programHash";
        |constexpr auto PackageHash = "$packageHash";
        |constexpr auto PackageWireHash = "$packageWireHash";
        |using msgpack::decodeMaybeInterned;
        |using msgpack::encodeInterned;
        |using msgpack::MsgpackReader;
        |using msgpack::MsgpackWriter;
        |${json.flatMap(_.impls).mkString("\n")}
        |json hashed_from_json(const json& j_) {
        |  auto hash_ = j_.at(0).get<std::string>();
        |  auto data_ = j_.at(1);
        |  if(hash_ != AdtHash) {
        |   throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
        |  }
        |  return data_;
        |}
        |
        |json hashed_to_json(const json& x_) {
        |  return json::array({AdtHash, x_});
        |}
        |
        |$msgpackForwardDecls
        |$msgpackImpls
        |$msgpackRoots
        |} // namespace $namespace
        |""".stripMargin
  }
}
