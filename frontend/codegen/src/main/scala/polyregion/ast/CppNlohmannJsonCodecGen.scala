package polyregion.ast

import polyregion.ast.CppStructGen.*
import cats.syntax.all.*

private[polyregion] case class CppNlohmannJsonCodecGen(
    namespace: List[String],
    decls: List[String],
    impls: List[String],
    msgpackForwardDecls: List[String],
    msgpackImpls: List[String]
)
private[polyregion] object CppNlohmannJsonCodecGen {

  def fromJsonFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_from_json"
  def toJsonFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_to_json"
  def jsonAt(idx: Int)       = s"j_.at($idx)"

  def fromMsgpackFn(t: CppType)       = t.ref(qualified = false).toLowerCase + "_from_msgpack"
  def toMsgpackFn(t: CppType)         = t.ref(qualified = false).toLowerCase + "_to_msgpack"
  def fromMsgpackFieldsFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_fields_from_msgpack"
  def toMsgpackFieldsFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_fields_to_msgpack"

  private def isInt32Like(tpe: CppType) =
    tpe.kind == CppType.Kind.StdLib &&
      tpe.namespace.isEmpty &&
      Set("int8_t", "uint16_t", "int16_t", "int32_t").contains(tpe.name)

  private def readMsgpackValue(tpe: CppType, name: String): List[String] =
    tpe.kind match {
      case CppType.Kind.StdLib =>
        (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
          case ("std" :: "optional" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              s"if(!r_.tryReadNil()) {" ::
              readMsgpackValue(x, s"${name}_value").map("  " + _) :::
              s"  $name = std::move(${name}_value);" ::
              s"}" :: Nil
          case ("std" :: "vector" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              "{" ::
              s"  auto ${name}_size = r_.readArrayHeader();" ::
              s"  $name.reserve(${name}_size);" ::
              s"  for(size_t ${name}_idx = 0; ${name}_idx < ${name}_size; ++${name}_idx) {" ::
              readMsgpackValue(x, s"${name}_elem").map("    " + _) :::
              s"    ${name}.emplace_back(std::move(${name}_elem));" ::
              "  }" ::
              "}" :: Nil
          case ("std" :: "set" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              "{" ::
              s"  auto ${name}_size = r_.readArrayHeader();" ::
              s"  for(size_t ${name}_idx = 0; ${name}_idx < ${name}_size; ++${name}_idx) {" ::
              readMsgpackValue(x, s"${name}_elem").map("    " + _) :::
              s"    ${name}.emplace(std::move(${name}_elem));" ::
              "  }" ::
              "}" :: Nil
          case ("std" :: "string" :: Nil, Nil) => s"auto $name = r_.readString();" :: Nil
          case ("bool" :: Nil, Nil)            => s"auto $name = r_.readBoolean();" :: Nil
          case ("float" :: Nil, Nil)           => s"auto $name = r_.readFloat32();" :: Nil
          case ("double" :: Nil, Nil)          => s"auto $name = r_.readFloat64();" :: Nil
          case ("int64_t" :: Nil, Nil)         => s"auto $name = r_.readInt64();" :: Nil
          case _ if isInt32Like(tpe) =>
            s"auto $name = static_cast<${tpe.ref(qualified = true)}>(r_.readInt32());" :: Nil
          case _ => s"auto $name = r_.readInt32();" :: Nil
        }
      case _ => s"auto $name = ${tpe.ns(fromMsgpackFn(tpe))}(r_);" :: Nil
    }

  private def writeMsgpackValue(tpe: CppType, value: String, depth: Int = 0): List[String] =
    tpe.kind match {
      case CppType.Kind.StdLib =>
        (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
          case ("std" :: "optional" :: Nil, x :: Nil) =>
            s"if($value) {" ::
              writeMsgpackValue(x, s"(*$value)", depth + 1).map("  " + _) :::
              "} else {" ::
              "  w_.writeNil();" ::
              "}" :: Nil
          case ("std" :: ("vector" | "set") :: Nil, x :: Nil) =>
            val elem = s"v${depth}_"
            s"w_.writeArrayHeader($value.size());" ::
              s"for(const auto &$elem : $value) {" ::
              writeMsgpackValue(x, elem, depth + 1).map("  " + _) :::
              "}" :: Nil
          case ("std" :: "string" :: Nil, Nil) => s"w_.writeString($value);" :: Nil
          case ("bool" :: Nil, Nil)            => s"w_.writeBoolean($value);" :: Nil
          case ("float" :: Nil, Nil)           => s"w_.writeFloat32($value);" :: Nil
          case ("double" :: Nil, Nil)          => s"w_.writeFloat64($value);" :: Nil
          case ("int64_t" :: Nil, Nil)         => s"w_.writeInt64($value);" :: Nil
          case _                               => s"w_.writeInt32(static_cast<int32_t>($value));" :: Nil
        }
      case _ => s"${tpe.ns(toMsgpackFn(tpe))}(w_, $value);" :: Nil
    }

  private def readMsgpackFieldsBody(s: StructNode, countExpr: String): List[String] = {
    val ctorInvocation = s.members match {
      case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)"
      case _                => s.members.map(_._1).mkString("{", ", ", "}")
    }
    s"if($countExpr != ${s.members.size}) throw std::runtime_error(\"Expected ${s.tpe.ref(qualified = true)} with ${s.members.size} field(s)\");" ::
      s.members.flatMap { case (name, tpe) => readMsgpackValue(tpe, name) } :::
      s"return ${ctorInvocation};" ::
      Nil
  }

  private def writeMsgpackFieldsBody(s: StructNode): List[String] =
    s.members.flatMap { case (name, tpe) => writeMsgpackValue(tpe, s"x_.$name") }

  private def msgpackForwardDecls(s: StructNode): List[String] =
    if (s.tpe.kind == CppType.Kind.Base) {
      s"${s.tpe.ref(qualified = true)} ${fromMsgpackFn(s.tpe)}(MsgpackReader &);" ::
        s"void ${toMsgpackFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" ::
        Nil
    } else {
      s"${s.tpe.ref(qualified = true)} ${fromMsgpackFieldsFn(s.tpe)}(MsgpackReader &, size_t);" ::
        s"void ${toMsgpackFieldsFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" ::
        s"${s.tpe.ref(qualified = true)} ${fromMsgpackFn(s.tpe)}(MsgpackReader &);" ::
        s"void ${toMsgpackFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" ::
        Nil
    }

  private def msgpackImpls(s: StructNode): List[String] =
    if (s.tpe.kind == CppType.Kind.Base) {
      val fromCases = s.variants.zipWithIndex.flatMap { case (c, i) =>
        s"case $i: return ${c.tpe.ns(fromMsgpackFieldsFn(c.tpe))}(r_, n_ - 1);" :: Nil
      }
      val fromNullaryCases = s.variants.zipWithIndex.flatMap { case (c, i) =>
        if (c.members.isEmpty) s"case $i: return ${c.tpe.ns(fromMsgpackFieldsFn(c.tpe))}(r_, 0);" :: Nil
        else s"case $i: throw std::runtime_error(\"Expected array payload for non-nullary sum ordinal\");" :: Nil
      }
      val toCases =
        s.variants.zipWithIndex
          .map { case (c, i) =>
            val body =
              if (c.members.isEmpty) s"w_.writeInt32($i);" :: Nil
              else
                s"w_.writeArrayHeader(${c.members.size + 1});" ::
                  s"w_.writeInt32($i);" ::
                  s"${c.tpe.ns(toMsgpackFieldsFn(c.tpe))}(w_, y_);" :: Nil
            s"[&](const ${c.tpe.ref(qualified = true)} &y_) -> void {" :: body.map("  " + _) ::: "}" :: Nil
          }
          .intercalate("," :: Nil)
      "" ::
        s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFn(s.tpe))}(MsgpackReader& r_) {" ::
        "  if(r_.nextIsArray()) {" ::
        "    auto n_ = r_.readArrayHeader();" ::
        "    if(n_ == 0) throw std::runtime_error(\"Expected non-empty sum payload\");" ::
        "    auto ord_ = r_.readInt32();" ::
        "    switch(ord_) {" ::
        fromCases.map("      " + _) :::
        "      default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
        "    }" ::
        "  } else {" ::
        "    auto ord_ = r_.readInt32();" ::
        "    switch(ord_) {" ::
        fromNullaryCases.map("      " + _) :::
        "      default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
        "    }" ::
        "  }" ::
        "}" ::
        "" ::
        s"void ${s.tpe.ns(toMsgpackFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
        "  x_.match_total(" ::
        toCases.map("    " + _) :::
        "  );" ::
        "}" ::
        Nil
    } else {
      "" ::
        s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFieldsFn(s.tpe))}(MsgpackReader& r_, size_t n_) {" ::
        readMsgpackFieldsBody(s, "n_").map("  " + _) :::
        "}" ::
        "" ::
        s"void ${s.tpe.ns(toMsgpackFieldsFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
        writeMsgpackFieldsBody(s).map("  " + _) :::
        "}" ::
        "" ::
        s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFn(s.tpe))}(MsgpackReader& r_) {" ::
        "  auto n_ = r_.readArrayHeader();" ::
        s"  return ${s.tpe.ns(fromMsgpackFieldsFn(s.tpe))}(r_, n_);" ::
        "}" ::
        "" ::
        s"void ${s.tpe.ns(toMsgpackFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
        s"  w_.writeArrayHeader(${s.members.size});" ::
        s"  ${s.tpe.ns(toMsgpackFieldsFn(s.tpe))}(w_, x_);" ::
        "}" ::
        Nil
    }

  def fromJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    s"size_t ord_ = ${jsonAt(0)}.get<size_t>();" ::
      s"const auto &t_ = ${jsonAt(1)};" ::
      "switch (ord_) {" ::
      s.variants.zipWithIndex.map((c, i) => s"case ${i}: return ${c.tpe.ns(fromJsonFn(c.tpe))}(t_);") :::
      s"default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
      "}" :: Nil
  } else {

    val ctorInvocation = s.members match {
      case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)"
      case _                => s.members.map(_._1).mkString("{", ", ", "}")
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
            case ("std" :: (c @ ("vector" | "set")) :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
              s"${tpe.ref(qualified = true)} $name;" ::
                s"for(const auto &v_ : ${jsonAt(idx)}) { ${name}.${c match {
                    case "vector" => "emplace_back"
                    case "set"    => "emplace"
                  }}(${x.ns(fromJsonFn(x))}(v_)); }"
                :: Nil
            case _ => s"auto $name = ${jsonAt(idx)}.get<${tpe.ref(qualified = true)}>();" :: Nil
          }
        case _ => s"auto $name = ${tpe.ns(fromJsonFn(tpe))}(${jsonAt(idx)});" :: Nil
      }
    } :::
      s"return ${ctorInvocation};" ::
      Nil
  }

  def toJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    "return x_.match_total(" ::
      s.variants.zipWithIndex
        .map((c, i) =>
          s"[](const ${c.tpe.ref(qualified = true)} &y_) -> json { return {$i, ${c.tpe.ns(toJsonFn(c.tpe))}(y_)}; }" :: Nil
        )
        .intercalate("," :: Nil) :::
      ");" :: Nil
  } else {
    s.members.flatMap { case (name, tpe) =>
      tpe.kind match {
        case CppType.Kind.StdLib =>
          (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case ("std" :: "optional" :: Nil, x :: Nil) =>
              val deref =
                if (x.kind != CppType.Kind.StdLib) s"${x.ns(toJsonFn(x))}(*x_.${name})"
                else s"json(*x_.${name})"
              s"auto $name = x_.${name} ? $deref : json();" :: Nil

            case ("std" :: ("vector" | "set") :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
              s"std::vector<json> $name;" ::
                s"for(const auto &v_ : x_.${name}) { ${name}.emplace_back(${x.ns(toJsonFn(x))}(v_)); }"
                :: Nil
            case _ => s"auto $name = x_.${name};" :: Nil
          }
        case _ => s"auto $name = ${tpe.ns(toJsonFn(tpe))}(x_.${name});" :: Nil
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
      s"[[nodiscard]] POLYREGION_EXPORT ${s.tpe.ref(qualified = true)} ${fromJsonFn(s.tpe)}(const json &);" ::
        s"[[nodiscard]] POLYREGION_EXPORT json ${toJsonFn(s.tpe)}(const ${s.tpe.ref(qualified = true)} &);" ::
        Nil

    s.variants.flatMap(s => emit(s)) :+
      CppNlohmannJsonCodecGen(
        s.tpe.namespace,
        decls,
        fromJsonImpl ::: toJsonImpl,
        msgpackForwardDecls(s),
        msgpackImpls(s)
      )
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
          |#include <cstdint>
          |#include <vector>
          |
          |#include "nlohmann/json.hpp"
          |#include "polyast.h"
          |#include "polyregion/export.h"
          |
          |using json = nlohmann::json;
          |
          |namespace ${namespace} { 
          |${lines.mkString("\n")}
          |[[nodiscard]] POLYREGION_EXPORT json hashed_to_json(const json&);
          |[[nodiscard]] POLYREGION_EXPORT json hashed_from_json(const json&);
          |
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> program_to_msgpack(const Program&);
          |[[nodiscard]] POLYREGION_EXPORT Program program_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT Program program_from_msgpack(const std::vector<uint8_t>&);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> hashed_program_to_msgpack(const Program&);
          |[[nodiscard]] POLYREGION_EXPORT Program hashed_program_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT Program hashed_program_from_msgpack(const std::vector<uint8_t>&);
          |
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> structdefs_to_msgpack(const std::vector<StructDef>&);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<StructDef> structdefs_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<StructDef> structdefs_from_msgpack(const std::vector<uint8_t>&);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> hashed_structdefs_to_msgpack(const std::vector<StructDef>&);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<StructDef> hashed_structdefs_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT std::vector<StructDef> hashed_structdefs_from_msgpack(const std::vector<uint8_t>&);
          |
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> compileresult_to_msgpack(const CompileResult&);
          |[[nodiscard]] POLYREGION_EXPORT CompileResult compileresult_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT CompileResult compileresult_from_msgpack(const std::vector<uint8_t>&);
          |
          |[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> passrunresult_to_msgpack(const PassRunResult&);
          |[[nodiscard]] POLYREGION_EXPORT PassRunResult passrunresult_from_msgpack(const uint8_t*, const uint8_t*);
          |[[nodiscard]] POLYREGION_EXPORT PassRunResult passrunresult_from_msgpack(const std::vector<uint8_t>&);
          |} // namespace $namespace
          |
          |""".stripMargin
  }

  def emitImpl(namespace: String, headerName: String, hash: String, xs: List[CppNlohmannJsonCodecGen]) = {
    val msgpackForwardDecls = xs
      .groupMapReduce(_.namespace.mkString("::"))(_.msgpackForwardDecls)(_ ::: _)
      .toList
      .flatMap {
        case ("", lines) => lines
        case (ns, lines) => s"namespace $ns {" :: lines ::: s"} // namespace $ns" :: Nil
      }
      .mkString("\n")
    val msgpackImpls = xs.flatMap(_.msgpackImpls).mkString("\n")

    val msgpackHelpers =
      s"""|#include <cmath>
          |#include <cstring>
          |#include <limits>
          |#include <stdexcept>
          |#include <string>
          |#include <unordered_map>
          |#include <utility>
          |
          |constexpr auto AdtHash = "$hash";
          |
          |namespace {
          |
          |constexpr int32_t MsgpackInternedMagic = 0x4d504349; // "MPCI"
          |
          |class StringInterner {
          |  std::unordered_map<std::string, int32_t> ids_;
          |  std::vector<std::string> entries_;
          |
          |public:
          |  int32_t id(const std::string &x) {
          |    if(auto it = ids_.find(x); it != ids_.end()) return it->second;
          |    if(entries_.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
          |      throw std::runtime_error("String table too large");
          |    auto next = static_cast<int32_t>(entries_.size());
          |    entries_.push_back(x);
          |    ids_.emplace(entries_.back(), next);
          |    return next;
          |  }
          |
          |  [[nodiscard]] const std::vector<std::string> &entries() const { return entries_; }
          |};
          |
          |class MsgpackWriter {
          |  std::vector<uint8_t> bytes_;
          |  StringInterner *interner_ = nullptr;
          |  bool collectOnly_ = false;
          |
          |  void byte(uint8_t x) {
          |    if(!collectOnly_) bytes_.push_back(x);
          |  }
          |
          |  void raw16(uint16_t x) {
          |    byte(static_cast<uint8_t>(x >> 8));
          |    byte(static_cast<uint8_t>(x));
          |  }
          |
          |  void raw32(uint32_t x) {
          |    byte(static_cast<uint8_t>(x >> 24));
          |    byte(static_cast<uint8_t>(x >> 16));
          |    byte(static_cast<uint8_t>(x >> 8));
          |    byte(static_cast<uint8_t>(x));
          |  }
          |
          |  void raw64(uint64_t x) {
          |    byte(static_cast<uint8_t>(x >> 56));
          |    byte(static_cast<uint8_t>(x >> 48));
          |    byte(static_cast<uint8_t>(x >> 40));
          |    byte(static_cast<uint8_t>(x >> 32));
          |    byte(static_cast<uint8_t>(x >> 24));
          |    byte(static_cast<uint8_t>(x >> 16));
          |    byte(static_cast<uint8_t>(x >> 8));
          |    byte(static_cast<uint8_t>(x));
          |  }
          |
          |public:
          |  explicit MsgpackWriter(size_t initialSize = 256, StringInterner *interner = nullptr, bool collectOnly = false)
          |      : interner_(interner), collectOnly_(collectOnly) {
          |    if(!collectOnly_) bytes_.reserve(initialSize);
          |  }
          |
          |  void setStringInterner(StringInterner *interner) { interner_ = interner; }
          |  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }
          |
          |  void writeNil() { byte(0xc0); }
          |  void writeBoolean(bool x) { byte(x ? 0xc3 : 0xc2); }
          |
          |  void writeInt32(int32_t x) {
          |    if(x >= 0 && x <= 0x7f) byte(static_cast<uint8_t>(x));
          |    else if(x >= -32 && x < 0) byte(static_cast<uint8_t>(x));
          |    else if(x >= std::numeric_limits<int8_t>::min() && x <= std::numeric_limits<int8_t>::max()) {
          |      byte(0xd0);
          |      byte(static_cast<uint8_t>(x));
          |    } else if(x >= std::numeric_limits<int16_t>::min() && x <= std::numeric_limits<int16_t>::max()) {
          |      byte(0xd1);
          |      raw16(static_cast<uint16_t>(x));
          |    } else {
          |      byte(0xd2);
          |      raw32(static_cast<uint32_t>(x));
          |    }
          |  }
          |
          |  void writeInt64(int64_t x) {
          |    if(x >= std::numeric_limits<int32_t>::min() && x <= std::numeric_limits<int32_t>::max()) writeInt32(static_cast<int32_t>(x));
          |    else {
          |      byte(0xd3);
          |      raw64(static_cast<uint64_t>(x));
          |    }
          |  }
          |
          |  void writeFloat32(float x) {
          |    uint32_t bits;
          |    std::memcpy(&bits, &x, sizeof(bits));
          |    byte(0xca);
          |    raw32(bits);
          |  }
          |
          |  void writeFloat64(double x) {
          |    uint64_t bits;
          |    std::memcpy(&bits, &x, sizeof(bits));
          |    byte(0xcb);
          |    raw64(bits);
          |  }
          |
          |  void writeString(const std::string &x) {
          |    if(interner_) {
          |      const auto n = interner_->id(x);
          |      if(!collectOnly_) writeInt32(n);
          |    } else writeStringLiteral(x);
          |  }
          |
          |  void writeStringLiteral(const std::string &x) {
          |    const auto n = x.size();
          |    if(n <= 31) byte(static_cast<uint8_t>(0xa0 | n));
          |    else if(n <= 0xff) {
          |      byte(0xd9);
          |      byte(static_cast<uint8_t>(n));
          |    } else if(n <= 0xffff) {
          |      byte(0xda);
          |      raw16(static_cast<uint16_t>(n));
          |    } else if(n <= std::numeric_limits<uint32_t>::max()) {
          |      byte(0xdb);
          |      raw32(static_cast<uint32_t>(n));
          |    } else throw std::runtime_error("String too large");
          |    if(!collectOnly_) bytes_.insert(bytes_.end(), x.begin(), x.end());
          |  }
          |
          |  void writeArrayHeader(size_t n) {
          |    if(n <= 15) byte(static_cast<uint8_t>(0x90 | n));
          |    else if(n <= 0xffff) {
          |      byte(0xdc);
          |      raw16(static_cast<uint16_t>(n));
          |    } else if(n <= std::numeric_limits<uint32_t>::max()) {
          |      byte(0xdd);
          |      raw32(static_cast<uint32_t>(n));
          |    } else throw std::runtime_error("Array too large");
          |  }
          |};
          |
          |class MsgpackReader {
          |  const uint8_t *begin_;
          |  const uint8_t *cursor_;
          |  const uint8_t *end_;
          |  const std::vector<std::string> *stringTable_ = nullptr;
          |
          |  [[noreturn]] void fail(const std::string &message) const {
          |    throw std::runtime_error(message + " at byte " + std::to_string(offset()));
          |  }
          |
          |  void require(size_t n) const {
          |    if(static_cast<size_t>(end_ - cursor_) < n) fail("Unexpected end of input");
          |  }
          |
          |  uint8_t u8() {
          |    require(1);
          |    return *cursor_++;
          |  }
          |
          |  int8_t i8() { return static_cast<int8_t>(u8()); }
          |
          |  uint16_t u16() {
          |    require(2);
          |    uint16_t x = (static_cast<uint16_t>(cursor_[0]) << 8) | static_cast<uint16_t>(cursor_[1]);
          |    cursor_ += 2;
          |    return x;
          |  }
          |
          |  int16_t i16() { return static_cast<int16_t>(u16()); }
          |
          |  uint32_t u32() {
          |    require(4);
          |    uint32_t x = (static_cast<uint32_t>(cursor_[0]) << 24) | (static_cast<uint32_t>(cursor_[1]) << 16) |
          |                 (static_cast<uint32_t>(cursor_[2]) << 8) | static_cast<uint32_t>(cursor_[3]);
          |    cursor_ += 4;
          |    return x;
          |  }
          |
          |  int32_t i32() { return static_cast<int32_t>(u32()); }
          |
          |  uint64_t u64() {
          |    require(8);
          |    uint64_t x = (static_cast<uint64_t>(cursor_[0]) << 56) | (static_cast<uint64_t>(cursor_[1]) << 48) |
          |                 (static_cast<uint64_t>(cursor_[2]) << 40) | (static_cast<uint64_t>(cursor_[3]) << 32) |
          |                 (static_cast<uint64_t>(cursor_[4]) << 24) | (static_cast<uint64_t>(cursor_[5]) << 16) |
          |                 (static_cast<uint64_t>(cursor_[6]) << 8) | static_cast<uint64_t>(cursor_[7]);
          |    cursor_ += 8;
          |    return x;
          |  }
          |
          |  int64_t i64() { return static_cast<int64_t>(u64()); }
          |
          |  int64_t readIntegralLong() {
          |    const auto m = u8();
          |    if(m <= 0x7f) return static_cast<int64_t>(m);
          |    if(m >= 0xe0) return static_cast<int64_t>(static_cast<int8_t>(m));
          |    switch(m) {
          |      case 0xcc: return static_cast<int64_t>(u8());
          |      case 0xcd: return static_cast<int64_t>(u16());
          |      case 0xce: return static_cast<int64_t>(u32());
          |      case 0xcf: {
          |        const auto x = u64();
          |        if(x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail("uint64 value exceeds int64_t");
          |        return static_cast<int64_t>(x);
          |      }
          |      case 0xd0: return static_cast<int64_t>(i8());
          |      case 0xd1: return static_cast<int64_t>(i16());
          |      case 0xd2: return static_cast<int64_t>(i32());
          |      case 0xd3: return i64();
          |      default: fail("Expected integer");
          |    }
          |  }
          |
          |public:
          |  MsgpackReader(const uint8_t *begin, const uint8_t *end) : begin_(begin), cursor_(begin), end_(end) {}
          |
          |  [[nodiscard]] size_t offset() const { return static_cast<size_t>(cursor_ - begin_); }
          |  [[nodiscard]] bool isAtEnd() const { return cursor_ == end_; }
          |  void setStringTable(const std::vector<std::string> *table) { stringTable_ = table; }
          |
          |  [[nodiscard]] bool nextIsArray() const {
          |    if(cursor_ >= end_) return false;
          |    const auto m = *cursor_;
          |    return (m & 0xf0) == 0x90 || m == 0xdc || m == 0xdd;
          |  }
          |
          |  void readNil() {
          |    if(u8() != 0xc0) fail("Expected nil");
          |  }
          |
          |  bool tryReadNil() {
          |    if(cursor_ < end_ && *cursor_ == 0xc0) {
          |      ++cursor_;
          |      return true;
          |    }
          |    return false;
          |  }
          |
          |  bool readBoolean() {
          |    switch(u8()) {
          |      case 0xc2: return false;
          |      case 0xc3: return true;
          |      default: fail("Expected boolean");
          |    }
          |  }
          |
          |  int32_t readInt32() {
          |    const auto x = readIntegralLong();
          |    if(x < std::numeric_limits<int32_t>::min() || x > std::numeric_limits<int32_t>::max()) fail("Integer out of int32_t range");
          |    return static_cast<int32_t>(x);
          |  }
          |
          |  int64_t readInt64() { return readIntegralLong(); }
          |
          |  float readFloat32() {
          |    switch(u8()) {
          |      case 0xca: {
          |        const auto bits = u32();
          |        float out;
          |        std::memcpy(&out, &bits, sizeof(out));
          |        return out;
          |      }
          |      case 0xcb: {
          |        const auto bits = u64();
          |        double d;
          |        std::memcpy(&d, &bits, sizeof(d));
          |        const auto f = static_cast<float>(d);
          |        if(std::isnan(d) || static_cast<double>(f) == d) return f;
          |        fail("Float64 to Float32 conversion with loss of precision");
          |      }
          |      default: fail("Expected float32/float64");
          |    }
          |  }
          |
          |  double readFloat64() {
          |    if(u8() != 0xcb) fail("Expected float64");
          |    const auto bits = u64();
          |    double out;
          |    std::memcpy(&out, &bits, sizeof(out));
          |    return out;
          |  }
          |
          |  std::string readString() {
          |    if(!stringTable_) return readStringLiteral();
          |    const auto id = readInt32();
          |    if(id < 0 || static_cast<size_t>(id) >= stringTable_->size()) fail("Bad string table id");
          |    return stringTable_->at(static_cast<size_t>(id));
          |  }
          |
          |  std::string readStringLiteral() {
          |    const auto m = u8();
          |    size_t n;
          |    if((m & 0xe0) == 0xa0) n = m & 0x1f;
          |    else {
          |      switch(m) {
          |        case 0xd9: n = u8(); break;
          |        case 0xda: n = u16(); break;
          |        case 0xdb: n = u32(); break;
          |        default: fail("Expected string");
          |      }
          |    }
          |    require(n);
          |    std::string out(reinterpret_cast<const char *>(cursor_), n);
          |    cursor_ += n;
          |    return out;
          |  }
          |
          |  size_t readArrayHeader() {
          |    const auto m = u8();
          |    if((m & 0xf0) == 0x90) return m & 0x0f;
          |    switch(m) {
          |      case 0xdc: return u16();
          |      case 0xdd: return u32();
          |      default: fail("Expected array");
          |    }
          |  }
          |};
          |
          |bool isInternedEnvelope(const uint8_t *begin, const uint8_t *end) {
          |  return end - begin >= 6 && begin[0] == 0x93 && begin[1] == 0xd2 && begin[2] == 0x4d && begin[3] == 0x50 &&
          |         begin[4] == 0x43 && begin[5] == 0x49;
          |}
          |
          |template <typename F> std::vector<uint8_t> encodeInterned(F &&writeValue) {
          |  StringInterner table;
          |  MsgpackWriter collect(16, &table, true);
          |  writeValue(collect);
          |
          |  MsgpackWriter w;
          |  w.writeArrayHeader(3);
          |  w.writeInt32(MsgpackInternedMagic);
          |  w.writeArrayHeader(table.entries().size());
          |  for(const auto &s : table.entries()) w.writeStringLiteral(s);
          |  w.setStringInterner(&table);
          |  writeValue(w);
          |  return w.take();
          |}
          |
          |template <typename F> auto decodeMaybeInterned(const uint8_t *begin, const uint8_t *end, F &&readValue)
          |    -> decltype(readValue(std::declval<MsgpackReader &>())) {
          |  MsgpackReader r(begin, end);
          |  if(isInternedEnvelope(begin, end)) {
          |    const auto n = r.readArrayHeader();
          |    if(n != 3) throw std::runtime_error("Expected interned envelope array of size 3");
          |    const auto magic = r.readInt32();
          |    if(magic != MsgpackInternedMagic) throw std::runtime_error("Bad interned envelope magic");
          |    const auto tableSize = r.readArrayHeader();
          |    std::vector<std::string> table;
          |    table.reserve(tableSize);
          |    for(size_t i = 0; i < tableSize; ++i) table.emplace_back(r.readStringLiteral());
          |    r.setStringTable(&table);
          |    auto out = readValue(r);
          |    if(!r.isAtEnd()) throw std::runtime_error("Trailing bytes after MessagePack value");
          |    return out;
          |  }
          |  auto out = readValue(r);
          |  if(!r.isAtEnd()) throw std::runtime_error("Trailing bytes after MessagePack value");
          |  return out;
          |}
          |
          |} // namespace
          |""".stripMargin

    val msgpackTopLevel =
      s"""|
          |static void structdefs_value_to_msgpack(MsgpackWriter& w_, const std::vector<StructDef>& xs_) {
          |  w_.writeArrayHeader(xs_.size());
          |  for(const auto& x_ : xs_) structdef_to_msgpack(w_, x_);
          |}
          |
          |static std::vector<StructDef> structdefs_value_from_msgpack(MsgpackReader& r_) {
          |  auto n_ = r_.readArrayHeader();
          |  std::vector<StructDef> xs_;
          |  xs_.reserve(n_);
          |  for(size_t i_ = 0; i_ < n_; ++i_) xs_.emplace_back(structdef_from_msgpack(r_));
          |  return xs_;
          |}
          |
          |std::vector<uint8_t> program_to_msgpack(const Program& x_) {
          |  return encodeInterned([&](MsgpackWriter& w_) { program_to_msgpack(w_, x_); });
          |}
          |
          |Program program_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) { return program_from_msgpack(r_); });
          |}
          |
          |Program program_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return program_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |
          |std::vector<uint8_t> hashed_program_to_msgpack(const Program& x_) {
          |  return encodeInterned([&](MsgpackWriter& w_) {
          |    w_.writeArrayHeader(2);
          |    w_.writeString(std::string(AdtHash));
          |    program_to_msgpack(w_, x_);
          |  });
          |}
          |
          |Program hashed_program_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) {
          |    auto n_ = r_.readArrayHeader();
          |    if(n_ != 2) throw std::runtime_error("Expected versioned Program array of size 2");
          |    auto hash_ = r_.readString();
          |    if(hash_ != AdtHash) throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
          |    return program_from_msgpack(r_);
          |  });
          |}
          |
          |Program hashed_program_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return hashed_program_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |
          |std::vector<uint8_t> structdefs_to_msgpack(const std::vector<StructDef>& xs_) {
          |  return encodeInterned([&](MsgpackWriter& w_) { structdefs_value_to_msgpack(w_, xs_); });
          |}
          |
          |std::vector<StructDef> structdefs_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) { return structdefs_value_from_msgpack(r_); });
          |}
          |
          |std::vector<StructDef> structdefs_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |
          |std::vector<uint8_t> hashed_structdefs_to_msgpack(const std::vector<StructDef>& xs_) {
          |  return encodeInterned([&](MsgpackWriter& w_) {
          |    w_.writeArrayHeader(2);
          |    w_.writeString(std::string(AdtHash));
          |    structdefs_value_to_msgpack(w_, xs_);
          |  });
          |}
          |
          |std::vector<StructDef> hashed_structdefs_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) {
          |    auto n_ = r_.readArrayHeader();
          |    if(n_ != 2) throw std::runtime_error("Expected versioned StructDef list array of size 2");
          |    auto hash_ = r_.readString();
          |    if(hash_ != AdtHash) throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
          |    return structdefs_value_from_msgpack(r_);
          |  });
          |}
          |
          |std::vector<StructDef> hashed_structdefs_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return hashed_structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |
          |std::vector<uint8_t> compileresult_to_msgpack(const CompileResult& x_) {
          |  return encodeInterned([&](MsgpackWriter& w_) { compileresult_to_msgpack(w_, x_); });
          |}
          |
          |CompileResult compileresult_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) { return compileresult_from_msgpack(r_); });
          |}
          |
          |CompileResult compileresult_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return compileresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |
          |std::vector<uint8_t> passrunresult_to_msgpack(const PassRunResult& x_) {
          |  return encodeInterned([&](MsgpackWriter& w_) { passrunresult_to_msgpack(w_, x_); });
          |}
          |
          |PassRunResult passrunresult_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {
          |  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) { return passrunresult_from_msgpack(r_); });
          |}
          |
          |PassRunResult passrunresult_from_msgpack(const std::vector<uint8_t>& xs_) {
          |  return passrunresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
          |}
          |""".stripMargin

    s"""|#include "$headerName.h"
          |$msgpackHelpers
          |
          |template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
          |template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
          |
          |namespace $namespace {
          |${xs.flatMap(_.impls).mkString("\n")}
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
          |$msgpackTopLevel
          |} // namespace $namespace
          |""".stripMargin
  }

}
