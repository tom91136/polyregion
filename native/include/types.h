#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "json.hpp"

namespace polyregion::runtime {

enum class POLYREGION_EXPORT Type : uint8_t {
  Void = 1,
  Bool8,
  Byte8,
  CharU16,
  Short16,
  Int32,
  Long64,
  Float32,
  Double64,
  Ptr,
  Scratch,
};

static constexpr POLYREGION_EXPORT size_t byteOfType(Type t) {
  switch (t) {
    case Type::Void: return 0;
    case Type::Bool8:
    case Type::Byte8: return 8 / 8;
    case Type::CharU16:
    case Type::Short16: return 16 / 8;
    case Type::Int32: return 32 / 8;
    case Type::Long64: return 64 / 8;
    case Type::Float32: return 32 / 8;
    case Type::Double64: return 64 / 8;
    case Type::Scratch:
    case Type::Ptr: return sizeof(void *);
  }
}

static constexpr POLYREGION_EXPORT const char *typeName(Type t) {
  switch (t) {
    case Type::Void: return "Void";
    case Type::Bool8: return "Bool8";
    case Type::Byte8: return "Byte8";
    case Type::CharU16: return "Char";
    case Type::Short16: return "Short16";
    case Type::Int32: return "Int32";
    case Type::Long64: return "Long64";
    case Type::Float32: return "Float32";
    case Type::Double64: return "Double64";
    case Type::Ptr: return "Ptr";
    case Type::Scratch: return "Scratch";
  }
}

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime

namespace polyregion::runtime {

enum class POLYREGION_EXPORT PlatformKind : uint8_t { HostThreaded = 1, Managed };
static constexpr std::string_view POLYREGION_EXPORT to_string(const PlatformKind &b) {
  switch (b) {
    case PlatformKind::HostThreaded: return "HostThreaded";
    case PlatformKind::Managed: return "Managed";
  }
}
static constexpr std::optional<PlatformKind> POLYREGION_EXPORT parsePlatformKind(std::string_view name) {
  if (name == "host" || name == "hostthreaded") return PlatformKind::HostThreaded;
  if (name == "managed") return PlatformKind::Managed;
  return {};
}

enum class POLYREGION_EXPORT ModuleFormat : uint8_t { Source = 1, Object, DSO, PTX, HSACO, SPIRV };
static constexpr std::string_view POLYREGION_EXPORT to_string(const ModuleFormat &x) {
  switch (x) {
    case ModuleFormat::Source: return "Source";
    case ModuleFormat::Object: return "Object";
    case ModuleFormat::DSO: return "DSO";
    case ModuleFormat::PTX: return "PTX";
    case ModuleFormat::HSACO: return "HSACO";
    case ModuleFormat::SPIRV: return "SPIRV";
  }
}
static constexpr std::optional<ModuleFormat> POLYREGION_EXPORT parseModuleFormat(std::string_view name) {
  if (name == "src" || name == "source") return ModuleFormat::Source;
  if (name == "obj" || name == "object") return ModuleFormat::Object;
  if (name == "dso") return ModuleFormat::DSO;
  if (name == "ptx") return ModuleFormat::PTX;
  if (name == "hsaco") return ModuleFormat::HSACO;
  if (name == "spirv") return ModuleFormat::SPIRV;
  return {};
}

}; // namespace polyregion::runtime

NLOHMANN_JSON_NAMESPACE_BEGIN
using namespace polyregion::runtime;
template <> struct adl_serializer<ModuleFormat> {
  static void to_json(json &j, const ModuleFormat &value) {
    switch (value) {
      case ModuleFormat::Source: j = "source"; break;
      case ModuleFormat::Object: j = "object"; break;
      case ModuleFormat::DSO: j = "dso"; break;
      case ModuleFormat::PTX: j = "ptx"; break;
      case ModuleFormat::HSACO: j = "hsaco"; break;
      case ModuleFormat::SPIRV: j = "spirv"; break;
      default: j = "unknown"; break;
    }
  }

  static void from_json(const json &j, ModuleFormat &value) {
    std::string s = j.get<std::string>();
    if (s == "source") value = ModuleFormat::Source;
    else if (s == "object") value = ModuleFormat::Object;
    else if (s == "dso") value = ModuleFormat::DSO;
    else if (s == "ptx") value = ModuleFormat::PTX;
    else if (s == "hsaco") value = ModuleFormat::HSACO;
    else if (s == "spirv") value = ModuleFormat::SPIRV;
    else throw std::logic_error("Unexpected module format:" + s);
  }
};

template <> struct adl_serializer<PlatformKind> {
  static void to_json(json &j, const PlatformKind &value) {
    switch (value) {
      case PlatformKind::HostThreaded: j = "hostthreaded"; break;
      case PlatformKind::Managed: j = "managed"; break;
    }
  }

  static void from_json(const json &j, PlatformKind &value) {
    std::string s = j.get<std::string>();
    if (s == "hostthreaded") value = PlatformKind::HostThreaded;
    else if (s == "managed") value = PlatformKind::Managed;
    else throw std::logic_error("Unexpected kind:" + s);
  }
};
NLOHMANN_JSON_NAMESPACE_END

namespace polyregion::runtime {

struct POLYREGION_EXPORT KernelObject {
  ModuleFormat format{};
  PlatformKind kind{};
  std::vector<std::string> features{};
  std::string moduleName{};
  std::string moduleImage{};
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(KernelObject, format, kind, features, moduleName, moduleImage);
};

struct POLYREGION_EXPORT KernelBundle {
  std::vector<KernelObject> objects{};
  nlohmann::json metadata{};
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(KernelBundle, objects, metadata);
  static POLYREGION_EXPORT KernelBundle fromMsgPack(size_t size, const unsigned char *data) {
    KernelBundle b;
    if (size != 0) nlohmann::from_json(nlohmann::json::from_msgpack(data, data + size), b);
    return b;
  }
  [[nodiscard]] std::vector<unsigned char> toMsgPack() const {
    nlohmann::json j;
    nlohmann::to_json(j, *this);
    return nlohmann::json::to_msgpack(j);
  }
};

} // namespace polyregion::runtime