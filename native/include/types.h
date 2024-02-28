#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "export.h"
#include "json.hpp"

namespace polyregion::compiletime {

enum class POLYREGION_EXPORT Target : uint8_t {
  Object_LLVM_HOST = 10,
  Object_LLVM_x86_64,
  Object_LLVM_AArch64,
  Object_LLVM_ARM,

  Object_LLVM_NVPTX64 = 20,
  Object_LLVM_AMDGCN,
  Object_LLVM_SPIRV32,
  Object_LLVM_SPIRV64,

  Source_C_C11 = 30,
  Source_C_OpenCL1_1,
  Source_C_Metal1_0,
};
static inline constexpr std::string_view to_string(const Target &target) {
  switch (target) {
    case Target::Object_LLVM_HOST: return "Object_LLVM_HOST";
    case Target::Object_LLVM_x86_64: return "Object_LLVM_x86_64";
    case Target::Object_LLVM_AArch64: return "Object_LLVM_AArch64";
    case Target::Object_LLVM_ARM: return "Object_LLVM_ARM";
    case Target::Object_LLVM_NVPTX64: return "Object_LLVM_NVPTX64";
    case Target::Object_LLVM_AMDGCN: return "Object_LLVM_AMDGCN";
    case Target::Object_LLVM_SPIRV32: return "Object_LLVM_SPIRV32";
    case Target::Object_LLVM_SPIRV64: return "Object_LLVM_SPIRV64";
    case Target::Source_C_C11: return "Source_C_C11";
    case Target::Source_C_OpenCL1_1: return "Source_C_OpenCL1_1";
    case Target::Source_C_Metal1_0: return "Source_C_Metal1_0";
  }
}
static inline constexpr std::optional<Target> targetFromOrdinal(std::underlying_type_t<Target> ordinal) {
  auto target = static_cast<Target>(ordinal);
  switch (target) {
    case Target::Object_LLVM_HOST:
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_ARM:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN:
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_C11:
    case Target::Source_C_Metal1_0:
    case Target::Object_LLVM_SPIRV32:
    case Target::Object_LLVM_SPIRV64:
      return target;
      // XXX do not add default here, see -Werror=switch
  }
}

enum class POLYREGION_EXPORT OptLevel : uint8_t {
  O0 = 10,
  O1,
  O2,
  O3,
  Ofast,
};
static inline constexpr std::string_view to_string(const OptLevel &target) {
  switch (target) {
    case OptLevel::O0: return "O0";
    case OptLevel::O1: return "O1";
    case OptLevel::O2: return "O2";
    case OptLevel::O3: return "O3";
    case OptLevel::Ofast: return "Ofast";
  }
}
static inline constexpr std::optional<OptLevel> optFromOrdinal(std::underlying_type_t<OptLevel> ordinal) {
  auto target = static_cast<OptLevel>(ordinal);
  switch (target) {
    case OptLevel::O0:
    case OptLevel::O1:
    case OptLevel::O2:
    case OptLevel::O3:
    case OptLevel::Ofast:
      return target;
      // XXX do not add default here, see -Werror=switch
  }
}

static inline const std::unordered_map<std::string, Target> &targets() {
  static std::unordered_map<std::string, Target> //
      data{                                      // obj
           {"host", Target::Object_LLVM_HOST},
           {"native", Target::Object_LLVM_HOST},
           {"x86_64", Target::Object_LLVM_x86_64},
           {"aarch64", Target::Object_LLVM_AArch64},
           {"arm", Target::Object_LLVM_ARM},

           // ptx
           {"nvptx64", Target::Object_LLVM_NVPTX64},
           {"ptx", Target::Object_LLVM_NVPTX64},
           {"cuda", Target::Object_LLVM_NVPTX64},

           // hsaco
           {"amdgcn", Target::Object_LLVM_AMDGCN},
           {"hsa", Target::Object_LLVM_AMDGCN},
           {"hip", Target::Object_LLVM_AMDGCN},

           // spirv
           {"spirv32", Target::Object_LLVM_SPIRV32},
           {"spirv64", Target::Object_LLVM_SPIRV64},
           {"spirv", Target::Object_LLVM_SPIRV64},
           {"vulkan", Target::Object_LLVM_SPIRV64},

           // src
           {"c11", Target::Source_C_C11},
           {"opencl1_1", Target::Source_C_OpenCL1_1},
           {"opencl", Target::Source_C_OpenCL1_1},
           {"metal1_0", Target::Source_C_Metal1_0},
           {"metal", Target::Source_C_Metal1_0}};
  return data;
}

static inline std::optional<Target> parseTarget(const std::string &name) {
  std::string lower(name.size(), {});
  std::transform(name.begin(), name.end(), lower.begin(), [](auto c) { return std::tolower(c); });
  if (auto it = targets().find(lower); it != targets().end()) return it->second;
  else return {};
}

} // namespace polyregion::compiletime

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

static inline constexpr size_t byteOfType(Type t) {
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

static inline constexpr std::string_view to_string(const Type &type) {
  switch (type) {
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
static inline constexpr std::string_view to_string(const PlatformKind &b) {
  switch (b) {
    case PlatformKind::HostThreaded: return "HostThreaded";
    case PlatformKind::Managed: return "Managed";
  }
}
static inline constexpr std::optional<PlatformKind> parsePlatformKind(std::string_view name) {
  if (name == "host" || name == "hostthreaded") return PlatformKind::HostThreaded;
  if (name == "managed") return PlatformKind::Managed;
  return {};
}
static inline constexpr runtime::PlatformKind targetPlatformKind(const compiletime::Target &target) {
  switch (target) {
    case compiletime::Target::Object_LLVM_HOST:
    case compiletime::Target::Object_LLVM_x86_64:
    case compiletime::Target::Object_LLVM_AArch64:
    case compiletime::Target::Object_LLVM_ARM:
    case compiletime::Target::Source_C_C11: //
      return runtime::PlatformKind::HostThreaded;
    case compiletime::Target::Object_LLVM_NVPTX64:
    case compiletime::Target::Object_LLVM_AMDGCN:
    case compiletime::Target::Object_LLVM_SPIRV32:
    case compiletime::Target::Object_LLVM_SPIRV64:
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0: //
      return runtime::PlatformKind::Managed;
  }
}

enum class POLYREGION_EXPORT ModuleFormat : uint8_t { Source = 1, Object, DSO, PTX, HSACO, SPIRV };
static inline constexpr std::string_view to_string(const ModuleFormat &x) {
  switch (x) {
    case ModuleFormat::Source: return "Source";
    case ModuleFormat::Object: return "Object";
    case ModuleFormat::DSO: return "DSO";
    case ModuleFormat::PTX: return "PTX";
    case ModuleFormat::HSACO: return "HSACO";
    case ModuleFormat::SPIRV: return "SPIRV";
  }
}
static inline constexpr std::optional<ModuleFormat> parseModuleFormat(std::string_view name) {
  if (name == "src" || name == "source") return ModuleFormat::Source;
  if (name == "obj" || name == "object") return ModuleFormat::Object;
  if (name == "dso") return ModuleFormat::DSO;
  if (name == "ptx") return ModuleFormat::PTX;
  if (name == "hsaco") return ModuleFormat::HSACO;
  if (name == "spirv") return ModuleFormat::SPIRV;
  return {};
}

static inline constexpr runtime::ModuleFormat targetFormat(const compiletime::Target &target) {
  switch (target) {
    case compiletime::Target::Object_LLVM_HOST:
    case compiletime::Target::Object_LLVM_x86_64:
    case compiletime::Target::Object_LLVM_AArch64:
    case compiletime::Target::Object_LLVM_ARM: //
      return runtime::ModuleFormat::Object;
    case compiletime::Target::Object_LLVM_NVPTX64: //
      return runtime::ModuleFormat::PTX;
    case compiletime::Target::Object_LLVM_AMDGCN: //
      return runtime::ModuleFormat::HSACO;
    case compiletime::Target::Object_LLVM_SPIRV32:
    case compiletime::Target::Object_LLVM_SPIRV64: //
      return runtime::ModuleFormat::PTX;
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0:
    case compiletime::Target::Source_C_C11: //
      return runtime::ModuleFormat::Source;
  }
}

} // namespace polyregion::runtime

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
  static KernelBundle fromMsgPack(size_t size, const unsigned char *data) {
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