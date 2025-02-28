#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "export.h"

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
static constexpr std::string_view to_string(const Target &target) {
  switch (target) {
    case Target::Object_LLVM_HOST: return "host";
    case Target::Object_LLVM_x86_64: return "x86_64";
    case Target::Object_LLVM_AArch64: return "aarch64";
    case Target::Object_LLVM_ARM: return "arm";
    case Target::Object_LLVM_NVPTX64: return "nvptx64";
    case Target::Object_LLVM_AMDGCN: return "amdgcn";
    case Target::Object_LLVM_SPIRV32: return "spirv32";
    case Target::Object_LLVM_SPIRV64: return "spirv64";
    case Target::Source_C_C11: return "c11";
    case Target::Source_C_OpenCL1_1: return "opencl1_1";
    case Target::Source_C_Metal1_0: return "metal1_0";
  }
}
static constexpr std::optional<Target> targetFromOrdinal(std::underlying_type_t<Target> ordinal) {
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
static constexpr std::string_view to_string(const OptLevel &target) {
  switch (target) {
    case OptLevel::O0: return "O0";
    case OptLevel::O1: return "O1";
    case OptLevel::O2: return "O2";
    case OptLevel::O3: return "O3";
    case OptLevel::Ofast: return "Ofast";
  }
}
static constexpr std::optional<OptLevel> optFromOrdinal(std::underlying_type_t<OptLevel> ordinal) {
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

static const std::unordered_map<std::string, Target> &targets() {
  static std::unordered_map<std::string, Target> //
      data{                                      // obj
           {"host", Target::Object_LLVM_HOST},
           {"native", Target::Object_LLVM_HOST},
           {"x86_64", Target::Object_LLVM_x86_64},
           {"aarch64", Target::Object_LLVM_AArch64},
           {"arm64", Target::Object_LLVM_AArch64},
           {"arm", Target::Object_LLVM_ARM},

           // ptx
           {"nvptx64", Target::Object_LLVM_NVPTX64},
           {"ptx", Target::Object_LLVM_NVPTX64},
           {"cuda", Target::Object_LLVM_NVPTX64},

           // hsaco
           {"amdgcn", Target::Object_LLVM_AMDGCN},
           {"amdgpu", Target::Object_LLVM_AMDGCN},
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

static std::optional<Target> parseTarget(const std::string &name) {
  std::string lower(name.size(), {});
  for (size_t i = 0; i < name.size(); ++i)
    lower[i] = std::tolower(name[i]);
  if (auto it = targets().find(lower); it != targets().end()) return it->second;
  else return {};
}

} // namespace polyregion::compiletime

namespace polyregion::runtime {

enum class POLYREGION_EXPORT Type : uint8_t {
  Void = 1,
  Bool1,

  IntU8,
  IntU16,
  IntU32,
  IntU64,

  IntS8,
  IntS16,
  IntS32,
  IntS64,

  Float16,
  Float32,
  Float64,

  Ptr,
  Scratch,
};

static constexpr size_t byteOfType(Type t) {
  switch (t) {
    case Type::Void: return 0;
    case Type::Bool1: [[fallthrough]];
    case Type::IntU8: [[fallthrough]];
    case Type::IntS8: return 1;
    case Type::IntU16: [[fallthrough]];
    case Type::IntS16: [[fallthrough]];
    case Type::Float16: return 2;
    case Type::IntU32: [[fallthrough]];
    case Type::IntS32: [[fallthrough]];
    case Type::Float32: return 4;
    case Type::IntU64: [[fallthrough]];
    case Type::IntS64: [[fallthrough]];
    case Type::Float64: return 8;
    case Type::Ptr: [[fallthrough]];
    case Type::Scratch: return sizeof(void *);
    default: std::fprintf(stderr, "Unimplemented Type\n"); std::abort();
  }
}

static constexpr std::string_view to_string(const Type &type) {
  switch (type) {
    case Type::Void: return "Void";
    case Type::Bool1: return "Bool1";
    case Type::IntU8: return "UInt8";
    case Type::IntU16: return "UInt16";
    case Type::IntU32: return "UInt32";
    case Type::IntU64: return "UInt64";
    case Type::IntS8: return "Int8";
    case Type::IntS16: return "Int16";
    case Type::IntS32: return "Int32";
    case Type::IntS64: return "Int64";
    case Type::Float16: return "Float16";
    case Type::Float32: return "Float32";
    case Type::Float64: return "Float64";
    case Type::Ptr: return "Ptr";
    case Type::Scratch: return "Scratch";
    default: std::fprintf(stderr, "Unimplemented Type\n"); std::abort();
  }
}

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime

namespace polyregion::runtime {

enum class POLYREGION_EXPORT PlatformKind : uint8_t { HostThreaded = 1, Managed };
static constexpr std::string_view to_string(const PlatformKind &b) {
  switch (b) {
    case PlatformKind::HostThreaded: return "HostThreaded";
    case PlatformKind::Managed: return "Managed";
    default: std::fprintf(stderr, "Unimplemented PlatformKind\n"); std::abort();
  }
}
static constexpr std::optional<PlatformKind> parsePlatformKind(std::string_view name) {
  if (name == "host" || name == "hostthreaded") return PlatformKind::HostThreaded;
  if (name == "managed") return PlatformKind::Managed;
  return {};
}
static constexpr runtime::PlatformKind targetPlatformKind(const compiletime::Target &target) {
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
static constexpr std::string_view to_string(const ModuleFormat &x) {
  switch (x) {
    case ModuleFormat::Source: return "Source";
    case ModuleFormat::Object: return "Object";
    case ModuleFormat::DSO: return "DSO";
    case ModuleFormat::PTX: return "PTX";
    case ModuleFormat::HSACO: return "HSACO";
    case ModuleFormat::SPIRV: return "SPIRV";
    default: std::fprintf(stderr, "Unimplemented ModuleFormat\n"); std::abort();
  }
}
static constexpr std::optional<ModuleFormat> parseModuleFormat(std::string_view name) {
  if (name == "src" || name == "source") return ModuleFormat::Source;
  if (name == "obj" || name == "object") return ModuleFormat::Object;
  if (name == "dso") return ModuleFormat::DSO;
  if (name == "ptx") return ModuleFormat::PTX;
  if (name == "hsaco") return ModuleFormat::HSACO;
  if (name == "spirv") return ModuleFormat::SPIRV;
  return {};
}

static constexpr std::optional<runtime::ModuleFormat> moduleFormatOf(const compiletime::Target &target) {
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
  return {};
}

enum class LayoutAttrs : size_t {
  None = /*           */ 0,
  Opaque = /*    */ 1 << 0, // no members are pointers, member types are also opaque
  SelfOpaque = /**/ 1 << 1, // no members are pointers, member types may have pointers
  Primitive = /* */ 1 << 2, // no members
};

template <typename T> constexpr std::underlying_type_t<T> to_underlying(T e) { return static_cast<std::underlying_type_t<T>>(e); }
constexpr LayoutAttrs operator|(const LayoutAttrs l, const LayoutAttrs r) {
  return static_cast<LayoutAttrs>(to_underlying(l) | to_underlying(r));
}
constexpr LayoutAttrs &operator|=(LayoutAttrs &lhs, const LayoutAttrs rhs) {
  lhs = lhs | rhs;
  return lhs;
}
constexpr bool isSet(const LayoutAttrs value, const LayoutAttrs flag) { return (to_underlying(value) & to_underlying(flag)) != 0; }

struct TypeLayout;
struct AggregateMember {
  using ResolvePtrSize = size_t (*)(const void *ptr);

  const char *name;
  size_t offsetInBytes, sizeInBytes;
  size_t ptrIndirection;
  size_t componentSize;
  const TypeLayout *type;
  ResolvePtrSize resolvePtrSizeInBytes;
};

struct TypeLayout {
  const char *name;
  size_t sizeInBytes;
  size_t alignmentInBytes;
  LayoutAttrs attrs;
  size_t memberCount;
  const AggregateMember *members;

  template <typename T> static TypeLayout named(const char *name) {
    return {
        .name = name,
        .sizeInBytes = sizeof(T),
        .alignmentInBytes = alignof(T),
        .attrs = LayoutAttrs::None,
        .memberCount = 0,
        .members = {},
    };
  }

  void visualise(std::FILE *fd, const std::function<void(size_t, const AggregateMember &)> &show = {}, //
                 const size_t level = 0, const size_t offset = 0) const {
    constexpr size_t maxCol = 128;
    constexpr size_t alignColumn = 60; //

    const bool topLevel = level == 0;
    if (topLevel) std::fprintf(fd, "[%*zu]    ╭── ", 3, sizeInBytes);
    else std::fprintf(fd, "╭── [%*zu] ", 3, sizeInBytes);

    std::fprintf(fd, "`%s` (alignment=%ld, members=%ld, attrs={", name, alignmentInBytes, memberCount);
    if (isSet(attrs, LayoutAttrs::Opaque)) std::fprintf(fd, "Opaque ");
    if (isSet(attrs, LayoutAttrs::SelfOpaque)) std::fprintf(fd, "SelfOpaque ");
    if (isSet(attrs, LayoutAttrs::Primitive)) std::fprintf(fd, "Primitive ");
    std::fprintf(fd, "}) ──\n");

    for (size_t i = 0; i < memberCount; ++i) {
      const auto &m = members[i];
      std::fprintf(fd, "+%3zu~%-3zu │", offset + m.offsetInBytes, offset + m.offsetInBytes + m.sizeInBytes);
      for (size_t l = 0; l < level; ++l)
        std::fprintf(fd, "│");
      if (m.type && m.type->memberCount > 0 && m.ptrIndirection == 0) {
        m.type->visualise(fd, show, level + 1, offset + m.offsetInBytes);
      } else if (m.sizeInBytes == 0) {
        std::fprintf(fd, "[0-width] %s: %s\n", m.name, m.type ? m.type->name : "???");
      } else {
        const size_t nextOffset = i + 1 < memberCount ? members[i + 1].offsetInBytes : sizeInBytes;
        const size_t cols = nextOffset - m.offsetInBytes;
        for (size_t c = 0; c < std::min(maxCol, cols); ++c)
          std::fprintf(fd, c < m.sizeInBytes ? "■" : "□");
        if (cols > maxCol) std::fprintf(fd, "...");
        std::fprintf(fd, " %s: %s", m.name, m.type ? m.type->name : "???");
        for (size_t s = 0; s < m.ptrIndirection; ++s)
          std::fprintf(fd, "*");

        std::fprintf(fd, " (%ld bytes) ", m.sizeInBytes);
        if (show && m.type) {
          const size_t currentColumn = 8 + 3 + 1 +                                      // "+%3zu~%-3zu │"
                                       level +                                          // "│"
                                       std::min(maxCol, nextOffset - m.offsetInBytes) + // blocks
                                       std::strlen(m.name) + 2 +                        // "name: "
                                       (m.type ? std::strlen(m.type->name) : 3) +       // "type" or "???"
                                       m.ptrIndirection +                               // '*' pointer indirection
                                       9;                                               // " (N bytes) "
          std::fprintf(fd, "%*s", static_cast<int>(currentColumn < alignColumn ? alignColumn - currentColumn : 1), "");
          show(offset + m.offsetInBytes, m);
        }
        std::fprintf(fd, "\n");
      }
    }
    if (topLevel) std::fprintf(fd, "         ╰────────\n");
  }

  void print(std::FILE *fd) const {
    fprintf(fd, "TypeLayout{\n");
    fprintf(fd, "    .name = \"%s\",\n", name);
    fprintf(fd, "    .sizeInBytes = %zuULL,\n", sizeInBytes);
    fprintf(fd, "    .alignmentInBytes = %zuULL,\n", alignmentInBytes);
    fprintf(fd, "    .memberCount = %zuULL,\n", memberCount);
    fprintf(fd, "    .members = new AggregateMember[%zu]{\n", memberCount);
    for (size_t i = 0; i < memberCount; ++i) {
      const auto &m = members[i];
      fprintf(fd,
              "        { .name = \"%s\", .offsetInBytes = %zuULL, .sizeInBytes = %zuULL, .ptrIndirection = %zuULL, .componentSize = "
              "%zuULL, .type = (%s), resolver = %p }",
              m.name, m.offsetInBytes, m.sizeInBytes, m.ptrIndirection, m.componentSize, m.type ? m.type->name : "???",
              (void *)(m.resolvePtrSizeInBytes));
      if (i + 1 < memberCount) fprintf(fd, ",");
      fprintf(fd, "\n");
    }
    fprintf(fd, "    }\n");
    fprintf(fd, "};\n");
  }
};

struct KernelObject {
  PlatformKind kind;
  ModuleFormat format;
  size_t featureCount;
  const char **features;
  size_t imageLength;
  const unsigned char *image;
};

struct KernelBundle {
  const char *moduleName;

  size_t objectCount;
  const KernelObject *objects;

  size_t structCount;
  const TypeLayout *structs;
  size_t interfaceLayoutIdx;

  const char *metadata;
};

static_assert(std::is_standard_layout_v<TypeLayout>);
static_assert(std::is_standard_layout_v<AggregateMember>);
static_assert(std::is_standard_layout_v<KernelObject>);
static_assert(std::is_standard_layout_v<KernelBundle>);

} // namespace polyregion::runtime

template <> struct std::hash<polyregion::compiletime::Target> {
  std::size_t operator()(polyregion::compiletime::Target t) const noexcept {
    return static_cast<std::underlying_type_t<polyregion::compiletime::Target>>(t);
  }
};