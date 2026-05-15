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
#include <vector>

#include "export.h"

#ifdef __clang__
  #define POLYREGION_RT_PROTECT [[clang::annotate("polyreflect-rt-protect")]]
#else
  #define POLYREGION_RT_PROTECT
#endif

namespace polyregion::invoke {

enum class POLYREGION_EXPORT Backend : uint8_t {
  CUDA,
  HIP,
  HSA,
  OpenCL,
  Vulkan,
  Metal,
  SharedObject,
  RelocatableObject,
  LevelZero,
};

} // namespace polyregion::invoke

namespace polyregion::compiletime {

enum class POLYREGION_EXPORT Target : uint8_t {
  Object_LLVM_HOST = 10,
  Object_LLVM_x86_64,
  Object_LLVM_AArch64,
  Object_LLVM_ARM,

  Object_LLVM_NVPTX64 = 20,
  Object_LLVM_AMDGCN,
  Object_LLVM_SPIRV32_Kernel,
  Object_LLVM_SPIRV64_Kernel,
  Object_LLVM_SPIRV_GLCompute,

  Source_C_C11 = 30,
  Source_C_OpenCL1_1,
  Source_C_Metal1_0,
};

enum class POLYREGION_EXPORT OptLevel : uint8_t {
  O0 = 10,
  O1,
  O2,
  O3,
  Ofast,
};

struct POLYREGION_EXPORT TargetSpec {
  std::string_view canonical;
  std::vector<std::string_view> aliases;
  Target codegen;
  invoke::Backend runtime;
  bool sharedAddressSpace;
  std::vector<std::string_view> requiredDeviceFeatures;

  POLYREGION_RT_PROTECT static const std::vector<TargetSpec> &registry() {
    using T = Target;
    using B = invoke::Backend;
    // clang-format off
    static const std::vector<TargetSpec> r = {
      {"host",            {"native"},                       T::Object_LLVM_HOST,             B::RelocatableObject, true,  {}},
      {"x86_64",          {},                               T::Object_LLVM_x86_64,           B::RelocatableObject, true,  {}},
      {"aarch64",         {"arm64"},                        T::Object_LLVM_AArch64,          B::RelocatableObject, true,  {}},
      {"arm",             {},                               T::Object_LLVM_ARM,              B::RelocatableObject, true,  {}},
      {"cuda",            {"ptx", "nvptx64"},               T::Object_LLVM_NVPTX64,          B::CUDA,              true,  {}},
      {"hsa",             {"amdgcn", "amdgpu"},             T::Object_LLVM_AMDGCN,           B::HSA,               true,  {}},
      {"hip",             {},                               T::Object_LLVM_AMDGCN,           B::HIP,               true,  {}},
      {"spirv64_kernel",  {"spirv", "spirv64",
                           "spirv_kernel", "opencl_spirv"}, T::Object_LLVM_SPIRV64_Kernel,   B::OpenCL,            false, {"spirv_kernel"}},
      {"spirv32_kernel",  {"spirv32"},                      T::Object_LLVM_SPIRV32_Kernel,   B::OpenCL,            false, {"spirv_kernel"}},
      {"spirv_glcompute", {"vulkan", "vulkan_spirv"},       T::Object_LLVM_SPIRV_GLCompute,  B::Vulkan,            false, {"spirv_glcompute"}},
      {"opencl1_1",       {"opencl"},                       T::Source_C_OpenCL1_1,           B::OpenCL,            false, {"source"}},
      {"metal1_0",        {"metal"},                        T::Source_C_Metal1_0,            B::Metal,             true,  {}},
      {"c11",             {},                               T::Source_C_C11,                 B::SharedObject,      true,  {}},
    };
    // clang-format on
    return r;
  }

  POLYREGION_RT_PROTECT static std::optional<TargetSpec> findByCodegen(Target t) {
    for (const auto &s : registry())
      if (s.codegen == t) return s;
    return std::nullopt;
  }

  POLYREGION_RT_PROTECT static std::optional<TargetSpec> findByName(std::string_view name) {
    auto eq = [](std::string_view a, std::string_view b) {
      if (a.size() != b.size()) return false;
      for (size_t i = 0; i < a.size(); ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y + 32);
        if (x != y) return false;
      }
      return true;
    };
    for (const auto &s : registry()) {
      if (eq(s.canonical, name)) return s;
      for (const auto &a : s.aliases)
        if (eq(a, name)) return s;
    }
    return std::nullopt;
  }

  struct ParsedRef;
  POLYREGION_RT_PROTECT static std::optional<ParsedRef> parse(std::string_view input);
};

struct POLYREGION_EXPORT TargetSpec::ParsedRef {
  TargetSpec spec;
  std::string hint;
};

POLYREGION_RT_PROTECT inline std::optional<TargetSpec::ParsedRef> TargetSpec::parse(std::string_view input) {
  auto at = input.find('@');
  auto head = (at == std::string_view::npos) ? input : input.substr(0, at);
  auto tail = (at == std::string_view::npos) ? std::string_view{} : input.substr(at + 1);
  if (auto s = findByName(head)) return ParsedRef{*s, std::string(tail)};
  return std::nullopt;
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

POLYREGION_RT_PROTECT static constexpr size_t byteOfType(Type t) {
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

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime

namespace polyregion::runtime {

enum class POLYREGION_EXPORT PlatformKind : uint8_t { HostThreaded = 1, Managed };

POLYREGION_RT_PROTECT static constexpr std::optional<PlatformKind> parsePlatformKind(std::string_view name) {
  if (name == "host" || name == "hostthreaded") return PlatformKind::HostThreaded;
  if (name == "managed") return PlatformKind::Managed;
  return {};
}
POLYREGION_RT_PROTECT static constexpr runtime::PlatformKind targetPlatformKind(const compiletime::Target &target) {
  switch (target) {
    case compiletime::Target::Object_LLVM_HOST:
    case compiletime::Target::Object_LLVM_x86_64:
    case compiletime::Target::Object_LLVM_AArch64:
    case compiletime::Target::Object_LLVM_ARM:
    case compiletime::Target::Source_C_C11: //
      return runtime::PlatformKind::HostThreaded;
    case compiletime::Target::Object_LLVM_NVPTX64:
    case compiletime::Target::Object_LLVM_AMDGCN:
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0: //
      return runtime::PlatformKind::Managed;
  }
}

enum class POLYREGION_EXPORT ModuleFormat : uint8_t { Source = 1, Object, DSO, PTX, HSACO, SPIRV_Kernel, SPIRV_GLCompute };
POLYREGION_RT_PROTECT static constexpr std::optional<ModuleFormat> parseModuleFormat(std::string_view name) {
  if (name == "src" || name == "source") return ModuleFormat::Source;
  if (name == "obj" || name == "object") return ModuleFormat::Object;
  if (name == "dso") return ModuleFormat::DSO;
  if (name == "ptx") return ModuleFormat::PTX;
  if (name == "hsaco") return ModuleFormat::HSACO;
  if (name == "spirv_kernel" || name == "spirv") return ModuleFormat::SPIRV_Kernel;
  if (name == "spirv_glcompute") return ModuleFormat::SPIRV_GLCompute;
  return {};
}

POLYREGION_RT_PROTECT static constexpr std::optional<runtime::ModuleFormat> moduleFormatOf(const compiletime::Target &target) {
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
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel: //
      return runtime::ModuleFormat::SPIRV_Kernel;
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute: //
      return runtime::ModuleFormat::SPIRV_GLCompute;
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

template <typename T> POLYREGION_RT_PROTECT constexpr std::underlying_type_t<T> to_underlying(T e) {
  return static_cast<std::underlying_type_t<T>>(e);
}
POLYREGION_RT_PROTECT constexpr LayoutAttrs operator|(const LayoutAttrs l, const LayoutAttrs r) {
  return static_cast<LayoutAttrs>(to_underlying(l) | to_underlying(r));
}
POLYREGION_RT_PROTECT constexpr LayoutAttrs &operator|=(LayoutAttrs &lhs, const LayoutAttrs rhs) {
  lhs = lhs | rhs;
  return lhs;
}
POLYREGION_RT_PROTECT constexpr bool isSet(const LayoutAttrs value, const LayoutAttrs flag) {
  return (to_underlying(value) & to_underlying(flag)) != 0;
}

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

  template <typename T> POLYREGION_RT_PROTECT static TypeLayout named(const char *name) {
    return {
        .name = name,
        .sizeInBytes = sizeof(T),
        .alignmentInBytes = alignof(T),
        .attrs = LayoutAttrs::None,
        .memberCount = 0,
        .members = {},
    };
  }

  POLYREGION_RT_PROTECT void visualise(std::FILE *fd, const std::function<void(size_t, const AggregateMember &)> &show = {}, //
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

  POLYREGION_RT_PROTECT void print(std::FILE *fd) const {
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
  POLYREGION_RT_PROTECT std::size_t operator()(polyregion::compiletime::Target t) const noexcept {
    return static_cast<std::underlying_type_t<polyregion::compiletime::Target>>(t);
  }
};