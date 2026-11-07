#pragma once

#include <array>
#include <cstdlib>
#include <string>
#include <string_view>

#include "polyregion/env_keys.h"
#include "polyregion/types.h"

namespace polyregion::compiletime {

enum class GroupKind { Reduce, Inclusive, Exclusive };

inline constexpr std::array<GroupKind, 3> GroupKinds{GroupKind::Reduce, GroupKind::Inclusive, GroupKind::Exclusive};
inline constexpr std::array<std::string_view, 8> GroupOps{"Xchg", "Add", "Sub", "And", "Or", "Xor", "Min", "Max"};
inline constexpr std::array<std::string_view, 12> GroupTypes{"Bool1", "I8",  "I16", "I32", "I64", "U8",
                                                             "U16",   "U32", "U64", "F16", "F32", "F64"};

inline std::string_view groupKindName(const GroupKind kind) {
  switch (kind) {
    case GroupKind::Reduce: return "reduce";
    case GroupKind::Inclusive: return "inclusive";
    case GroupKind::Exclusive: return "exclusive";
  }
  return "";
}

namespace detail {

inline bool in(std::string_view value, std::initializer_list<std::string_view> set) {
  for (const auto &x : set)
    if (x == value) return true;
  return false;
}

inline bool wideNumeric(std::string_view type) { return in(type, {"I32", "U32", "I64", "U64", "F32", "F64"}); }
inline bool wideInteger(std::string_view type) { return in(type, {"I32", "U32", "I64", "U64"}); }

inline bool openclCollective(const GroupKind kind, std::string_view op, std::string_view type) {
  if (kind == GroupKind::Reduce && type == "Bool1") return op == "Or" || op == "And";
  return in(op, {"Add", "Min", "Max"}) && wideNumeric(type);
}

} // namespace detail

// the single decision of which work-group collectives a target lowers itself: backends handle exactly these and the
// SubgroupLower pass lowers the rest, see groupNativeArg
inline bool groupNative(const Target target, const std::string_view arch, const GroupKind kind, const std::string_view op,
                        const std::string_view type) {
  switch (target) {
    case Target::Source_C_OpenCL2_0: return detail::openclCollective(kind, op, type);
    case Target::Object_LLVM_SPIRV32_Kernel:
    case Target::Object_LLVM_SPIRV64_Kernel: return spirvKernelGeneric(arch) && detail::openclCollective(kind, op, type);
    case Target::Object_LLVM_NVPTX64:
      if (kind != GroupKind::Reduce) return false;
      if (detail::in(op, {"Add", "Min", "Max"})) return detail::wideNumeric(type);
      return detail::in(op, {"And", "Or", "Xor"}) && detail::wideInteger(type);
    default: return false;
  }
}

// `kind.op.type` entries joined by `|` for every collective the target handles natively; empty when none
inline std::string groupNativeArg(const Target target, const std::string_view arch) {
  std::string out;
  for (const auto kind : GroupKinds)
    for (const auto op : GroupOps)
      for (const auto type : GroupTypes)
        if (groupNative(target, arch, kind, op, type)) {
          if (!out.empty()) out += '|';
          out += groupKindName(kind);
          out += '.';
          out += op;
          out += '.';
          out += type;
        }
  return out;
}

// SubgroupLower arguments that lower every work-group collective except those the target handles natively
inline std::string groupLowerArgs(const Target target, const std::string_view arch) {
  const auto native = groupNativeArg(target, arch);
  return native.empty() ? "lowerGroups=true" : "lowerGroups=true,nativeGroups=" + native;
}

// POLYREGION_HOST_ABI=lane builds the thread-per-lane kernels that the group abi is checked against
inline bool hostLaneAbiForced() {
  const char *abi = std::getenv(env::PolyregionHostAbi);
  return abi && std::string_view(abi) == "lane";
}

// pass tail shared by the host and C11 entry pipelines; GroupFission picks the host launch abi
inline std::string hostGroupPipelineTail(const std::string &groupArgs, const bool respace) {
  return ";SubgroupLower(lowerSubgroups=false," + groupArgs + ",maxGroupSize=" + std::to_string(program_meta::HostWorkgroupMaxLanes)
         + ");StructuredExit" + (respace ? ";RegionRespace" : "") + (hostLaneAbiForced() ? "" : ";GroupFission");
}

} // namespace polyregion::compiletime
