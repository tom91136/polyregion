#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "aspartame/all.hpp"

#include "polyfront/package_program.hpp"

namespace polyregion::polystl::hip {

using namespace aspartame;
using namespace polyregion::polyast;

namespace detail {

constexpr std::string_view BlockSize = "block_size";
constexpr std::string_view DevicePartition = "device_partition";

inline bool launchStateType(const Type::Any &type) {
  if (!type.is<Type::Struct>()) return false;
  const auto name = canonicalName(type);
  return name.find("rocprim") != std::string::npos && name.find(DevicePartition) != std::string::npos;
}

inline std::optional<std::pair<std::string, uint32_t>> blockSize(const Function &function) {
  for (const auto &variable : function.collect_all<Stmt::Var>()) {
    if (!variable.name.symbol.starts_with(BlockSize) || !variable.expr) continue;
    const auto constants = variable.expr->collect_all<Term::IntU32Const>();
    if (constants.size() == 1) return std::pair{variable.name.symbol, constants.front().value};
  }
  return {};
}

} // namespace detail

inline Program reconcileLaunchConstants(Program merged, const Program &device) {
  std::unordered_map<std::string, uint32_t> blockSizeByType;
  std::set<std::string> ambiguous;
  for (const auto &function : device.functions) {
    if (!function.convention.is<CallConvention::OffloadEntry>()) continue;
    const auto blockSize = detail::blockSize(function);
    if (!blockSize) continue;
    for (const auto &argument : function.decl.args) {
      if (!detail::launchStateType(argument.named.tpe)) continue;
      const auto key = canonicalName(argument.named.tpe);
      const auto [entry, inserted] = blockSizeByType.emplace(key, blockSize->second);
      if (!inserted && entry->second != blockSize->second) ambiguous.emplace(key);
    }
  }
  if (blockSizeByType.empty()) return merged;

  for (auto &function : merged.functions) {
    if (function.convention.is<CallConvention::OffloadEntry>()) continue;
    const auto hostBlockSize = detail::blockSize(function);
    if (!hostBlockSize) continue;
    std::set<uint32_t> targets;
    for (const auto &argument : function.decl.args) {
      if (!detail::launchStateType(argument.named.tpe)) continue;
      const auto key = canonicalName(argument.named.tpe);
      if (ambiguous.contains(key)) continue;
      if (const auto target = blockSizeByType.find(key); target != blockSizeByType.end()) targets.emplace(target->second);
    }
    if (targets.size() != 1 || *targets.begin() == hostBlockSize->second) continue;
    const auto target = *targets.begin();
    function = function.modify_all<Stmt::Var>([&](const auto &variable) {
      if (variable.name.symbol != hostBlockSize->first || !variable.expr) return variable;
      return variable.withExpr(
          variable.expr->template modify_all<Term::IntU32Const>([&](const auto &) { return Term::IntU32Const(target); }));
    });
  }
  return polyfront::packageProgram(std::move(merged.functions), std::move(merged.defs));
}

} // namespace polyregion::polystl::hip
