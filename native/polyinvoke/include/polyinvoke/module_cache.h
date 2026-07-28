#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

#include "polyregion/cache.hpp"

namespace polyregion::invoke::detail {

inline constexpr const char *ModuleCompilerEnv[]{"OverrideDefaultFP64Settings", "IGC_EnableDPEmulation"};

inline std::string moduleCachePath(const std::string_view domain, const std::string_view identity, const std::string_view image) {
  if (identity.empty()) return {};
  std::string compilerEnv;
  for (const auto name : ModuleCompilerEnv) {
    const char *value = std::getenv(name);
    compilerEnv.append(name).append("=").append(value ? value : "").append(";");
  }
  return cache::path(domain, {identity, std::string_view(compilerEnv), image}, ".bin");
}

} // namespace polyregion::invoke::detail
