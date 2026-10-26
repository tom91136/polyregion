#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "aspartame/all.hpp"

#include "polyregion/env_keys.h"
#include "polyregion/types.h"

namespace polyregion::polyfront {

constexpr auto PolyfrontExe = env::PolyfrontExe;
constexpr auto PolyfrontVerbose = env::PolyfrontVerbose;
constexpr auto PolyfrontTargets = env::PolyfrontTargets;
constexpr auto PolyfrontStackDepth = env::PolyfrontStackDepth;
constexpr auto PolyfrontJit = env::PolyfrontJit;
constexpr auto PolyfrontEmitLibrary = env::PolyfrontEmitLibrary;
constexpr auto PolyfrontLibraryPath = env::PolyfrontLibraryPath;
constexpr auto PolyfrontLibraryCapabilities = env::PolyfrontLibraryCapabilities;
constexpr auto PolyfrontRequireOffload = env::PolyfrontRequireOffload;

using namespace aspartame;

inline std::optional<int> parsePositiveInt(std::string_view s) {
  int n = 0;
  const auto *end = s.data() + s.size();
  if (const auto [ptr, ec] = std::from_chars(s.data(), end, n); ec == std::errc{} && ptr == end && n > 0) return n;
  return {};
}

inline std::vector<std::string> capabilityErrors(const std::vector<std::string> &capabilities) {
  return capabilities ^ collect([](const auto &capability) -> std::optional<std::string> {
           const auto qualified = capability ^ split_once('@');
           if (!qualified) return {};
           if (qualified->first.empty() || qualified->second.empty()) return "malformed library capability `" + capability + "`";
           if (!compiletime::TargetSpec::findByName(qualified->second))
             return "unknown target `" + qualified->second + "` in library capability `" + capability + "`";
           return {};
         });
}

} // namespace polyregion::polyfront
