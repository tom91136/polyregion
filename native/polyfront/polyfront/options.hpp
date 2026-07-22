#pragma once

#include <charconv>
#include <optional>
#include <string_view>

#include "polyregion/env_keys.h"

namespace polyregion::polyfront {

constexpr auto PolyfrontExe = env::PolyfrontExe;
constexpr auto PolyfrontVerbose = env::PolyfrontVerbose;
constexpr auto PolyfrontTargets = env::PolyfrontTargets;
constexpr auto PolyfrontStackDepth = env::PolyfrontStackDepth;
constexpr auto PolyfrontJit = env::PolyfrontJit;

inline std::optional<int> parsePositiveInt(std::string_view s) {
  int n = 0;
  const auto *end = s.data() + s.size();
  if (const auto [ptr, ec] = std::from_chars(s.data(), end, n); ec == std::errc{} && ptr == end && n > 0) return n;
  return {};
}

} // namespace polyregion::polyfront
