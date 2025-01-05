#pragma once

#include <string>
#include <vector>

#include "polyregion/types.h"

namespace polyregion::polystl {
struct Options {
  using Target = std::pair<compiletime::Target, std::string>;

  bool verbose = false;
  std::string executable;
  std::vector<Target> targets;
};
} // namespace polyregion::polystl

template <> struct std::hash<std::pair<polyregion::compiletime::Target, std::string>> {
  std::size_t operator()(const std::pair<polyregion::compiletime::Target, std::string> &p) const noexcept;
};