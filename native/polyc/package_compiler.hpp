#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "polyregion/types.h"

#include "ast.h"

namespace polyregion::compiler::package {

template <typename T> struct Result {
  std::optional<T> value;
  std::vector<std::string> errors;
  explicit operator bool() const { return value.has_value(); }
};

[[nodiscard]] Result<polyast::Package> link(const polyast::PackageLinkRequest &request);
[[nodiscard]] Result<polyast::CompileBundle> compile(const polyast::ProgramLinkRequest &request, compiletime::Target hostTarget,
                                                     const std::string &hostArch,
                                                     const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                                                     std::optional<int> stackDepth = {});

} // namespace polyregion::compiler::package
