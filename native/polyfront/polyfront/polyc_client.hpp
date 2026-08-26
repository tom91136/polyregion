#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "polyfront/package.hpp"
#include "polyregion/types.h"

#include "ast.h"

namespace polyregion::polyfront::package {

[[nodiscard]] Checked<polyast::CompileBundle> compileProgram(const polyast::ProgramLinkRequest &request, const std::string &executable,
                                                             compiletime::Target hostTarget, const std::string &hostArch,
                                                             const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                                                             std::optional<int> stackDepth = {});

} // namespace polyregion::polyfront::package
