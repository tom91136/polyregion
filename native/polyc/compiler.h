#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ast.h"
#include "polyregion/export.h"
#include "polyregion/types.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

POLYREGION_EXPORT void initialise();

struct POLYREGION_EXPORT Options {
  POLYREGION_EXPORT compiletime::Target target;
  POLYREGION_EXPORT std::string arch;
};

POLYREGION_EXPORT std::vector<polyast::CompileLayout> layoutOf(const std::vector<polyast::StructDef> &sdefs, const Options &options);
POLYREGION_EXPORT std::vector<polyast::CompileLayout> layoutOf(const polyast::Bytes &bytes, const Options &options);

POLYREGION_EXPORT polyast::CompileResult compile(const polyast::Program &program, const Options &options, const compiletime::OptLevel &opt);
POLYREGION_EXPORT polyast::CompileResult compile(const polyast::Bytes &astBytes, const Options &options, const compiletime::OptLevel &opt);

} // namespace polyregion::compiler
