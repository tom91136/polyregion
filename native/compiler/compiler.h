#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "export.h"
#include "ast.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

POLYREGION_EXPORT void initialise();

struct POLYREGION_EXPORT Options {
  POLYREGION_EXPORT polyast::Target target;
  POLYREGION_EXPORT std::string arch;
};

POLYREGION_EXPORT std::vector<polyast::Layout> layoutOf(const std::vector<polyast::StructDef> &sdefs, const Options &options);
POLYREGION_EXPORT std::vector<polyast::Layout> layoutOf(const polyast::Bytes &bytes, const Options &options);

POLYREGION_EXPORT polyast::Compilation compile(const polyast::Program &program, const Options &options, const polyast::OptLevel &opt);
POLYREGION_EXPORT polyast::Compilation compile(const polyast::Bytes &astBytes, const Options &options, const polyast::OptLevel &opt);

} // namespace polyregion::compiler
