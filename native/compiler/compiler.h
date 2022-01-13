#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "export.h"

namespace polyregion::compiler {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;
constexpr inline uint64_t elapsedNs(const TimePoint &a, const TimePoint &b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

enum class EXPORT Backend : uint8_t { Invalid = 0, LLVM, OpenCL, CUDA };

struct EXPORT Compilation {
  std::optional<std::vector<uint8_t>> binary;
  std::optional<std::string> disassembly;
  std::vector<std::pair<std::string, uint64_t>> elapsed;
  std::string messages;
  Compilation() = default;
  explicit Compilation(decltype(messages) messages) : messages(std::move(messages)) {}
  Compilation(decltype(binary) binary,           //
              decltype(disassembly) disassembly, //
              decltype(elapsed) elapsed,         //
              decltype(messages) messages)
      : binary(std::move(binary)), disassembly(std::move(disassembly)), elapsed(std::move(elapsed)),
        messages(std::move(messages)) {}
};

EXPORT void initialise();

EXPORT Compilation compile(std::vector<uint8_t> ast);

} // namespace polyregion::compiler
