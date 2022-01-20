#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "export.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] constexpr int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

enum class EXPORT Backend : uint8_t { Invalid = 0, LLVM, OpenCL, CUDA };

struct EXPORT Event {
  int64_t epochMillis;
  std::string name;
  int64_t elapsedNanos;
  Event(int64_t epochMillis, std::string name, int64_t elapsedNanos)
      : epochMillis(epochMillis), name(std::move(name)), elapsedNanos(elapsedNanos) {}
};

struct EXPORT Compilation {
  std::optional<std::vector<uint8_t>> binary;
  std::optional<std::string> disassembly;
  std::vector<Event> events;
  std::string messages;
  Compilation() = default;
  explicit Compilation(decltype(messages) messages) : messages(std::move(messages)) {}
  Compilation(decltype(binary) binary,           //
              decltype(disassembly) disassembly, //
              decltype(events) events,           //
              decltype(messages) messages)
      : binary(std::move(binary)), disassembly(std::move(disassembly)), events(std::move(events)),
        messages(std::move(messages)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Compilation &compilation);
};

EXPORT void initialise();

enum class Tpe { Int };

struct EXPORT Member {
  Tpe tpe;
  uint32_t offset;
  uint32_t size;
};

struct EXPORT Layout {
  uint64_t sizeInBytes;
  uint64_t alignment;
  std::vector<Member> members;
};

EXPORT Layout layoutOf(std::vector<Tpe> members, bool packed);

EXPORT Compilation compile(std::vector<uint8_t> ast);

} // namespace polyregion::compiler
