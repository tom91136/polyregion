#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "export.h"
#include "generated/polyast.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

enum class EXPORT Backend : uint8_t { Invalid = 0, LLVM, OpenCL, CUDA };

struct EXPORT Member {
  polyast::Named name;
  uint64_t offsetInBytes;
  uint64_t sizeInBytes;
  Member(decltype(name) name, decltype(offsetInBytes) offsetInBytes, decltype(sizeInBytes) sizeInBytes)
      : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
  friend std::ostream &operator<<(std::ostream &, const Member &);
};

struct EXPORT Layout {
  polyast::Sym name;
  uint64_t sizeInBytes;
  uint64_t alignment;
  std::vector<Member> members;
  Layout(decltype(name) name, decltype(sizeInBytes) sizeInBytes, decltype(alignment) alignment,
         decltype(members) members)
      : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
  friend std::ostream &operator<<(std::ostream &, const Layout &);
};

struct EXPORT Event {
  int64_t epochMillis;
  std::string name;
  int64_t elapsedNanos;
  Event(int64_t epochMillis, std::string name, int64_t elapsedNanos)
      : epochMillis(epochMillis), name(std::move(name)), elapsedNanos(elapsedNanos) {}
};

using Bytes = std::vector<uint8_t>;

struct EXPORT Compilation {
  std::vector<Layout> layouts;
  std::optional<Bytes> binary;
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
  EXPORT friend std::ostream &operator<<(std::ostream &, const Compilation &);
};

EXPORT void initialise();

EXPORT Layout layoutOf(const polyast::StructDef &);
EXPORT Layout layoutOf(const Bytes &);

EXPORT Compilation compile(const polyast::Program &);
EXPORT Compilation compile(const Bytes &);

} // namespace polyregion::compiler
