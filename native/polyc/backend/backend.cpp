#include "backend.h"

#include <charconv>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyregion/conventions.h"
#include "polyregion/types.h"

using namespace aspartame;

polyregion::backend::BackendException::BackendException(const char *what, const char *file, const char *fn, const int line)
    : BackendException(std::string(what), file, fn, line) {}

polyregion::backend::BackendException::BackendException(const std::string &what, const char *file, const char *fn, const int line)
    : std::logic_error("[Backend] In " + std::string(file) + ":" + std::to_string(line) + " (" + std::string(fn) + ")\n" + what) {}

void polyregion::backend::requireGroupNative(const compiletime::Target target, const std::string &arch, const compiletime::GroupKind kind,
                                             const polyast::AtomicOp::Any &op, const polyast::Type::Any &type) {
  const auto opName = polyast::repr(op);
  const auto typeName = polyast::canonicalName(type);
  if (!compiletime::groupNative(target, arch, kind, opName, typeName))
    throw BackendException(fmt::format("work-group {} {} on {} is not native, SubgroupLower must lower it",
                                       compiletime::groupKindName(kind), opName, typeName));
}

std::optional<int64_t> polyregion::backend::hostAbiOf(const polyast::Program &program, const polyast::Function &fn) {
  const auto key = conventions::HostAbiMeta + polyast::fqcn(fn.decl.name);
  return program.metadata ^ find([&](const auto &entry) { return entry.key == key; }) ^ map([&](const auto &entry) {
           int64_t value = 0;
           const auto *end = entry.value.data() + entry.value.size();
           if (const auto [ptr, ec] = std::from_chars(entry.value.data(), end, value); ec != std::errc{} || ptr != end || value < 0)
             throw BackendException(fmt::format("Program metadata `{}` is not a non-negative integer: `{}`", key, entry.value));
           if (value > program_meta::HostAbiGroup) throw BackendException(fmt::format("Unknown host abi {} for {}", value, key));
           return value;
         });
}
