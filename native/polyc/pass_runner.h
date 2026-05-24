#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "polyregion/aliases.h"
#include "polyregion/export.h"

namespace polyregion::polypass {

struct POLYREGION_EXPORT PassRunner {
  virtual ~PassRunner() = default;
  virtual String load() = 0;
  virtual const Vector<String> &passNames() const = 0;
  virtual std::optional<String> passDescr(std::string_view name) const = 0;
  virtual Vector<uint8_t> runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) = 0;
  virtual std::string_view tag() const = 0;
};

} // namespace polyregion::polypass
