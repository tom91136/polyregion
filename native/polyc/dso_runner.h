#pragma once

#include <memory>
#include <string>

#include "pass_runner.h"

namespace polyregion::polypass {

class POLYREGION_EXPORT DsoPassRunner final : public PassRunner {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  explicit DsoPassRunner(std::string path);
  ~DsoPassRunner() override;
  DsoPassRunner(const DsoPassRunner &) = delete;
  DsoPassRunner &operator=(const DsoPassRunner &) = delete;

  String load() override;
  [[nodiscard]] const Vector<String> &passNames() const override;
  [[nodiscard]] std::optional<String> passDescr(std::string_view name) const override;
  Vector<uint8_t> runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) override;
  [[nodiscard]] std::optional<uint32_t> packageAbiVersion() const override;
  Vector<uint8_t> runPackage(std::string_view operation, const Vector<uint8_t> &request, String &error) override;
  [[nodiscard]] std::string_view tag() const override;
};

} // namespace polyregion::polypass
