#pragma once

#include <memory>
#include <string>

#include "pass_runner.h"

namespace polyregion::polypass {

class POLYREGION_EXPORT JsPassRunner final : public PassRunner {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  explicit JsPassRunner(std::string path);
  JsPassRunner();
  ~JsPassRunner() override;
  JsPassRunner(const JsPassRunner &) = delete;
  JsPassRunner &operator=(const JsPassRunner &) = delete;

  String load() override;
  const Vector<String> &passNames() const override;
  std::optional<String> passDescr(std::string_view name) const override;
  Vector<uint8_t> runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) override;
  std::string_view tag() const override;

  String loadModule(std::string_view source);
};

String hostArchTag();

} // namespace polyregion::polypass
