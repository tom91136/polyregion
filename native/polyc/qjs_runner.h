#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "polyregion/aliases.h"
#include "polyregion/export.h"

struct JSRuntime;
struct JSContext;

namespace polyregion::polypass {

class POLYREGION_EXPORT JsPassRunner {
  JSRuntime *rt = nullptr;
  JSContext *ctx = nullptr;

public:
  JsPassRunner();
  ~JsPassRunner();
  JsPassRunner(const JsPassRunner &) = delete;
  JsPassRunner &operator=(const JsPassRunner &) = delete;

  String loadModule(std::string_view source, std::string_view moduleId = "polypass.js");

  Vector<uint8_t> runPipeline(std::string_view spec, const Vector<uint8_t> &programBytes, String &error);
};

} // namespace polyregion::polypass
