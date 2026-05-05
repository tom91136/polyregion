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

  Vector<uint8_t> runPass(std::string_view passName, const Vector<uint8_t> &programBytes, String &error);

  // Resolution order: $POLYPASS_JS env, <exe-dir>/polypass.js, <exe-dir>/../lib/polypass.js,
  // then the build-time POLYPASS_JS_DEV_PATH baked in by CMake. Returns "" if unfound.
  static String findBundle();
};

} // namespace polyregion::polypass
