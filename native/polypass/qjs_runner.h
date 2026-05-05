#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "polyregion/aliases.h"
#include "polyregion/export.h"

struct JSRuntime;
struct JSContext;

namespace polyregion::polypass {

// Loads a JS bundle (the Scala.js-emitted pass module) into a QuickJS runtime and dispatches
// pass invocations against it. The wire format is msgpack-encoded `polyast::Program`, the same
// bytes produced by `nlohmann::json::to_msgpack(polyast::program_to_json(p))` on the C++ side
// and by upickle's MsgPack.Codec on the Scala side.
class POLYREGION_EXPORT JsPassRunner {
  JSRuntime *rt = nullptr;
  JSContext *ctx = nullptr;

public:
  JsPassRunner();
  ~JsPassRunner();
  JsPassRunner(const JsPassRunner &) = delete;
  JsPassRunner &operator=(const JsPassRunner &) = delete;

  // Evaluate `source` as the entry-point module. moduleId is the filename used in stack traces.
  // Returns the JS exception message on failure, or empty on success.
  String loadModule(std::string_view source, std::string_view moduleId = "polypass.js");

  // Invoke the JS-side `runPass(name: string, bytes: Uint8Array): Uint8Array` and return the
  // resulting bytes. On failure, `error` is populated and the returned vector is empty.
  Vector<uint8_t> runPass(std::string_view passName, const Vector<uint8_t> &programBytes, String &error);
};

} // namespace polyregion::polypass
