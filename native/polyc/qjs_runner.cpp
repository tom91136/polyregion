#include "qjs_runner.h"

#include <cstdlib>
#include <filesystem>

#include "fmt/format.h"
#include "quickjs.h"

#include "llvm/Support/FileSystem.h"

#ifndef POLYPASS_JS_DEV_PATH
  #define POLYPASS_JS_DEV_PATH ""
#endif

namespace polyregion::polypass {

namespace {
[[maybe_unused]] void addrAnchor() {}
} // namespace

String JsPassRunner::findBundle() {
  namespace fs = std::filesystem;
  if (auto env = std::getenv("POLYPASS_JS"); env && *env && fs::exists(env)) return env;
  const auto exe = llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&addrAnchor));
  if (!exe.empty()) {
    const fs::path dir = fs::path(exe).parent_path();
    for (const fs::path candidate : {dir / "polypass.js", dir / ".." / "lib" / "polypass.js"})
      if (fs::exists(candidate)) return fs::canonical(candidate).string();
  }
  if (constexpr std::string_view dev = POLYPASS_JS_DEV_PATH; !dev.empty() && fs::exists(dev)) return String(dev);
  return {};
}

namespace {

String formatException(JSContext *ctx) {
  JSValue ex = JS_GetException(ctx);
  const char *msg = JS_ToCString(ctx, ex);
  String out = msg ? msg : "<no exception message>";
  JSValue stack = JS_GetPropertyStr(ctx, ex, "stack");
  if (!JS_IsUndefined(stack)) {
    if (const char *s = JS_ToCString(ctx, stack); s) {
      out += "\n";
      out += s;
      JS_FreeCString(ctx, s);
    }
  }
  JS_FreeValue(ctx, stack);
  if (msg) JS_FreeCString(ctx, msg);
  JS_FreeValue(ctx, ex);
  return out;
}

} // namespace

JsPassRunner::JsPassRunner() {
  rt = JS_NewRuntime();
  ctx = JS_NewContext(rt);
}

JsPassRunner::~JsPassRunner() {
  if (ctx) JS_FreeContext(ctx);
  if (rt) JS_FreeRuntime(rt);
}

String JsPassRunner::loadModule(std::string_view source, std::string_view moduleId) {
  // Scala.js NoModule emits `let runPass;` at the script top, which doesn't create a globalThis
  // property. Pin known exports in the same eval where the lexical binding is in scope.
  String wrapped;
  wrapped.reserve(source.size() + 64);
  wrapped.append(source);
  wrapped.append("\n;try{globalThis.runPass=runPass;}catch(_){}\n");
  JSValue r = JS_Eval(ctx, wrapped.data(), wrapped.size(), String(moduleId).c_str(), JS_EVAL_TYPE_GLOBAL);
  if (JS_IsException(r)) {
    JS_FreeValue(ctx, r);
    return formatException(ctx);
  }
  JS_FreeValue(ctx, r);
  return {};
}

Vector<uint8_t> JsPassRunner::runPass(std::string_view passName, const Vector<uint8_t> &programBytes, String &error) {
  Vector<uint8_t> out;
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue runPass = JS_GetPropertyStr(ctx, global, "runPass");
  if (!JS_IsFunction(ctx, runPass)) {
    error = "runPass: JS bundle does not export a global `runPass` function";
    JS_FreeValue(ctx, runPass);
    JS_FreeValue(ctx, global);
    return out;
  }

  JSValue nameVal = JS_NewStringLen(ctx, passName.data(), passName.size());
  JSValue bufVal = JS_NewArrayBufferCopy(ctx, reinterpret_cast<const uint8_t *>(programBytes.data()), programBytes.size());

  JSValue argv[2] = {nameVal, bufVal};
  JSValue result = JS_Call(ctx, runPass, global, 2, argv);
  JS_FreeValue(ctx, nameVal);
  JS_FreeValue(ctx, bufVal);
  JS_FreeValue(ctx, runPass);
  JS_FreeValue(ctx, global);

  if (JS_IsException(result)) {
    error = formatException(ctx);
    JS_FreeValue(ctx, result);
    return out;
  }

  size_t size = 0;
  uint8_t *data = JS_GetArrayBuffer(ctx, &size, result);
  if (!data) {
    JSValue buf = JS_GetTypedArrayBuffer(ctx, result, nullptr, nullptr, nullptr);
    if (JS_IsException(buf)) {
      error = "runPass: return value is neither an ArrayBuffer nor a typed array";
      JS_FreeValue(ctx, buf);
      JS_FreeValue(ctx, result);
      return out;
    }
    data = JS_GetArrayBuffer(ctx, &size, buf);
    JS_FreeValue(ctx, buf);
  }
  if (data && size > 0) out.assign(data, data + size);
  JS_FreeValue(ctx, result);
  return out;
}

} // namespace polyregion::polypass
