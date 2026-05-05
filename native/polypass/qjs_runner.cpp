#include "qjs_runner.h"

#include "fmt/format.h"
#include "quickjs.h"

namespace polyregion::polypass {

namespace {

// Convert a JS exception attached to the given context into a printable message.
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
  // GLOBAL eval rather than MODULE so the Scala.js bundle's IIFE-style top level can attach
  // exports to globalThis. If/when we switch to ES modules emitted by Scala.js, change this to
  // JS_EVAL_TYPE_MODULE and grab the exports via the returned promise.
  JSValue r = JS_Eval(ctx, source.data(), source.size(), String(moduleId).c_str(), JS_EVAL_TYPE_GLOBAL);
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
  // ArrayBuffer is the cheapest binary handoff; the Scala.js side wraps it in Uint8Array if needed.
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
    // Maybe the JS returned a Uint8Array; pull the underlying buffer.
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
