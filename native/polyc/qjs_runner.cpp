#include "qjs_runner.h"

#include <cstdio>

#include "quickjs.h"

namespace polyregion::polypass {

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

JSValue consolePrint(JSContext *ctx, JSValueConst /*this_val*/, int argc, JSValueConst *argv, FILE *stream) {
  for (int i = 0; i < argc; ++i) {
    if (i != 0) std::fputc(' ', stream);
    if (const char *s = JS_ToCString(ctx, argv[i])) {
      std::fputs(s, stream);
      JS_FreeCString(ctx, s);
    }
  }
  std::fputc('\n', stream);
  return JS_UNDEFINED;
}
JSValue consoleLog(JSContext *ctx, JSValueConst t, int argc, JSValueConst *argv) { return consolePrint(ctx, t, argc, argv, stderr); }
JSValue consoleErr(JSContext *ctx, JSValueConst t, int argc, JSValueConst *argv) { return consolePrint(ctx, t, argc, argv, stderr); }

void noopFreeArrayBuffer(JSRuntime * /*rt*/, void * /*opaque*/, void * /*ptr*/) {}

} // namespace

JsPassRunner::JsPassRunner() {
  rt = JS_NewRuntime();
  ctx = JS_NewContext(rt);
  // Scala.js `println` lowers to `console.log`; without a binding QuickJS sees an undefined
  // global and fails silently. Wire log/error/warn/info/debug to stderr so pass-side tree
  // logs surface alongside polyc's own diagnostics.
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue console = JS_NewObject(ctx);
  JS_SetPropertyStr(ctx, console, "log", JS_NewCFunction(ctx, consoleLog, "log", 1));
  JS_SetPropertyStr(ctx, console, "error", JS_NewCFunction(ctx, consoleErr, "error", 1));
  JS_SetPropertyStr(ctx, console, "warn", JS_NewCFunction(ctx, consoleErr, "warn", 1));
  JS_SetPropertyStr(ctx, console, "info", JS_NewCFunction(ctx, consoleLog, "info", 1));
  JS_SetPropertyStr(ctx, console, "debug", JS_NewCFunction(ctx, consoleLog, "debug", 1));
  JS_SetPropertyStr(ctx, global, "console", console);
  JS_FreeValue(ctx, global);
}

JsPassRunner::~JsPassRunner() {
  if (ctx) JS_FreeContext(ctx);
  if (rt) JS_FreeRuntime(rt);
}

String JsPassRunner::loadModule(std::string_view source, std::string_view moduleId) {
  // Scala.js CommonJSModule emits `exports.runPass = ...`; install a fresh `exports` object so
  // reloading the bundle does not retain stale exports.
  JSValue global = JS_GetGlobalObject(ctx);
  JS_SetPropertyStr(ctx, global, "exports", JS_NewObject(ctx));
  JS_FreeValue(ctx, global);

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
  JSValue exports = JS_GetPropertyStr(ctx, global, "exports");
  JSValue runPass = JS_GetPropertyStr(ctx, exports, "runPass");
  JSValueConst thisVal = exports;
  if (!JS_IsFunction(ctx, runPass)) {
    JS_FreeValue(ctx, runPass);
    runPass = JS_GetPropertyStr(ctx, global, "runPass");
    thisVal = global;
  }
  if (!JS_IsFunction(ctx, runPass)) {
    error = "runPass: JS bundle does not export `exports.runPass`";
    JS_FreeValue(ctx, runPass);
    JS_FreeValue(ctx, exports);
    JS_FreeValue(ctx, global);
    return out;
  }

  JSValue nameVal = JS_NewStringLen(ctx, passName.data(), passName.size());
  static uint8_t empty = 0;
  auto *inputData = programBytes.empty() ? &empty : const_cast<uint8_t *>(programBytes.data());
  // Pass an aliasing free_func so QuickJS does not memcpy programBytes into a fresh ArrayBuffer.
  // Safe because runPass is synchronous and the input is freed only after JS_Call returns.
  JSValue bufVal = JS_NewUint8Array(ctx, inputData, programBytes.size(), noopFreeArrayBuffer, nullptr, false);
  if (JS_IsException(bufVal)) {
    error = formatException(ctx);
    JS_FreeValue(ctx, nameVal);
    JS_FreeValue(ctx, runPass);
    JS_FreeValue(ctx, exports);
    JS_FreeValue(ctx, global);
    return out;
  }

  JSValue argv[2] = {nameVal, bufVal};
  JSValue result = JS_Call(ctx, runPass, thisVal, 2, argv);
  JS_FreeValue(ctx, nameVal);
  JS_FreeValue(ctx, bufVal);
  JS_FreeValue(ctx, runPass);
  JS_FreeValue(ctx, exports);
  JS_FreeValue(ctx, global);

  if (JS_IsException(result)) {
    error = formatException(ctx);
    JS_FreeValue(ctx, result);
    return out;
  }

  if (JS_GetTypedArrayType(result) != JS_TYPED_ARRAY_UINT8) {
    error = "runPass: return value must be Uint8Array";
    JS_FreeValue(ctx, result);
    return out;
  }
  size_t size = 0;
  uint8_t *data = JS_GetUint8Array(ctx, &size, result);
  if (!data && size != 0) error = "runPass: failed to read Uint8Array result";
  if (data && size > 0) out.assign(data, data + size);
  JS_FreeValue(ctx, result);
  return out;
}

} // namespace polyregion::polypass
