#include <cstdio>
#include <unordered_map>

#include "fmt/format.h"
#include "quickjs.h"

#include "polyregion/io.hpp"
#include "polyregion/polypass.h"

#include "generated/polypass_symbols.h"
#include "js_cache.h"
#include "js_runner.h"

namespace polyregion::polypass {

struct JsPassRunner::Impl {
  std::string path;
  std::string tag;
  JSRuntime *rt = nullptr;
  JSContext *ctx = nullptr;
  Vector<String> names;
  std::unordered_map<String, String> descrs;
  bool loaded = false;

  String loadFromSource(std::string_view source, std::string_view moduleId);
  String enumeratePasses();
};

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

JSValue consoleWrite(JSContext *ctx, JSValueConst /*this_val*/, int argc, JSValueConst *argv) {
  for (int i = 0; i < argc; ++i) {
    if (i != 0) std::fputc(' ', stderr);
    if (const char *s = JS_ToCString(ctx, argv[i])) {
      std::fputs(s, stderr);
      JS_FreeCString(ctx, s);
    }
  }
  std::fputc('\n', stderr);
  return JS_UNDEFINED;
}

void noopFreeArrayBuffer(JSRuntime * /*rt*/, void * /*opaque*/, void * /*ptr*/) {}

String qjsCacheTag() { return fmt::format("qjs-{}.{}.{}-{}", QJS_VERSION_MAJOR, QJS_VERSION_MINOR, QJS_VERSION_PATCH, hostArchTag()); }

JSValue getExportFn(JSContext *ctx, const char *name) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue exports = JS_GetPropertyStr(ctx, global, "exports");
  JSValue fn = JS_GetPropertyStr(ctx, exports, name);
  JS_FreeValue(ctx, exports);
  JS_FreeValue(ctx, global);
  return fn;
}

int32_t callExportInt(JSContext *ctx, const char *name) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue exports = JS_GetPropertyStr(ctx, global, "exports");
  JSValue fn = JS_GetPropertyStr(ctx, exports, name);
  int32_t out = -1;
  if (JS_IsFunction(ctx, fn)) {
    JSValue r = JS_Call(ctx, fn, exports, 0, nullptr);
    if (JS_IsNumber(r)) JS_ToInt32(ctx, &out, r);
    JS_FreeValue(ctx, r);
  }
  JS_FreeValue(ctx, fn);
  JS_FreeValue(ctx, exports);
  JS_FreeValue(ctx, global);
  return out;
}

} // namespace

JsPassRunner::JsPassRunner() : impl(std::make_unique<Impl>()) {
  impl->rt = JS_NewRuntime();
  impl->ctx = JS_NewContext(impl->rt);
  auto *ctx = impl->ctx;
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue console = JS_NewObject(ctx);
  for (const char *m : {"log", "error", "warn", "info", "debug"})
    JS_SetPropertyStr(ctx, console, m, JS_NewCFunction(ctx, consoleWrite, m, 1));
  JS_SetPropertyStr(ctx, global, "console", console);
  JS_FreeValue(ctx, global);
}

JsPassRunner::JsPassRunner(std::string path) : JsPassRunner() {
  impl->path = std::move(path);
  impl->tag = "PolyPass[" + impl->path + "]";
}

JsPassRunner::~JsPassRunner() {
  if (impl->ctx) JS_FreeContext(impl->ctx);
  if (impl->rt) JS_FreeRuntime(impl->rt);
}

String JsPassRunner::load() {
  if (impl->loaded) return {};
  if (impl->path.empty()) return "JS plugin path not set";
  const auto source = polyregion::read_string(impl->path);
  if (source.empty()) return fmt::format("PolyPass JS: failed to read {}", impl->path);
  if (auto err = impl->loadFromSource(source, impl->path); !err.empty()) return err;
  if (auto err = impl->enumeratePasses(); !err.empty()) return err;
  impl->loaded = true;
  return {};
}

String JsPassRunner::loadModule(std::string_view source) {
  if (auto err = impl->loadFromSource(source, "<inline>"); !err.empty()) return err;
  if (auto err = impl->enumeratePasses(); !err.empty()) return err;
  impl->loaded = true;
  return {};
}

String JsPassRunner::Impl::loadFromSource(std::string_view source, std::string_view moduleId) {
  JSValue global = JS_GetGlobalObject(ctx);
  JS_SetPropertyStr(ctx, global, "exports", JS_NewObject(ctx));
  JS_FreeValue(ctx, global);

  const auto tag = qjsCacheTag();

  // Cached HBC may fail (corruption or intra-tag skew); fall through to recompile.
  if (auto cached = readJsCache(tag, source)) {
    JSValue obj = JS_ReadObject(ctx, cached->data(), cached->size(), JS_READ_OBJ_BYTECODE);
    if (!JS_IsException(obj)) {
      JSValue r = JS_EvalFunction(ctx, obj);
      if (!JS_IsException(r)) {
        JS_FreeValue(ctx, r);
        return {};
      }
      JS_FreeValue(ctx, r);
    } else {
      JS_FreeValue(ctx, obj);
    }
    JS_FreeValue(ctx, JS_GetException(ctx));
  }

  const String moduleIdStr(moduleId);
  JSValue obj = JS_Eval(ctx, source.data(), source.size(), moduleIdStr.c_str(), JS_EVAL_TYPE_GLOBAL | JS_EVAL_FLAG_COMPILE_ONLY);
  if (JS_IsException(obj)) {
    JS_FreeValue(ctx, obj);
    return formatException(ctx);
  }
  size_t bcSize = 0;
  if (uint8_t *bc = JS_WriteObject(ctx, &bcSize, obj, JS_WRITE_OBJ_BYTECODE | JS_WRITE_OBJ_REFERENCE)) {
    writeJsCache(tag, source, bc, bcSize);
    js_free(ctx, bc);
  }
  JSValue r = JS_EvalFunction(ctx, obj);
  if (JS_IsException(r)) {
    JS_FreeValue(ctx, r);
    return formatException(ctx);
  }
  JS_FreeValue(ctx, r);
  return {};
}

String JsPassRunner::Impl::enumeratePasses() {
  const int32_t reported = callExportInt(ctx, abi::AbiVersion);
  if (reported < 0) return fmt::format("PolyPass JS: missing or non-numeric exports.{}()", abi::AbiVersion);
  if (static_cast<uint32_t>(reported) != POLYPASS_ABI_VERSION)
    return fmt::format("PolyPass ABI mismatch: plugin={} polyc={}", reported, POLYPASS_ABI_VERSION);

  JSValue nameFn = getExportFn(ctx, abi::PassName);
  JSValue descrFn = getExportFn(ctx, abi::PassDescr);
  if (!JS_IsFunction(ctx, nameFn)) {
    JS_FreeValue(ctx, nameFn);
    JS_FreeValue(ctx, descrFn);
    return fmt::format("PolyPass JS: exports.{} is not a function", abi::PassName);
  }
  const int32_t count = callExportInt(ctx, abi::PassCount);
  if (count < 0) {
    JS_FreeValue(ctx, nameFn);
    JS_FreeValue(ctx, descrFn);
    return fmt::format("PolyPass JS: missing or non-numeric exports.{}", abi::PassCount);
  }
  names.reserve(count);
  JSValue global = JS_GetGlobalObject(ctx);
  for (int32_t i = 0; i < count; ++i) {
    JSValue arg = JS_NewInt32(ctx, i);
    JSValue rn = JS_Call(ctx, nameFn, global, 1, &arg);
    if (JS_IsException(rn) || !JS_IsString(rn)) {
      if (JS_IsException(rn)) JS_FreeValue(ctx, JS_GetException(ctx));
      JS_FreeValue(ctx, arg);
      JS_FreeValue(ctx, rn);
      JS_FreeValue(ctx, nameFn);
      JS_FreeValue(ctx, descrFn);
      JS_FreeValue(ctx, global);
      return fmt::format("PolyPass JS: {}({}) returned non-string", abi::PassName, i);
    }
    const char *cs = JS_ToCString(ctx, rn);
    String nm = cs ? cs : "";
    if (cs) JS_FreeCString(ctx, cs);
    JS_FreeValue(ctx, rn);

    if (JS_IsFunction(ctx, descrFn)) {
      JSValue rd = JS_Call(ctx, descrFn, global, 1, &arg);
      if (JS_IsException(rd)) JS_FreeValue(ctx, JS_GetException(ctx));
      else if (JS_IsString(rd)) {
        if (const char *ds = JS_ToCString(ctx, rd); ds && *ds) {
          descrs.emplace(nm, ds);
          JS_FreeCString(ctx, ds);
        } else if (ds) {
          JS_FreeCString(ctx, ds);
        }
      }
      JS_FreeValue(ctx, rd);
    }
    JS_FreeValue(ctx, arg);
    names.emplace_back(std::move(nm));
  }
  JS_FreeValue(ctx, global);
  JS_FreeValue(ctx, nameFn);
  JS_FreeValue(ctx, descrFn);
  return {};
}

const Vector<String> &JsPassRunner::passNames() const { return impl->names; }

std::optional<String> JsPassRunner::passDescr(std::string_view name) const { return impl->descrs ^ get_maybe(String(name)); }

Vector<uint8_t> JsPassRunner::runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) {
  if (!impl->loaded) {
    error = "JS plugin not loaded";
    return {};
  }
  auto *ctx = impl->ctx;
  Vector<uint8_t> out;

  JSValue runFn = getExportFn(ctx, abi::RunPasses);
  if (!JS_IsFunction(ctx, runFn)) {
    error = fmt::format("PolyPass JS: exports.{} is not a function", abi::RunPasses);
    JS_FreeValue(ctx, runFn);
    return out;
  }

  JSValue stepsArr = JS_NewArray(ctx);
  steps | zip_with_index<uint32_t>() | for_each([&](auto &s, auto i) { //
    JS_SetPropertyUint32(ctx, stepsArr, i, JS_NewStringLen(ctx, s.data(), s.size()));
  });

  static uint8_t empty = 0;
  auto *inputData = programBytes.empty() ? &empty : const_cast<uint8_t *>(programBytes.data());
  JSValue bufVal = JS_NewUint8Array(ctx, inputData, programBytes.size(), noopFreeArrayBuffer, nullptr, false);
  if (JS_IsException(bufVal)) {
    error = formatException(ctx);
    JS_FreeValue(ctx, stepsArr);
    JS_FreeValue(ctx, runFn);
    return out;
  }

  JSValue argv[2] = {stepsArr, bufVal};
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue result = JS_Call(ctx, runFn, global, 2, argv);
  JS_FreeValue(ctx, global);
  JS_FreeValue(ctx, stepsArr);
  JS_FreeValue(ctx, bufVal);
  JS_FreeValue(ctx, runFn);

  if (JS_IsException(result)) {
    error = formatException(ctx);
    JS_FreeValue(ctx, result);
    return out;
  }
  if (JS_GetTypedArrayType(result) != JS_TYPED_ARRAY_UINT8) {
    error = "runPasses: return value must be Uint8Array";
    JS_FreeValue(ctx, result);
    return out;
  }
  size_t size = 0;
  uint8_t *data = JS_GetUint8Array(ctx, &size, result);
  if (!data && size != 0) error = "runPasses: failed to read Uint8Array result";
  if (data && size > 0) out.assign(data, data + size);
  JS_FreeValue(ctx, result);
  return out;
}

std::string_view JsPassRunner::tag() const { return impl->tag; }

} // namespace polyregion::polypass
