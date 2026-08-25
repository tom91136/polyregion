#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/io.hpp"
#include "polyregion/polypackage_symbols.h"
#include "polyregion/polypass.h"

#include "generated/polypass_symbols.h"
#include "js_cache.h"
#include "js_runner.h"
#include <hermes/BCGen/HBC/BytecodeVersion.h>
#include <hermes/CompileJS.h>
#include <hermes/hermes.h>
#include <jsi/jsi.h>

using namespace aspartame;

namespace polyregion::polypass {

namespace jsi = facebook::jsi;
namespace fbhm = facebook::hermes;

struct JsPassRunner::Impl {
  std::string path;
  std::string tag;
  std::unique_ptr<jsi::Runtime> runtime;
  Vector<String> names;
  std::unordered_map<String, String> descrs;
  bool loaded = false;

  String loadFromSource(std::string_view source, std::string_view moduleId);
  String enumeratePasses();
};

namespace {

jsi::Function makeConsoleWrite(jsi::Runtime &rt, const char *name) {
  return jsi::Function::createFromHostFunction( //
      rt, jsi::PropNameID::forAscii(rt, name), 1, [](jsi::Runtime &r, const jsi::Value &, const jsi::Value *args, size_t n) {
        for (size_t i = 0; i < n; ++i) {
          if (i != 0) std::fputc(' ', stderr);
          std::fputs(args[i].toString(r).utf8(r).c_str(), stderr);
        }
        std::fputc('\n', stderr);
        return jsi::Value::undefined();
      });
}

class StringBuffer final : public jsi::Buffer {
public:
  explicit StringBuffer(std::string s) : s_(std::move(s)) {}
  size_t size() const override { return s_.size(); }
  const uint8_t *data() const override { return reinterpret_cast<const uint8_t *>(s_.data()); }

private:
  std::string s_;
};

class VectorBuffer final : public jsi::Buffer {
public:
  explicit VectorBuffer(Vector<uint8_t> v) : v_(std::move(v)) {}
  size_t size() const override { return v_.size(); }
  const uint8_t *data() const override { return v_.data(); }

private:
  Vector<uint8_t> v_;
};

String hermesCacheTag() { return fmt::format("hermes-bc{}-{}", hermes::hbc::BYTECODE_VERSION, hostArchTag()); }

String formatException(const jsi::JSError &e) {
  String out = e.getMessage();
  if (const auto &stack = e.getStack(); !stack.empty() && stack != "No stack") {
    out += "\n";
    out += stack;
  }
  return out;
}

jsi::Value getExport(jsi::Runtime &rt, const char *name) {
  auto exports = rt.global().getProperty(rt, "exports");
  if (!exports.isObject()) return jsi::Value::undefined();
  return exports.asObject(rt).getProperty(rt, name);
}

bool hasExport(jsi::Runtime &rt, const char *name) { return !getExport(rt, name).isUndefined(); }

bool hasFunctionExport(jsi::Runtime &rt, const char *name) {
  auto value = getExport(rt, name);
  return value.isObject() && value.asObject(rt).isFunction(rt);
}

// Invoke `exports.<name>()` and return the result, or undefined on missing.
jsi::Value callExport(jsi::Runtime &rt, const char *name) {
  auto fnVal = getExport(rt, name);
  if (!fnVal.isObject()) return jsi::Value::undefined();
  auto obj = fnVal.asObject(rt);
  if (!obj.isFunction(rt)) return jsi::Value::undefined();
  return obj.asFunction(rt).call(rt);
}

} // namespace

JsPassRunner::JsPassRunner() : impl(std::make_unique<Impl>()) {
  impl->runtime = fbhm::makeHermesRuntime();
  auto &rt = *impl->runtime;
  auto g = rt.global();
  auto console = jsi::Object(rt);
  for (const char *m : {"log", "error", "warn", "info", "debug"})
    console.setProperty(rt, m, makeConsoleWrite(rt, m));
  g.setProperty(rt, "console", console);
}

JsPassRunner::JsPassRunner(std::string path) : JsPassRunner() {
  impl->path = std::move(path);
  impl->tag = "PolyPass[" + impl->path + "]";
}

JsPassRunner::~JsPassRunner() = default;

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

std::optional<int32_t> JsPassRunner::intExport(const std::string_view name, String &error) const {
  if (!impl->loaded) {
    error = "JS plugin not loaded";
    return std::nullopt;
  }
  auto &rt = *impl->runtime;
  try {
    const String exportName(name);
    auto value = callExport(rt, exportName.c_str());
    if (!value.isNumber()) {
      error = fmt::format("PolyPass JS: exports.{} returned a non-integer", name);
      return std::nullopt;
    }
    return static_cast<int32_t>(value.asNumber());
  } catch (const jsi::JSError &e) {
    error = formatException(e);
    return std::nullopt;
  } catch (const std::exception &e) {
    error = String("intExport: ") + e.what();
    return std::nullopt;
  }
}

Vector<uint8_t> JsPassRunner::bytesExport(const std::string_view name, const Vector<uint8_t> &input, String &error) {
  if (!impl->loaded) {
    error = "JS plugin not loaded";
    return {};
  }
  auto &rt = *impl->runtime;
  Vector<uint8_t> out;
  try {
    const String exportName(name);
    auto fnValue = getExport(rt, exportName.c_str());
    if (!fnValue.isObject() || !fnValue.asObject(rt).isFunction(rt)) {
      error = fmt::format("PolyPass JS: exports.{} is not a function", name);
      return out;
    }
    auto fn = fnValue.asObject(rt).asFunction(rt);
    auto ctor = rt.global().getPropertyAsFunction(rt, "ArrayBuffer");
    auto bufferValue = ctor.callAsConstructor(rt, {jsi::Value(static_cast<double>(input.size()))});
    auto buffer = bufferValue.asObject(rt).getArrayBuffer(rt);
    if (!input.empty()) std::memcpy(buffer.data(rt), input.data(), input.size());
    auto uint8Ctor = rt.global().getPropertyAsFunction(rt, "Uint8Array");
    auto uint8 = uint8Ctor.callAsConstructor(rt, {jsi::Value(rt, bufferValue)});
    auto result = fn.call(rt, {jsi::Value(rt, uint8)});
    if (!result.isObject()) {
      error = fmt::format("{}: return value must be Uint8Array", name);
      return out;
    }
    auto resultObject = result.asObject(rt);
    auto resultBufferValue = resultObject.getProperty(rt, "buffer");
    if (!resultBufferValue.isObject() || !resultBufferValue.asObject(rt).isArrayBuffer(rt)) {
      error = fmt::format("{}: return value must be Uint8Array", name);
      return out;
    }
    auto resultBuffer = resultBufferValue.asObject(rt).getArrayBuffer(rt);
    const auto offset = static_cast<size_t>(resultObject.getProperty(rt, "byteOffset").asNumber());
    const auto length = static_cast<size_t>(resultObject.getProperty(rt, "byteLength").asNumber());
    const auto capacity = resultBuffer.size(rt);
    if (offset > capacity || length > capacity - offset) {
      error = fmt::format("{}: typed-array bounds out of range", name);
      return out;
    }
    if (length > 0) {
      auto *base = resultBuffer.data(rt) + offset;
      out.assign(base, base + length);
    }
    return out;
  } catch (const jsi::JSError &e) {
    error = formatException(e);
    return out;
  } catch (const std::exception &e) {
    error = String("bytesExport: ") + e.what();
    return out;
  }
}

std::optional<uint32_t> JsPassRunner::packageAbiVersion() const {
  String error;
  const auto version = intExport(polypackage::abi::AbiVersion, error);
  if (!version || !error.empty()) return std::nullopt;
  return static_cast<uint32_t>(*version);
}

Vector<uint8_t> JsPassRunner::runPackage(const std::string_view operation, const Vector<uint8_t> &request, String &error) {
  if (operation != polypackage::abi::LinkPackage && operation != polypackage::abi::ResolveSym) {
    error = fmt::format("unknown package operation {}", operation);
    return {};
  }
  return bytesExport(operation, request, error);
}

String JsPassRunner::Impl::loadFromSource(std::string_view source, std::string_view moduleId) {
  auto &rt = *runtime;
  try {
    rt.global().setProperty(rt, "exports", jsi::Object(rt));

    const auto tag = hermesCacheTag();

    if (auto cached = readJsCache(tag, source)) {
      try {
        auto buf = std::make_shared<VectorBuffer>(std::move(*cached));
        auto prepared = rt.prepareJavaScript(buf, String(moduleId));
        rt.evaluatePreparedJavaScript(prepared);
        return {};
      } catch (const jsi::JSError &) {
        // Cached HBC failed (corruption or intra-tag skew); recompile.
      }
    }

    std::string bc;
    if (!hermes::compileJS(String(source), String(moduleId), bc, /*optimize=*/true)) return "hermes::compileJS failed (syntax error)";
    writeJsCache(tag, source, reinterpret_cast<const uint8_t *>(bc.data()), bc.size());
    auto buf = std::make_shared<StringBuffer>(std::move(bc));
    auto prepared = rt.prepareJavaScript(buf, String(moduleId));
    rt.evaluatePreparedJavaScript(prepared);
    return {};
  } catch (const jsi::JSError &e) {
    return formatException(e);
  } catch (const std::exception &e) {
    return String("loadFromSource: ") + e.what();
  }
}

String JsPassRunner::Impl::enumeratePasses() {
  auto &rt = *runtime;
  try {
    const bool anyPass = hasExport(rt, abi::AbiVersion) || hasExport(rt, abi::PassCount) || hasExport(rt, abi::PassName)
                         || hasExport(rt, abi::PassDescr) || hasExport(rt, abi::RunPasses);
    const bool completePass = hasFunctionExport(rt, abi::AbiVersion) && hasFunctionExport(rt, abi::PassCount)
                              && hasFunctionExport(rt, abi::PassName) && hasFunctionExport(rt, abi::RunPasses);
    const bool anyPackage = hasExport(rt, polypackage::abi::AbiVersion) || hasExport(rt, polypackage::abi::LinkPackage)
                            || hasExport(rt, polypackage::abi::ResolveSym);
    const bool completePackage = hasFunctionExport(rt, polypackage::abi::AbiVersion) && hasFunctionExport(rt, polypackage::abi::LinkPackage)
                                 && hasFunctionExport(rt, polypackage::abi::ResolveSym);
    if (anyPass && !completePass) return "PolyPass JS: incomplete pass capability";
    if (anyPackage && !completePackage) return "PolyPass JS: incomplete package capability";
    if (!completePass && !completePackage) return "PolyPass JS: bundle exposes neither pass nor package capability";
    if (!completePass) return {};

    auto abiVal = callExport(rt, abi::AbiVersion);
    if (!abiVal.isNumber()) return fmt::format("PolyPass JS: missing or non-numeric exports.{}()", abi::AbiVersion);
    const auto reported = static_cast<int64_t>(abiVal.asNumber());
    if (reported < 0 || static_cast<uint32_t>(reported) != POLYPASS_ABI_VERSION)
      return fmt::format("PolyPass ABI mismatch: plugin={} polyc={}", reported, POLYPASS_ABI_VERSION);

    auto countVal = callExport(rt, abi::PassCount);
    if (!countVal.isNumber()) return fmt::format("PolyPass JS: missing or non-numeric exports.{}()", abi::PassCount);
    const auto count = static_cast<int32_t>(countVal.asNumber());

    auto nameFnVal = getExport(rt, abi::PassName);
    if (!nameFnVal.isObject()) return fmt::format("PolyPass JS: exports.{} is not a function", abi::PassName);
    auto nameObj = nameFnVal.asObject(rt);
    if (!nameObj.isFunction(rt)) return fmt::format("PolyPass JS: exports.{} is not a function", abi::PassName);
    auto nameFn = nameObj.asFunction(rt);

    auto descrFnVal = getExport(rt, abi::PassDescr);
    std::optional<jsi::Function> descrFn;
    if (descrFnVal.isObject()) {
      auto descrObj = descrFnVal.asObject(rt);
      if (descrObj.isFunction(rt)) descrFn = descrObj.asFunction(rt);
    }

    names.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
      auto rn = nameFn.call(rt, {jsi::Value(static_cast<double>(i))});
      if (!rn.isString()) return fmt::format("PolyPass JS: {}({}) returned non-string", abi::PassName, i);
      String nm = rn.asString(rt).utf8(rt);
      if (descrFn) {
        auto rd = descrFn->call(rt, {jsi::Value(static_cast<double>(i))});
        if (rd.isString()) {
          String ds = rd.asString(rt).utf8(rt);
          if (!ds.empty()) descrs.emplace(nm, std::move(ds));
        }
      }
      names.emplace_back(std::move(nm));
    }
    return {};
  } catch (const jsi::JSError &e) {
    return formatException(e);
  } catch (const std::exception &e) {
    return String("enumeratePasses: ") + e.what();
  }
}

const Vector<String> &JsPassRunner::passNames() const { return impl->names; }

std::optional<String> JsPassRunner::passDescr(std::string_view name) const {
  const auto it = impl->descrs.find(String(name));
  if (it == impl->descrs.end()) return std::nullopt;
  return it->second;
}

Vector<uint8_t> JsPassRunner::runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) {
  if (!impl->loaded) {
    error = "JS plugin not loaded";
    return {};
  }
  auto &rt = *impl->runtime;
  Vector<uint8_t> out;
  try {
    auto runFnVal = getExport(rt, abi::RunPasses);
    if (!runFnVal.isObject() || !runFnVal.asObject(rt).isFunction(rt))
      throw jsi::JSError(rt, fmt::format("PolyPass JS: exports.{} is not a function", abi::RunPasses));
    auto runFn = runFnVal.asObject(rt).asFunction(rt);

    auto stepsArr = jsi::Array(rt, steps.size());
    steps                                              //
        | zip_with_index<size_t>()                     //
        | for_each([&](const auto &s, const auto &i) { //
            stepsArr.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, s));
          });

    auto ctor = rt.global().getPropertyAsFunction(rt, "ArrayBuffer");
    auto abVal = ctor.callAsConstructor(rt, {jsi::Value(static_cast<double>(programBytes.size()))});
    auto ab = abVal.asObject(rt).getArrayBuffer(rt);
    if (!programBytes.empty()) std::memcpy(ab.data(rt), programBytes.data(), programBytes.size());
    auto u8Ctor = rt.global().getPropertyAsFunction(rt, "Uint8Array");
    auto u8 = u8Ctor.callAsConstructor(rt, {jsi::Value(rt, abVal)});

    jsi::Value result = runFn.call(rt, {jsi::Value(rt, stepsArr), jsi::Value(rt, u8)});

    if (!result.isObject()) {
      error = "runPasses: return value must be Uint8Array";
      return out;
    }
    auto resultObj = result.asObject(rt);
    auto bufProp = resultObj.getProperty(rt, "buffer");
    if (!bufProp.isObject() || !bufProp.asObject(rt).isArrayBuffer(rt)) {
      error = "runPasses: return value must be Uint8Array";
      return out;
    }
    auto resAb = bufProp.asObject(rt).getArrayBuffer(rt);
    size_t off = static_cast<size_t>(resultObj.getProperty(rt, "byteOffset").asNumber());
    size_t len = static_cast<size_t>(resultObj.getProperty(rt, "byteLength").asNumber());
    const size_t cap = resAb.size(rt);
    if (off > cap || len > cap - off) {
      error = "runPasses: typed-array bounds out of range";
      return out;
    }
    if (len > 0) {
      auto *base = resAb.data(rt) + off;
      out.assign(base, base + len);
    }
    return out;
  } catch (const jsi::JSError &e) {
    error = formatException(e);
    return out;
  } catch (const std::exception &e) {
    error = String("runPasses: ") + e.what();
    return out;
  }
}

std::string_view JsPassRunner::tag() const { return impl->tag; }

} // namespace polyregion::polypass
