#include "polypass_locate.h"

#include <cstdlib>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "generated/polypass_symbols.h"

namespace polyregion::polypass {

using namespace aspartame;

namespace {

const char locateAnchor = 0;

#if defined(_WIN32)
constexpr char PathSep = ';';
constexpr auto DsoSuffix = ".dll";
constexpr auto DsoBasename = "libpolypass.dll";
#elif defined(__APPLE__)
constexpr char PathSep = ':';
constexpr auto DsoSuffix = ".dylib";
constexpr auto DsoBasename = "libpolypass.dylib";
#else
constexpr char PathSep = ':';
constexpr auto DsoSuffix = ".so";
constexpr auto DsoBasename = "libpolypass.so";
#endif

constexpr auto JsBasename = "polypass.js";

String resolveSymlink(String candidate) {
  llvm::SmallString<256> resolved;
  if (!llvm::sys::fs::real_path(candidate, resolved)) return resolved.str().str();
  return candidate;
}

String joinPath(llvm::StringRef dir, std::initializer_list<llvm::StringRef> parts) {
  llvm::SmallString<256> c(dir);
  for (auto p : parts)
    llvm::sys::path::append(c, p);
  return c.str().str();
}

String findBundledPlugin() {
  namespace fs = llvm::sys::fs;
  namespace path = llvm::sys::path;
  const auto exe = fs::getMainExecutable(nullptr, reinterpret_cast<void *>(const_cast<char *>(&locateAnchor)));
  String dsoBeside, dsoLib, jsBeside, jsLib;
  if (!exe.empty()) {
    const auto dir = path::parent_path(exe);
    dsoBeside = joinPath(dir, {DsoBasename});
    dsoLib = joinPath(dir, {"../lib", DsoBasename});
    jsBeside = joinPath(dir, {JsBasename});
    jsLib = joinPath(dir, {"../lib", JsBasename});
  }
  const std::array candidates = {
      dsoBeside,
      dsoLib,
#ifdef POLYPASS_DSO_DEV_PATH
      String(POLYPASS_DSO_DEV_PATH),
#else
      String(),
#endif
      jsBeside,
      jsLib,
#ifdef POLYPASS_JS_DEV_PATH
      String(POLYPASS_JS_DEV_PATH),
#else
      String(),
#endif
  };
  return candidates | find([&](const auto &c) { return !c.empty() && fs::exists(c); }) | map(resolveSymlink) | get_or_else(String{});
}

} // namespace

std::optional<PluginKind> pluginKindFor(std::string_view path) {
  if (path.ends_with(".js")) return PluginKind::Js;
  if (path.ends_with(DsoSuffix)) return PluginKind::Dso;
  return std::nullopt;
}

Vector<PluginRef> resolvePlugins(String &error) {
  namespace fs = llvm::sys::fs;

  if (const char *envList = std::getenv(abi::EnvPlugins); envList && *envList) {

    const auto paths = (String(envList) ^ split(PathSep)) | filter([](const auto &e) { return !e.empty(); }) | to_vector();
    if (paths.empty()) {
      error = std::string(abi::EnvPlugins) + " set but empty after splitting";
      return {};
    }
    const auto out = paths ^ traverse([&](const auto &path) -> std::optional<PluginRef> {
                       if (!fs::exists(path)) {
                         error = std::string(abi::EnvPlugins) + ": missing file " + path;
                         return std::nullopt;
                       }
                       const auto kind = pluginKindFor(path);
                       if (!kind) {
                         error = std::string(abi::EnvPlugins) + ": unrecognised extension " + path;
                         return std::nullopt;
                       }
                       return PluginRef{resolveSymlink(path), *kind};
                     });
    return out ^ get_or_else(Vector<PluginRef>{});
  }

  const auto bundled = findBundledPlugin();
  if (bundled.empty()) {
    error = std::string("no polypass plugin found (set $") + abi::EnvPlugins + " or install the dist)";
    return {};
  }
  const auto kind = pluginKindFor(bundled);
  if (!kind) {
    error = "bundled plugin has unrecognised extension: " + bundled;
    return {};
  }
  return {{bundled, *kind}};
}

} // namespace polyregion::polypass
