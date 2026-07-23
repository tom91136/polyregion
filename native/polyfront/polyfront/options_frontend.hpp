#pragma once

#include <cstdlib>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "options.hpp"

namespace polyregion::polyfront {

using namespace aspartame;

template <typename... S> inline std::string joinPath(llvm::StringRef x, const S &...xs) {
  llvm::SmallString<256> p(x);
  auto appendOne = [&](llvm::StringRef s) {
    // XXX skip empties: llvm::sys::path::append unconditionally inserts a separator before each non-null Twine.
    if (!s.empty()) llvm::sys::path::append(p, s);
  };
  (appendOne(xs), ...);
  return p.str().str();
}

inline std::string executableName(llvm::StringRef name) {
#if defined(_WIN32)
  return name.str() + ".exe";
#else
  return name.str();
#endif
}

inline std::string staticLibraryName(llvm::StringRef name) {
#if defined(_WIN32)
  return name.str() + ".lib";
#else
  return "lib" + name.str() + ".a";
#endif
}

[[maybe_unused]] inline void addrAnchor() { /* anchor for llvm::sys::fs::getMainExecutable */ }

// Resolves ${execParent}/lib/<name>, falling back to ${execParent}/../lib/<name>.
inline std::string resolveResourcePath(const std::string &execParentPath, const std::string &resourceName) {
  auto p = joinPath(execParentPath, "lib", resourceName);
  if (!llvm::sys::fs::exists(p)) p = joinPath(execParentPath, "..", "lib", resourceName);
  return p;
}

// XXX clang aliases -lstdc++ to libc++ on macOS and stamps @rpath/libc++.1.dylib; the
// bundled libc++ sits in the dist's main lib/, not under polyfc/lib or polycpp/lib.
inline std::vector<std::string> appleDistLibcxxRpath(const std::string &execParentPath) {
#if defined(__APPLE__)
  return {fmt::format("-Wl,-rpath,{}", joinPath(execParentPath, "..", "lib"))};
#else
  (void)execParentPath;
  return {};
#endif
}

// XXX macOS: static-link our libc++ so we can see into the operators symbols; Apple's libc++ is a
// prebuilt dylib and there is no static system libc++ to relink
inline std::vector<std::string> appleDistLibcxxStatic(const std::string &distLibDir) {
#if defined(__APPLE__)
  return {"-nostdlib++", joinPath(distLibDir, "libc++.a"), joinPath(distLibDir, "libc++abi.a")};
#else
  (void)distLibDir;
  return {};
#endif
}

struct CliArgs {

  std::vector<const char *> data;
  std::unordered_set<size_t> deleted{};
  explicit CliArgs(const std::vector<const char *> &data) : data(data) {}

  bool has(const std::string &flag) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() |
           exists([&](auto &chars) { return chars == flag; });
  }

  bool has(const std::string &flag, size_t pos) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) |
           exists([&](auto &chars, auto i) { return chars == flag && i == pos; });
  }

  std::optional<std::string> get(const std::string &flag) const {
    return data                                                        //
           | zip_with_index()                                          //
           | filter([&](auto v, auto i) { return !deleted.count(i); }) //
           | collect_first([&](auto &chars, auto i) -> std::optional<std::string> {
               if (const std::string arg = chars; arg == flag) {
                 if (i + 1 < data.size()) return data[i + 1];                                // -foo :: bar :: Nil
                 else return {};                                                             // -foo :: Nil
               } else if (arg ^ starts_with(flag + "=")) return arg.substr(flag.size() + 1); // -foo=bar
               return {};
             });
  }

  bool popBool(const std::string &flag) {
    const auto oldSize = deleted.size();
    data | zip_with_index<size_t>() | for_each([&](auto &x, auto i) {
      if (!deleted.count(i) && x == flag) deleted.insert(i);
    });
    return oldSize != deleted.size();
  }

  std::optional<std::string> popValue(const std::string &flag) {
    std::optional<std::string> last;
    for (size_t i = 0; i < data.size(); ++i) {
      if (deleted.count(i)) continue;
      if (const std::string arg = data[i]; arg == flag && i + 1 < data.size()) { // -foo :: bar :: Nil
        deleted.insert(i);
        deleted.insert(i + 1);
        last = data[i + 1];
      } else if (arg ^ starts_with(flag + "=")) { // -foo=bar
        deleted.insert(i);
        last = arg.substr(flag.size() + 1);
      }
    }
    return last;
  }

  std::vector<const char *> remaining() const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() | to_vector();
  }
};

// Locates the external driver in priority order: `--driver <path>`, ${envVar},
// then ${execParent}/<coLocatedName>(.exe). Returns "" if none found.
inline std::string resolveExternalDriver(CliArgs &args, const char *envVar, const std::string &coLocatedName,
                                         const std::string &execParentPath) {
  if (auto driverArg = args.popValue("--driver")) return *driverArg;
  if (auto driverEnv = std::getenv(envVar)) return driverEnv;
  if (auto coLocated = joinPath(execParentPath, executableName(coLocatedName)); llvm::sys::fs::exists(coLocated)) return coLocated;
  return {};
}

struct Driver {

  static std::vector<std::string> enableLLDAndLTO(const CliArgs &args, const std::string &ltoType = "thin") {
    std::vector<std::string> result;
    if (!(args.has("-flto") || args.has("-flto=thin") || args.has("-flto=full"))) //
      result.emplace_back(fmt::format("-flto={}", ltoType));
    if (!args.has("-fuse-ld=lld")) //
      result.emplace_back("-fuse-ld=lld");
    return result;
  }

  static std::vector<std::string> dynamicOriginLinkFlags(const std::string &libsPath, const std::string &libName) {
#ifdef _WIN32
    return {fmt::format("{}/{}.lib", libsPath, libName)};
#elif defined(__APPLE__)
    return {fmt::format("-L{}", libsPath),          //
            fmt::format("-l{}", libName),           //
            fmt::format("-Wl,-rpath,{}", libsPath), //
            "-Wl,-rpath,@loader_path"};
#else
    return {fmt::format("-L{}", libsPath),          //
            fmt::format("-l{}", libName),           //
            fmt::format("-Wl,-rpath,{}", libsPath), //
            "-Wl,-rpath,$ORIGIN"};
#endif
  }

  static std::vector<std::string> clangPassPluginFlags(const std::string &pluginPath, const std::vector<std::string> &args = {}) {
    return std::vector{
               fmt::format("-fplugin={}", pluginPath),
               fmt::format("-fpass-plugin={}", pluginPath),
           } //
           ^ concat(args ^ map([](auto &arg) { return fmt::format("-mllvm={}", arg); }));
  }

  static std::vector<std::string> lldPassPluginFlags(const std::string &pluginPath, const std::vector<std::string> &args = {}) {
    // XXX ld64.lld accepts -mllvm only as Separate; comma form works on both ELF and MachO.
    return std::vector{
               fmt::format("-Wl,--load-pass-plugin,{}", pluginPath),
               fmt::format("-Wl,-mllvm,-load={}", pluginPath),
           } //
           ^ concat(args ^ map([](auto &arg) { return fmt::format("-Wl,-mllvm,{}", arg); }));
  }
};

struct StdParOptions {

  enum class LinkKind : uint8_t { Static = 1, Dynamic = 2, Disabled = 3 };

  enum class MemKind : uint8_t { Direct = 1, Interpose = 2, Reflect = 3 };

  enum class VerboseLevel : uint8_t { None = 0, Info = 1, Debug = 2 };

  static std::variant<std::string, LinkKind> parseLinkKind(const std::string &arg) {
    if (auto v = arg ^ to_lower(); v == "static") return LinkKind::Static;
    else if (v == "dynamic") return LinkKind::Dynamic;
    else if (v == "disabled") return LinkKind::Disabled;
    return "Unknown link kind `" + arg + "`";
  }

  static std::variant<std::string, MemKind> parseMemKind(const std::string &arg) {
    if (auto v = arg ^ to_lower(); v == "direct") return MemKind::Direct;
    else if (v == "interpose") return MemKind::Interpose;
    else if (v == "reflect") return MemKind::Reflect;
    return "Unknown mem kind `" + arg + "`";
  }

  static std::variant<std::string, VerboseLevel> parseVerboseLevel(const std::string &arg) {
    if (auto v = arg ^ to_lower(); v == "none") return VerboseLevel::None;
    else if (v == "info") return VerboseLevel::Info;
    else if (v == "debug") return VerboseLevel::Debug;
    return "Unknown debug level `" + arg + "`";
  }

  VerboseLevel verbose = VerboseLevel::None;
  bool noCompress = false;
  std::string targets{};
  MemKind mem = MemKind::Reflect;
  LinkKind rt = LinkKind::Static;
  LinkKind jit = LinkKind::Disabled;
  std::optional<int> stackDepth = {};

  static std::variant<std::vector<std::string>, std::optional<StdParOptions>> parse(CliArgs &args) {
    const std::string fStdParFlag = "-fstdpar";
    const std::string fStdParVerboseFlag = "-fstdpar-verbose";
    const std::string fStdParArchNoCompressFlag = "-fstdpar-no-compress";
    const std::string fStdParArchFlag = "-fstdpar-arch";
    const std::string fStdParMemFlag = "-fstdpar-mem";
    const std::string fStdParRtFlag = "-fstdpar-rt";
    const std::string fStdParJitFlag = "-fstdpar-jit";
    const std::string fStdParStackFlag = "-fstdpar-stack";

    auto fStdPar = false, fStdParDependents = false;
    StdParOptions options;
    std::vector<std::string> errors;

    auto markError = [&](const std::string &prefix) { return [&](const std::string &x) { errors.push_back("\"" + prefix + "\": " + x); }; };

    if (args.popBool(fStdParFlag)) {
      fStdPar = true;
    }
    if (auto verbose = args.popValue(fStdParVerboseFlag)) {
      fStdParDependents = true;
      parseVerboseLevel(*verbose) ^ foreach_total(markError(fStdParVerboseFlag), [&](const VerboseLevel &x) { options.verbose = x; });
    }
    if (args.popBool(fStdParArchNoCompressFlag)) {
      fStdParDependents = true;
      options.noCompress = true;
    }
    if (auto arch = args.popValue(fStdParArchFlag)) {
      fStdParDependents = true;
      options.targets = *arch;
    }
    if (auto mem = args.popValue(fStdParMemFlag)) {
      fStdParDependents = true;
      parseMemKind(*mem) ^ foreach_total(markError(fStdParMemFlag), [&](const MemKind &x) { options.mem = x; });
    }
    if (auto rt = args.popValue(fStdParRtFlag)) {
      fStdParDependents = true;
      parseLinkKind(*rt) ^ foreach_total(markError(fStdParRtFlag), [&](const LinkKind &x) { options.rt = x; });
    }
    if (auto jit = args.popValue(fStdParJitFlag)) {
      fStdParDependents = true;
      parseLinkKind(*jit) ^ foreach_total(markError(fStdParJitFlag), [&](const LinkKind &x) { options.jit = x; });
    }
    if (auto stack = args.popValue(fStdParStackFlag)) {
      fStdParDependents = true;
      if (auto n = parsePositiveInt(*stack)) options.stackDepth = *n;
      else markError(fStdParStackFlag)("expected a positive integer, got: " + *stack);
    }

    if (!fStdPar && fStdParDependents)
      errors.insert(errors.begin(), fStdParFlag + " not specified but StdPar dependent flags used, pleased add " + fStdParFlag);

    if (errors.empty()) return fStdPar ? std::optional{options} : std::nullopt;
    else return errors;
  }
};

// Retain the JIT ABI and expose static symbols to runtime lookup.
inline std::vector<std::string> jitCompilerLinkFlags(StdParOptions::LinkKind jit, const std::string &libsPath,
                                                     const std::string &llvmLibPath, bool needsCxxRuntime) {
  std::vector<std::string> out;
  const auto anchor = [] {
#if defined(_WIN32)
    return std::vector<std::string>{"-Xlinker", "/INCLUDE:polyc_jit_compile"};
#elif defined(__APPLE__)
    return std::vector<std::string>{"-Wl,-u,_polyc_jit_compile"};
#else
    return std::vector<std::string>{"-Wl,-u,polyc_jit_compile"};
#endif
  };
  switch (jit) {
    case StdParOptions::LinkKind::Dynamic: {
      out ^= concat(anchor());
#if defined(_WIN32)
      out ^= concat(Driver::dynamicOriginLinkFlags(libsPath, "polyc-jit"));
#else
      out ^= concat(Driver::dynamicOriginLinkFlags(libsPath, "polyc"));
#endif
      break;
    }
    case StdParOptions::LinkKind::Static: {
      out ^= concat(anchor());
#if defined(_WIN32)
      out ^= concat(std::vector{"polyc_jit_compile", "polyc_jit_last_error", "polyc_jit_free"} |
                    flat_map([](auto s) { return std::vector<std::string>{"-Xlinker", fmt::format("/EXPORT:{}", s)}; }));
#elif defined(__APPLE__)
      out ^= concat(std::vector{"_polyc_jit_compile", "_polyc_jit_last_error", "_polyc_jit_free"} |
                    map([](auto s) { return fmt::format("-Wl,-exported_symbol,{}", s); }));
#else
      out ^= concat(std::vector{"polyc_jit_compile", "polyc_jit_last_error", "polyc_jit_free"} |
                    map([](auto s) { return fmt::format("-Wl,--export-dynamic-symbol={}", s); }));
#endif
      out ^= append(joinPath(libsPath, staticLibraryName("polyc-jit-static")));
#if defined(POLYREGION_JIT_LLVM_DYLIB)
      out ^= concat(Driver::dynamicOriginLinkFlags(llvmLibPath, "LLVMpolyregion"));
#else
      (void)llvmLibPath;
#endif
#if defined(__APPLE__)
      if (needsCxxRuntime) out ^= concat(std::vector<std::string>{"-lc++"});
#elif !defined(_WIN32)
      if (needsCxxRuntime) out ^= concat(std::vector<std::string>{"-lstdc++", "-lm"});
#endif
#if defined(__linux__)
      // A merged archive cannot carry CMake's non-archive link dependencies. Keep
      // these after it for older glibc/sysroots where pthread, dl and rt are separate.
      out ^= concat(std::vector<std::string>{"-pthread", "-ldl", "-lrt"});
#endif
#if defined(POLYREGION_ASAN_BUILD) && !defined(_WIN32)
      // The static compiler archive is built from the sanitizer-instrumented native objects;
      // its client link must therefore provide the same ASan/UBSan shared runtimes.
      if (needsCxxRuntime) {
        // Flang does not accept Clang's -fsanitize/-shared-libsan driver flags.
        // Link the shipped shared runtime explicitly; the ASan test runner also
        // preloads this same DSO so its interceptors are installed first.
        std::error_code ec;
        const auto root = joinPath(llvmLibPath, "clang");
  #if defined(__APPLE__)
        constexpr auto runtimeName = "libclang_rt.asan_osx_dynamic.dylib";
  #else
        constexpr auto runtimeName = "libclang_rt.asan.so";
  #endif
        for (llvm::sys::fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
          const auto runtime = it->path();
          if (llvm::sys::path::filename(runtime) != runtimeName) continue;
          const auto runtimeDir = llvm::sys::path::parent_path(runtime).str();
  #if defined(__APPLE__)
          out ^= concat(std::vector<std::string>{runtime, fmt::format("-Wl,-rpath,{}", runtimeDir)});
  #else
          out ^= concat(std::vector<std::string>{"-Wl,--push-state,--no-as-needed", runtime, "-Wl,--pop-state",
                                                 fmt::format("-Wl,-rpath,{}", runtimeDir)});
  #endif
          break;
        }
      } else {
        out ^=
            concat(std::vector<std::string>{"-fsanitize=address,undefined", "-fno-sanitize=vptr", "-shared-libsan", "-frtlib-add-rpath"});
      }
#endif
      break;
    }
    case StdParOptions::LinkKind::Disabled: break;
  }
  return out;
}

inline std::vector<std::string> mkDelimitedEnvPaths(const char *env, std::optional<std::string> leading, const char separator) {
  std::vector<std::string> xs;
  if (auto line = std::getenv(env); line) {
    for (auto &path : line ^ split(separator)) {
      if (leading) xs.push_back(*leading);
      xs.push_back(path);
    }
  }
  return xs;
}

inline std::string dynamicLibSuffix() {
#if defined(__linux__)
  return "so";
#elif defined(__APPLE__)
  return "dylib";
#elif defined(_WIN32)
  return "dll";
#else
  #error "Unsupported platform"
#endif
}

inline std::string staticLibSuffix() {
#if defined(__linux__)
  return "a";
#elif defined(__APPLE__)
  return "a";
#elif defined(_WIN32)
  return "lib";
#else
  #error "Unsupported platform"
#endif
}

} // namespace polyregion::polyfront
