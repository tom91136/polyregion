#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "fmt/args.h"
#include "fmt/format.h"

#include "polyinvoke/device_lock.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/io.hpp"
#include "polyregion/test_process.h"

#include "fire.hpp"
#include "polytest/ctest_emit.hpp"
#include "polytest/lit.hpp"
#include "polytest/pch.hpp"
#include "polytest/profile.hpp"
#include "polytest/skip_proof.hpp"

// XXX on Apple, clang++/flang-new default to the triple they were themselves built with when no
// --target= is given, which is wrong for a cross-built dist so we pin directly
#if defined(__APPLE__) && defined(__x86_64__)
  #define POLYTEST_APPLE_TARGET_FLAG "--target=x86_64-apple-darwin "
#elif defined(__APPLE__) && defined(__aarch64__)
  #define POLYTEST_APPLE_TARGET_FLAG "--target=arm64-apple-darwin "
#else
  #define POLYTEST_APPLE_TARGET_FLAG ""
#endif

namespace polyregion::polytest {

using namespace aspartame;

struct DriverConfig {
  std::string driverPath;
  std::string binaryDir;
  std::string workDir;
  std::vector<std::string> testFiles;
  std::string profileDir;
  std::string archVar;
  // Each {label, value} variant produces one task; value binds to defaultsVar in templates,
  // label binds to defaultsLabelVar so the task display can show e.g. "opt=O0".
  std::string defaultsVar;
  std::string defaultsLabelVar;
  std::vector<std::pair<std::string, std::string>> defaultsVariants;
  std::vector<std::pair<std::string, std::string>> extraVars;
  std::pair<std::string, std::string> stdpar;
  std::string driverEnvVar;
  std::vector<std::string> passthroughEnvs;
  std::string outputPrefix;
  std::string directive;
  bool cleanupOnSuccess;
  // compile steps of sources including this header compile against a PCH of it, built lazily in the work dir
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::string preamble = {};
  // binds targetVar to the value keyed by the compile target's canonical backend; unmapped targets are not tested
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::string targetVar = {};
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::vector<std::pair<std::string, std::string>> targetValues = {};
  // per-entry ctest TIMEOUT for cases without a `timeout:` directive; none leaves it to `ctest --timeout`
  std::optional<unsigned> defaultTimeoutSecs = DefaultTimeoutSecs;
};

inline constexpr const char *OutputName = "out";

enum class Mode : std::uint8_t { Offload, Passthrough };
inline constexpr const char *modeName(Mode m) { return m == Mode::Offload ? "offload" : "passthrough"; }

// one execution of a compiled task: `id` names the (case, device) pair, `target` is the full `<compile>!<device>`
// string every run-side read of the target uses, `runs` are the steps after the compile expanded for that target
struct RunVariant {
  std::string id;
  std::string target;
  std::string suffix;
  std::vector<std::pair<std::string, std::string>> env;
  std::vector<TestCase::Run> runs;
  std::string display;
};

// ctest-name-safe: alnum + `_` + `-` kept, else hex-escaped so distinct inputs don't collide
inline std::string ctestSafe(std::string_view s) {
  return std::string(s) ^ mk_string("", [](char c) {
           return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-'
                      ? std::string(1, c)
                      : fmt::format("_{:02x}", static_cast<unsigned char>(c));
         });
}

struct Task {
  Mode mode;
  std::string testFile;
  std::string caseName;
  std::vector<std::pair<std::string, std::string>> variables;
  std::string dir;
  std::string output;
  std::optional<std::string> compileFailure;
  std::vector<TestCase::Run> runs; // the compile step, expanded against the compile target only
  std::vector<RunVariant> variants;
  std::optional<unsigned> timeoutSecs;
  std::optional<pch::Spec> pchSpec = std::nullopt;

  [[nodiscard]] std::string display() const {
    return fmt::format("[{}] {}/{} {}", modeName(mode), extractTestName(testFile), caseName,
                       variables | mk_string(" ", [](const auto &k, const auto &v) { return k + "=" + v; }));
  }

  [[nodiscard]] std::string id() const {
    auto vars = variables | mk_string("", [&](const auto &k, const auto &v) { return "-" + ctestSafe(k) + "_" + ctestSafe(v); });
    return fmt::format("{}-{}-{}{}", modeName(mode), ctestSafe(extractTestName(testFile)), ctestSafe(caseName), vars);
  }
};

enum class TaskStatus : std::uint8_t { Pass = 0, Fail = 1, Skip = 2 };

struct TaskOutcome {
  TaskStatus status = TaskStatus::Pass;
  std::string failReason;
  std::string stdoutCapture;
  std::string stderrCapture;
  std::string cmdline;
  double secs = 0.0;
};

namespace detail {

inline std::string archFor(const Task &t, const DriverConfig &cfg) {
  return t.variables                                                                                                       //
         ^ collect_first([&](const auto &k, const auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; }) //
         ^ get_or_else(std::string{});
}

inline std::string compileTargetFor(const Task &t, const DriverConfig &cfg) { return compileTargetOf(archFor(t, cfg)); }

inline std::vector<std::string> baseEnvs(const Task &t, const DriverConfig &cfg, bool preloadSanitizer, const std::string &target) {
  std::vector<std::pair<std::string, std::string>> kvs;
  auto put = [&](std::string_view k, std::string_view v) {
    for (auto &e : kvs)
      if (e.first == k) {
        e.second = std::string(v);
        return;
      }
    kvs.emplace_back(std::string(k), std::string(v));
  };
  const auto putKv = [&](const std::string &kv) {
    if (const auto pair = kv ^ split_once('=')) put(pair->first, pair->second);
  };
  if (t.mode == Mode::Passthrough)
    for (const auto &kv : cfg.passthroughEnvs)
      putKv(kv);
  put(cfg.driverEnvVar, cfg.driverPath);
  put(polyregion::env::PolyrtPlatform, target);
  put(polyregion::env::PolyrtHostFallback, "0");
  {
    llvm::SmallString<256> manifest(t.output + ".images");
    llvm::sys::fs::make_absolute(manifest);
    put(polyregion::env::PolyfrontImageManifest, std::string(manifest));
  }
  if (const char *const v = std::getenv(polyregion::env::PolyrtStrictSelect); v && v[0]) put(polyregion::env::PolyrtStrictSelect, v);
  else put(polyregion::env::PolyrtStrictSelect, "1");
  // XXX NVIDIA's shader disk cache (~/.cache/nvidia/GLCache) bloats to 100k+ files across a sweep and the
  // driver stat-scans it on every init (~20s spin); one-shot test kernels gain nothing, so disable it
  put("__GL_SHADER_DISK_CACHE", "0");
  // XXX rusticl loads Mesa's LLVM in-process; the ICD loader pulls it in for ANY OpenCL target, clashing
  // with polydco/polystl's LLVM (two libLLVMSPIRVLib) -> bad_alloc. expose it only when it's selected
  const bool rusticlTarget = target ^ contains_slice("rusticl");
  materialiseProfileEnvironment(loadProfileEnv(cfg.profileDir))                                         //
      | filter([&](const auto &kv) { return rusticlTarget || !(kv ^ starts_with("RUSTICL_ENABLE=")); }) //
      | for_each(putKv);
  const char *const lockEnv = std::getenv(polyregion::env::PolyinvokeTestLock);
  const bool configuredLock = kvs | exists([&](const auto &entry) { return entry.first == polyregion::env::PolyinvokeTestLock; });
  if (target ^ starts_with("metal")) put(polyregion::env::PolyinvokeTestLock, "1");
  else if (lockEnv) put(polyregion::env::PolyinvokeTestLock, lockEnv);
  else if (!configuredLock) put(polyregion::env::PolyinvokeTestLock, "1");
  if (const char *const v = std::getenv("ASAN_OPTIONS")) put("ASAN_OPTIONS", v);
  else
    put("ASAN_OPTIONS", "alloc_dealloc_mismatch=0,detect_leaks=0,protect_shadow_gap=0,verify_asan_link_order=0,strip_env=0"
#ifdef __APPLE__
                        ",detect_container_overflow=0"
#endif
    );
  if (const char *const v = std::getenv("UBSAN_OPTIONS")) put("UBSAN_OPTIONS", v);
  else put("UBSAN_OPTIONS", "print_stacktrace=1");
  if (std::getenv(polyregion::env::PolytestDebug)) put(polyregion::env::PolyrtDebug, "2");
  // child env replaced wholesale; forward overrides + QEMU shim inputs or they vanish
  for (const auto *name : {polyregion::env::PolycppLinkThreads,
                           polyregion::env::PolyfcLinkThreads, //
                           polyregion::env::PolyinvokeDisableSvm,
                           polyregion::env::PolyinvokeDisableBackends, //
                           polyregion::env::PolyrtMirror,
                           polyregion::env::PolyrtStats,       //
                           polyregion::env::PolyregionPassLog, //
                           polyregion::env::PolyregionDebug,
                           polyregion::env::PolyinvokeOpenclLib, //
                           polyregion::env::PolyinvokeCCompiler, //
                           polyregion::env::PolyregionSysroot,
                           polyregion::env::PolyregionBitcodeDir, //
                           polyregion::env::PolypassPlugins,
                           polyregion::env::PolyregionCacheDir, //
                           "XDG_CACHE_HOME",
                           "POCL_CACHE_DIR",
                           "QEMU_LD_PREFIX",
                           "CUEW_LIB_PATH",
                           "HIPEW_LIB_PATH",
                           "HSAEW_LIB_PATH"})
    if (const char *v = std::getenv(name)) put(name, v);
  for (const auto *name : {polyregion::env::PolyfrontLibraryPath, polyregion::env::PolyfrontLibraryCapabilities,
                           polyregion::env::PolyinvokeVulkanLib, "VK_LOADER_DEBUG"})
    if (const char *v = std::getenv(name)) put(name, v);
  if (target ^ starts_with("opencl")) put(polyregion::env::PolyinvokeDisableSvm, "1");
  // XXX forward OCL_ICD_VENDORS (dropped by the wholesale child env) narrowed to one vendor: rusticl pulls
  // Mesa's LLVM in-process clashing with polystl's, so a pocl target must never see rusticl.icd (or vice versa)
  if (const char *vendors = std::getenv("OCL_ICD_VENDORS")) {
    const char *vendorIcd = [&]() -> const char * {
      if (rusticlTarget) return "rusticl.icd";
      if (target ^ contains_slice("Portable")) return "pocl.icd";
      return nullptr;
    }();
    if (vendorIcd) {
      const std::string narrowed = std::string(vendors) + "/" + vendorIcd;
      put("OCL_ICD_VENDORS", llvm::sys::fs::exists(narrowed) ? narrowed : vendors);
      if (rusticlTarget) { // pin Mesa to the software rasteriser like env.sh
        const char *re = std::getenv("RUSTICL_ENABLE"), *gd = std::getenv("GALLIUM_DRIVER");
        put("RUSTICL_ENABLE", re ? re : "llvmpipe");
        put("GALLIUM_DRIVER", gd ? gd : "llvmpipe");
      }
    }
  }
  // XXX likewise forward VK_DRIVER_FILES narrowed to the selected device's manifest, else the child falls back
  // to a system Vulkan driver, or none at all on clean CI runners
#ifdef _WIN32
  constexpr char VkIcdSeparator = ';';
#else
  constexpr char VkIcdSeparator = ':';
#endif
  if (const char *icds = std::getenv("VK_DRIVER_FILES") ? std::getenv("VK_DRIVER_FILES") : std::getenv("VK_ICD_FILENAMES")) {
    const char *want = [&]() -> const char * {
      if (target ^ contains_slice("llvmpipe")) return "lvp_icd";
      if (target ^ contains_slice("SwiftShader")) return "swiftshader";
      return nullptr;
    }();
    if (want)
      if (const auto entry = std::string(icds) ^ split(VkIcdSeparator) ^ find([&](const auto &e) { return e ^ contains_slice(want); })) {
        put("VK_DRIVER_FILES", *entry);
        put("VK_ICD_FILENAMES", *entry);
      }
  }
  {
    // NOLINTNEXTLINE(misc-const-correctness)
    std::string pathVal = std::getenv("PATH") ? std::getenv("PATH") : "";
#ifdef _WIN32
    // no rpath on Windows: prepend <binaryDir>/lib/<stem>/lib so test exes find staged DLLs
    std::string bd = cfg.binaryDir;
    while (!bd.empty() && (bd.back() == '/' || bd.back() == '\\'))
      bd.pop_back();
    const auto slash = bd.find_last_of("/\\");
    const std::string stem = (slash == std::string::npos) ? bd : bd.substr(slash + 1);
    if (!bd.empty() && !stem.empty()) pathVal = bd + "\\lib\\" + stem + "\\lib;" + pathVal;
#endif
    if (!pathVal.empty()) put("PATH", pathVal);
  }
#ifdef __APPLE__
  // XXX forward DYLD_* (compiled test binaries can't find dist libs otherwise -> SIGTRAP) and SDKROOT/LIBRARY_PATH
  // (pocl links source kernels with clang -> ld needs the SDK for -lSystem)
  for (const auto *name : {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "TMPDIR", "HOME", "SDKROOT", "LIBRARY_PATH"}) {
    if (auto v = std::getenv(name)) put(name, v);
  }
#elif defined(__linux__)
  // XXX forward LD_LIBRARY_PATH (dist-relative .so when rpath misses; lavapipe must precede) and a parent
  // LD_PRELOAD shim into the wholesale-replaced child env; the run-only ASan block layers its rt ahead
  for (const auto *name : {"TMPDIR", "HOME", "LD_PRELOAD", polyregion::env::PolyrtDumpKernel})
    if (const char *v = std::getenv(name)) put(name, v);
  if (const auto *ld = std::getenv("LD_LIBRARY_PATH")) put("LD_LIBRARY_PATH", ld);
#elif defined(_WIN32)
  // XXX CreateProcess needs SYSTEMROOT; clang needs TEMP/TMP; link.exe needs LIB/INCLUDE for
  // CRT and SDK lookup (flang does not auto-populate -libpath). Inherit a vcvars-style env.
  for (const auto *name : {"SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE", "WINDIR", "PATHEXT", "COMSPEC", "ProgramFiles",
                           "ProgramFiles(x86)", "ProgramData", "LOCALAPPDATA", "APPDATA", "LIB", "INCLUDE", "LIBPATH"}) {
    if (auto v = std::getenv(name)) put(name, v);
  }
#endif
  // XXX the generated executable loads the instrumented runtime late, so force asan in first
#if defined(__APPLE__) || defined(__linux__)
  if (preloadSanitizer)
    if (const char *rt = std::getenv(polyregion::env::PolytestAsanPreload); rt && *rt) {
  #ifdef __APPLE__
      put("DYLD_INSERT_LIBRARIES", rt);
  #else
      const char *const prev = std::getenv("LD_PRELOAD");
      put("LD_PRELOAD", prev ? std::string(rt) + ":" + prev : std::string(rt));
  #endif
    }
#endif
  return kvs ^ map([](const auto &k, const auto &v) { return k + "=" + v; });
}

inline std::string resolveBin(const std::string &name, const DriverConfig &cfg) {
  // XXX Prefer cwd for generated test binaries: a stale copy in binaryDir from a previous build would otherwise shadow the fresh one.
  const auto exists = [](llvm::StringRef p) -> std::optional<std::string> {
    if (llvm::sys::fs::exists(p)) return std::string(p);
#ifdef _WIN32
    auto withExe = std::string(p) + ".exe";
    if (llvm::sys::fs::exists(withExe)) return withExe;
#endif
    return std::nullopt;
  };
  const auto toAbs = [](llvm::StringRef p) {
    llvm::SmallString<256> abs(p);
    llvm::sys::fs::make_absolute(abs);
    return std::string(abs);
  };
  if (name.starts_with(cfg.outputPrefix))
    if (auto p = exists(name)) return toAbs(*p);
  if (auto p = exists(fmt::format("{}/{}", cfg.binaryDir, name))) return *p;
  if (auto p = exists(name)) return toAbs(*p);
  // XXX binaryDir is baked at build time but absent in the dist-only CI job; fall back to PATH (the dist bin/)
  if (auto p = llvm::sys::findProgramByName(name)) return *p;
  return name;
}

struct StepResult {
  int exitCode = 0;
  std::string stdoutText;
  std::string stderrText;
  std::string cmdline;
};

inline void removeArtifacts(const std::string &dir) {
  llvm::SmallString<256> absolute(dir);
  llvm::sys::fs::make_absolute(absolute);
  llvm::sys::fs::remove_directories(absolute);
}

inline StepResult runStep(const Task &task, const DriverConfig &cfg, const std::string &command, const std::string &target) {
  const auto fragments = command ^ words();
  auto [envBits, args] = fragments ^ span([](const auto &x) { return x ^ contains('='); });

  auto envs = baseEnvs(task, cfg, args ^ head_maybe() ^ exists([&](const auto &arg) { return arg == task.output; }), target) //
              ^ filter([&](const auto &e) {
                  return !(envBits ^ exists([&](const auto &kv) {
                             return kv ^ split_once('=') ^ exists([&](const auto &entry) { return e ^ starts_with(entry.first + "="); });
                           }));
                });
  envs ^= concat(envBits);

  const std::vector<llvm::StringRef> envRefs = envs ^ map([](const auto &x) -> llvm::StringRef { return x; });
  const std::vector<llvm::StringRef> argRefs = args ^ map([](const auto &x) -> llvm::StringRef { return x; });

  const auto outPath = task.output + ".stdout";
  const auto errPath = task.output + ".stderr";
  const llvm::FileRemover outRemover(outPath), errRemover(errPath);

  if (args.empty()) return {.exitCode = -1, .stdoutText = {}, .stderrText = "empty command", .cmdline = command};
  const auto resolved = resolveBin(args[0], cfg);
  std::string execErr;
  const auto envUnsigned = [](const char *key) -> unsigned {
    if (const char *v = std::getenv(key)) {
      char *end = nullptr;
      const unsigned long n = std::strtoul(v, &end, 10);
      if (end != v && *end == '\0') return static_cast<unsigned>(n);
    }
    return 0;
  };
  const unsigned secondsToWait = envUnsigned(polyregion::env::PolytestTimeout);
  const unsigned memLimitMb = envUnsigned(polyregion::env::PolytestMemLimitMb);
  const auto code =
      llvm::sys::ExecuteAndWait(resolved, argRefs, envRefs, {std::nullopt, outPath, errPath}, secondsToWait, memLimitMb, &execErr);

  auto stdoutText = polyregion::read_string(outPath);
  auto stderrText = polyregion::read_string(errPath);

  // XXX -2 is LLVM's signal-killed sentinel; surface WTERMSIG via the ErrMsg out-param.
  if (code == -2 && !execErr.empty()) stderrText += "\n[polytest] signal: " + execErr + "\n";

  std::string cmdline = argRefs | drop(1) | prepend(llvm::StringRef(resolved)) | mk_string(" ", [](const auto &s) { return s.str(); });
  return {.exitCode = code, .stdoutText = std::move(stdoutText), .stderrText = std::move(stderrText), .cmdline = std::move(cmdline)};
}

inline skip::Verdict skipVerdict(const Task &task, const DriverConfig &cfg, const std::string &selector, const std::string &stderrText) {
  std::optional<skip::Inventory> inventory;
  if (compiletime::TargetSpec::parse(selector)) {
    const std::vector<std::string> args{resolveBin("polyinfo", cfg), "--json", "--select", selector};
    const auto envs = baseEnvs(task, cfg, false, selector);
    const std::vector<llvm::StringRef> envRefs = envs ^ map([](const auto &x) -> llvm::StringRef { return x; });
    const std::vector<llvm::StringRef> argRefs = args ^ map([](const auto &x) -> llvm::StringRef { return x; });
    const auto outPath = task.output + ".inventory";
    const llvm::FileRemover outRemover(outPath);
    std::string execErr;
    if (llvm::sys::ExecuteAndWait(args[0], argRefs, envRefs, {std::nullopt, outPath, llvm::StringRef()}, 120, 0, &execErr) == 0)
      inventory = skip::parseInventory(polyregion::read_string(outPath));
  }
  std::vector<std::string> runFeatures;
  if (const char *const mirror = std::getenv(polyregion::env::PolyrtMirror); mirror && std::string_view(mirror) == "off")
    runFeatures.emplace_back(runtime::SystemPagingFeature);
  const auto manifest = skip::parseManifest(polyregion::read_string(task.output + ".images"));
  return skip::judge(selector, inventory, manifest, stderrText, runFeatures);
}

inline void absorbStep(TaskOutcome &o, StepResult &&r) {
  o.cmdline = std::move(r.cmdline);
  o.stdoutCapture = std::move(r.stdoutText);
  o.stderrCapture = std::move(r.stderrText);
}

inline bool checkExpect(const TestCase::Run::Expect &e, const std::vector<std::string> &lines, const std::string &stdoutCapture,
                        TaskOutcome &o) {
  const auto &[lineNum, expected] = e;
  if (!lineNum) {
    if (stdoutCapture == expected) return true;
    o.status = TaskStatus::Fail;
    o.failReason = fmt::format("expect mismatch: got '{}', want '{}'", stdoutCapture, expected);
    return false;
  }
  const auto idx = *lineNum < 0 ? lines.size() + static_cast<std::size_t>(*lineNum) : static_cast<std::size_t>(*lineNum);
  const auto &got = idx < lines.size() ? lines[idx] : std::string("<out-of-bounds>");
  if (got == expected) return true;
  o.status = TaskStatus::Fail;
  o.failReason = fmt::format("expect mismatch at line {}: got '{}', want '{}'", idx, got, expected);
  return false;
}

inline bool checkStderrCounts(const TestCase::Run &run, TaskOutcome &o) {
  for (const auto &[expected, text] : run.stderrCounts) {
    size_t actual = 0;
    for (auto at = o.stderrCapture.find(text); !text.empty() && at != std::string::npos; at = o.stderrCapture.find(text, at + text.size()))
      ++actual;
    if (actual == expected) continue;
    o.status = TaskStatus::Fail;
    o.failReason = fmt::format("stderr contains '{}' {} time(s), want {}", text, actual, expected);
    return false;
  }
  return true;
}

struct PchUse {
  std::optional<std::string> path;
  std::string state;
  bool failed = false;
};

inline std::string firstDiagnostic(const std::string &text) {
  return text ^ split('\n') ^ find([](const auto &line) { return line ^ contains_slice("error:"); })
         ^ get_or_else(std::string("no diagnostic"));
}

inline std::vector<std::string> pchKey(const Task &task, const DriverConfig &cfg, const pch::Spec &spec) {
  const auto exe = resolveBin(
      spec.command ^ drop_while([](const auto &t) { return pch::isEnvBit(t); }) ^ head_maybe() ^ get_or_else(std::string{}), cfg);
  llvm::SmallString<256> libs(llvm::sys::path::parent_path(exe));
  llvm::sys::path::append(libs, "..", "lib", llvm::sys::path::stem(exe), "lib");
  std::vector<std::string> binaries{exe, cfg.driverPath};
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(libs, ec), end; !ec && it != end; it.increment(ec))
    binaries.push_back(it->path());
  const auto identity = binaries ^ sort() ^ map([](const auto &path) {
                          const auto dep = pch::statDep(path);
                          return dep ? fmt::format("binary {} {} {}", dep->path, dep->size, dep->mtime) : "binary " + path;
                        });
  return identity ^ append(fmt::format("mode {}", modeName(task.mode))) ^ append("header " + spec.header) ^ concat(spec.command);
}

// XXX not -MD: the driver then drops link-mode flags such as -pthread, which changes the LangOptions
inline std::optional<std::string> buildPch(const Task &task, const DriverConfig &cfg, const pch::Spec &spec, const std::string &dir,
                                           const std::string &key, const std::vector<std::string> &keyLines) {
  const auto tmp = fmt::format("{}/{}.{}", dir, key, llvm::sys::Process::getProcessId());
  const llvm::FileRemover depRemover(tmp + ".d"), pchRemover(tmp + ".pch"), imagesRemover(tmp + ".images");
  Task buildTask = task;
  buildTask.output = tmp;
  const auto header = spec.header.empty() ? fmt::format("{}/empty.h", dir) : spec.header;
  if (spec.header.empty()) std::ofstream(header, std::ios::app);
  const auto command =
      spec.command
      ^ concat(std::vector<std::string>{"-Wno-unused-command-line-argument", "-Xclang", "-dependency-file", "-Xclang", tmp + ".d",
                                        "-Xclang", "-MT", "-Xclang", tmp + ".pch", "-Xclang", "-sys-header-deps", "--precompile", "-x",
                                        "c++-header", header, "-x", "none", "-o", tmp + ".pch"})
      ^ mk_string(" ");
  const auto built = runStep(buildTask, cfg, command, compileTargetFor(task, cfg));
  const auto failed = [&](const std::string &reason) {
    std::ofstream(fmt::format("{}/{}.failed", dir, key)) << command << "\n" << built.stderrText;
    return std::optional{reason};
  };
  if (built.exitCode != 0 || !llvm::sys::fs::exists(tmp + ".pch"))
    return failed(fmt::format("build exited {}: {}", built.exitCode, firstDiagnostic(built.stderrText)));
  const auto paths = pch::parseDepfile(polyregion::read_string(tmp + ".d")) ^ map([](const auto &p) {
                       llvm::SmallString<256> absolute(p);
                       llvm::sys::fs::make_absolute(absolute);
                       return std::string(absolute);
                     });
  const auto deps = paths ^ collect([](const auto &p) { return pch::statDep(p); });
  if (deps.empty() || deps.size() != paths.size()) return failed("dependency list is empty or names a missing file");
  const auto name =
      fmt::format("{}-{}.pch", key,
                  pch::digest(deps ^ mk_string("\n", [](const auto &d) { return fmt::format("{} {} {}", d.path, d.size, d.mtime); })));
  const auto target = fmt::format("{}/{}", dir, name);
  if (llvm::sys::fs::rename(tmp + ".pch", target) && !llvm::sys::fs::exists(target)) return failed("cannot publish " + name);
  pch::writeManifest(fmt::format("{}/{}.manifest", dir, key), {keyLines, name, deps, {}}, pch::identifiersOf(paths));
  llvm::sys::fs::remove(fmt::format("{}/{}.failed", dir, key));
  return std::nullopt;
}

inline PchUse acquirePch(const Task &task, const DriverConfig &cfg) {
  const char *const toggle = std::getenv(polyregion::env::PolytestPch);
  if (!task.pchSpec || (toggle && toggle[0] == '0')) return {};
  const auto &spec = *task.pchSpec;
  llvm::SmallString<256> cwd;
  llvm::sys::fs::current_path(cwd);
  llvm::sys::path::append(cwd, cfg.outputPrefix + "pch");
  const std::string dir(cwd);
  if (dir ^ exists([](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }))
    return {{}, "off: work dir path has whitespace"};
  llvm::sys::fs::create_directories(dir);
  const auto keyLines = pchKey(task, cfg, spec);
  const auto key = pch::digest(keyLines ^ mk_string("\n"));
  const auto base = fmt::format("{}/{}", dir, key);
  const auto lookup = [&]() -> std::optional<pch::Manifest> {
    auto manifest = pch::readManifest(base + ".manifest", spec.macros);
    if (manifest && pch::fresh(*manifest, keyLines) && llvm::sys::fs::exists(fmt::format("{}/{}", dir, manifest->pch))) return manifest;
    return std::nullopt;
  };
  auto manifest = lookup();
  std::string state = "hit";
  if (!manifest) {
    static std::mutex builders;
    const std::scoped_lock inProcess(builders);
    const pch::FileLock lock(base + ".lock");
    if (!(manifest = lookup())) {
      if (const auto failure = buildPch(task, cfg, spec, dir, key, keyLines)) return {{}, "build failed: " + *failure, true};
      manifest = lookup();
      state = "built";
    }
  }
  if (!manifest) return {{}, "build failed: manifest unreadable after build", true};
  if (!manifest->macroReads.empty()) return {{}, "off: macro " + manifest->macroReads.front() + " is read by the preamble"};
  return {fmt::format("{}/{}", dir, manifest->pch), state};
}

inline TaskOutcome compileTask(const Task &task, const DriverConfig &cfg) {
  if (task.runs.empty()) return {};
  removeArtifacts(task.dir);
  llvm::sys::fs::create_directories(task.dir);
  const auto start = std::chrono::steady_clock::now();
  const auto target = compileTargetFor(task, cfg);
  const auto toks = task.runs[0].command ^ split(' ');
  const auto out = toks                                                            //
                   | sliding(2, 1)                                                 //
                   | fold_left(std::string{}, [](const auto &acc, const auto &w) { //
                       return w.size() == 2 && w[0] == "-o" ? w[1] : acc;
                     });
  const char *reproEnv = std::getenv(polyregion::env::PolytestReproCheck);
  const bool repro = !task.compileFailure && !out.empty() && !(reproEnv && reproEnv[0] == '0')
                     && toks ^ exists([](const auto &t) { return t == "-fstdpar"; });
  const auto &plain = task.runs[0].command;
  const auto pchUse = acquirePch(task, cfg);
  if (!pchUse.state.empty()) std::fprintf(stderr, "[polytest] pch %s: %s\n", pchUse.state.c_str(), task.display().c_str());
  if (pchUse.failed) {
    TaskOutcome o;
    o.status = TaskStatus::Fail;
    o.failReason = "pch " + pchUse.state;
    return o;
  }
  const auto cmd = pchUse.path ? plain + " -include-pch " + *pchUse.path : plain;
  auto r = runStep(task, cfg, cmd, target);
  const auto exit = r.exitCode;
  if (std::getenv(polyregion::env::PolyregionPassLog) && !r.stderrText.empty())
    std::fprintf(stderr, "[%s]\n%s\n", task.display().c_str(), r.stderrText.c_str());
  TaskOutcome o;
  std::string reproReason;
  std::optional<StepResult> reproFailure;
  if (exit == 0 && repro) {
#if defined(__APPLE__) || defined(_WIN32)
    // XXX Mach-O debug maps and PDBs record the driver's randomly named temporary objects, so compare two objects instead
    const std::string sample = out + ".repro.o";
  #ifdef _WIN32
    // Flang rejects this Clang-only flag and already emits reproducible objects
    const bool isPolyfc = toks ^ exists([](const auto &t) { return t.ends_with("polyfc") || t.ends_with("polyfc.exe"); });
    const auto rebuild = (toks ^ mk_string(" ", [&](const auto &t) { return t == out ? sample : t; }))
                         + (isPolyfc ? " -c" : " -mno-incremental-linker-compatible -c");
  #else
    const auto rebuild = (toks ^ mk_string(" ", [&](const auto &t) { return t == out ? sample : t; })) + " -c";
  #endif
    const auto firstBuild = runStep(task, cfg, rebuild, target);
#else
    const auto &sample = out;
    const auto &rebuild = cmd;
    const StepResult firstBuild;
#endif
    const auto first = firstBuild.exitCode == 0 ? polyregion::read_string(sample) : std::string{};
    if (auto again = firstBuild.exitCode != 0 ? firstBuild : runStep(task, cfg, rebuild, target); again.exitCode != 0)
      reproReason = fmt::format("repro compile failed (exit {})", again.exitCode), reproFailure = std::move(again);
    else if (first != polyregion::read_string(sample))
      reproReason = "non-reproducible build: two compiles of the same source produced different outputs";
    if (sample != out) llvm::sys::fs::remove(sample);
  }
  if (reproFailure) absorbStep(o, std::move(*reproFailure));
  else absorbStep(o, std::move(r));
  o.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (task.compileFailure) {
    if (exit == 0) o.failReason = "compile unexpectedly succeeded", o.status = TaskStatus::Fail;
    else if (o.stderrCapture.find(*task.compileFailure) == std::string::npos)
      o.failReason = fmt::format("compile failed without expected diagnostic '{}'", *task.compileFailure), o.status = TaskStatus::Fail;
  } else if (exit == 77) o.failReason = "compile step exited 77: no compile-time skip can be justified", o.status = TaskStatus::Fail;
  else if (exit != 0) o.failReason = fmt::format("compile failed (exit {})", exit), o.status = TaskStatus::Fail;
  else if (!reproReason.empty()) o.failReason = reproReason, o.status = TaskStatus::Fail;
  else checkStderrCounts(task.runs[0], o);
  return o;
}

inline TaskOutcome runTask(const Task &task, const DriverConfig &cfg, const RunVariant &variant) {
  if (task.compileFailure) return {};
  TaskOutcome o;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < variant.runs.size(); ++i) {
    const auto step = i + 1;
    auto r = runStep(task, cfg, variant.runs[i].command, variant.target);
    const auto exit = r.exitCode;
    absorbStep(o, std::move(r));
    if (const auto &failure = variant.runs[i].failure) {
      if (exit == 0) o.status = TaskStatus::Fail, o.failReason = fmt::format("run step {} unexpectedly succeeded", step);
      else if (o.stderrCapture.find(*failure) == std::string::npos)
        o.status = TaskStatus::Fail, o.failReason = fmt::format("run step {} failed without expected diagnostic '{}'", step, *failure);
      break;
    }
    if (exit == 77) {
      const auto verdict = skipVerdict(task, cfg, variant.target, o.stderrCapture);
      o.status = verdict.justified ? TaskStatus::Skip : TaskStatus::Fail;
      o.failReason = verdict.justified ? verdict.reason : "unjustified skip (exit 77): " + verdict.reason;
      break;
    }
    if (exit != 0) {
      o.status = TaskStatus::Fail;
      o.failReason = fmt::format("run step {} failed (exit {})", step, exit);
      break;
    }
    const auto lines = o.stdoutCapture ^ split('\n');
    if (!(variant.runs[i].expect | forall([&](const auto &e) { return checkExpect(e, lines, o.stdoutCapture, o); }))) break;
    if (!checkStderrCounts(variant.runs[i], o)) break;
  }
  o.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return o;
}

} // namespace detail

// the name comes from the compile target alone, so it does not depend on how many devices are selected
inline std::vector<Task> mergeByCompile(const std::vector<Task> &tasks, const DriverConfig &cfg) {
  std::vector<Task> merged;
  std::unordered_map<std::string, std::size_t> index;
  for (const auto &t : tasks) {
    const auto key = t.dir + '\0' + (t.runs.empty() ? std::string{} : t.runs[0].command);
    if (const auto it = index.find(key); it != index.end()) merged[it->second].variants ^= concat(t.variants);
    else index.emplace(key, merged.size()), merged.push_back(t);
  }
  for (auto &t : merged)
    t.variables = t.variables ^ map([&](const auto &k, const auto &v) { return std::pair{k, k == cfg.archVar ? compileTargetOf(v) : v}; });
  return merged;
}

inline std::vector<Task> enumerateTasks(const DriverConfig &cfg, bool offload, bool passthrough,
                                        const std::vector<std::string> &caseFilters, const std::string &source = {}) {
  auto mkArgStore = [](const std::vector<std::pair<std::string, std::string>> &xs) {
    fmt::dynamic_format_arg_store<fmt::format_context> s;
    xs | for_each([&](const auto &k, const auto &v) { s.push_back(fmt::arg(k.c_str(), v)); });
    return s;
  };
  const auto matches = [&](const std::string &shortName, const std::string &caseName) {
    return caseFilters.empty() || (caseFilters | exists([&](const auto &f) { return f == shortName || f == caseName; }));
  };
  const std::string libmFlag =
#ifdef _WIN32
      "";
#else
      "-lm";
#endif
  std::vector<Mode> modes;
  if (offload) modes.emplace_back(Mode::Offload);
  if (passthrough) modes.emplace_back(Mode::Passthrough);
  const auto targets = loadTestTargets(cfg.profileDir);
  const auto profileVars = parseProfileEnvironment(loadProfileEnv(cfg.profileDir)).testVars;
  const auto profileOverrides = [&](const std::string &key) {
    return profileVars ^ exists([&](const auto &pv) { return pv.first == key; });
  };
  const auto extraVars = cfg.extraVars ^ filter([&](const auto &kv) { return !profileOverrides(kv.first); }) ^ concat(profileVars);
  const auto targetValueFor = [&](const std::string &arch) -> std::optional<std::string> {
    const auto spec = compiletime::TargetSpec::findByName(arch.substr(0, arch.find('@')));
    if (!spec) return {};
    return cfg.targetValues ^ collect_first([&](const auto &backend, const auto &value) -> std::optional<std::string> {
             const auto mapped = compiletime::TargetSpec::findByName(backend);
             return mapped && mapped->canonical == spec->canonical ? std::optional{value} : std::nullopt;
           });
  };

  // one single-device Task per mode for a (case, matrix-row, defaults-variant) combination; output paths and the compile
  // step derive from the compile target only, so tasks that differ in device alone can be merged by `mergeByCompile`
  const auto tasksFor = [&](const std::string &file, const std::string &shortName, const auto &tc, const auto &vars,
                            const std::string &label, const std::string &value) {
    const auto varsWithLabel = vars ^ append(std::pair{cfg.defaultsLabelVar, label});
    if (!tc.targets.empty()) {
      const auto caseArch =
          vars ^ collect_first([&](const auto &k, const auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; });
      const auto canonical = [](const std::string &name) {
        return compiletime::TargetSpec::findByName(name.substr(0, name.find('@'))) ^ map([](const auto &s) { return s.canonical; });
      };
      const auto wanted = caseArch ^ flat_map(canonical);
      if (!wanted || !(tc.targets ^ exists([&](const auto &t) { return canonical(t) == wanted; }))) return std::vector<Task>{};
    }
    std::vector<std::pair<std::string, std::string>> targetVars;
    if (!cfg.targetVar.empty()) {
      const auto targetArch =
          vars ^ collect_first([&](const auto &k, const auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; });
      const auto targetValue = targetArch ^ flat_map(targetValueFor);
      if (!targetValue) return std::vector<Task>{};
      targetVars.emplace_back(cfg.targetVar, *targetValue);
    }
    const auto augment = [&](const auto &bound) {
      return bound //
             ^ concat(extraVars) ^ concat(targetVars)
             ^ concat(std::vector<std::pair<std::string, std::string>>{
                 {cfg.defaultsVar, value},
                 {cfg.stdpar.first, fmt::vformat(cfg.stdpar.second, mkArgStore(bound ^ concat(targetVars)))},
                 {"input", file},
                 {"libm", libmFlag}});
    };
    const auto compileVars =
        varsWithLabel ^ map([&](const auto &k, const auto &v) { return std::pair{k, k == cfg.archVar ? compileTargetOf(v) : v}; });
    const auto compileAugmented = augment(compileVars);
    const auto augmented = augment(varsWithLabel);
    const auto unevalStore = mkArgStore(compileAugmented ^ append(std::pair{std::string("output"), std::string("<unevaluated>")}));
    const auto runsCmd = tc.runs ^ mk_string("", [&](const auto &r) { return fmt::vformat(r.command, unevalStore); });
    const auto varsKey = compileVars ^ mk_string("|", [](const auto &k, const auto &v) { return k + "=" + v; });
    const auto runsHash = std::hash<std::string>{}(runsCmd + '\0' + varsKey + '\0' + tc.name);
    const auto arch = varsWithLabel                                                                                                     //
                      ^ collect_first([&](const auto &k, const auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; }) //
                      ^ get_or_else(std::string{});
    const bool physical = (arch ^ starts_with("cuda")) || (arch ^ starts_with("hsa")) || (arch ^ starts_with("hip"));
    const auto baseOutput =
        fmt::format("{}{}_{:08x}", cfg.outputPrefix, shortName.empty() ? "anon" : shortName, static_cast<std::uint32_t>(runsHash));
    const auto pchSpec =
        tc.runs.empty() || tc.compileFailure || cfg.preamble.empty()
            ? std::optional<pch::Spec>{}
            : pch::specFor(cfg.preamble, file, tc.runs[0].command, [&](const std::string &t) { return fmt::vformat(t, unevalStore); });
    const auto caseModes = tc.offloadOnly ? (modes ^ filter([](const auto &m) { return m == Mode::Offload; })) : modes;
    return caseModes ^ map([&](const auto &mode) -> Task {
             const auto dir = fmt::format("{}_{}", baseOutput, modeName(mode));
             const auto output = dir + "/" + OutputName;
             const auto compileStore = mkArgStore(compileAugmented ^ append(std::pair{std::string("output"), output}));
             const auto store = mkArgStore(augmented ^ append(std::pair{std::string("output"), output}));
             const auto expand = [](const auto &r, const auto &bound) {
               return TestCase::Run{fmt::vformat(r.command, bound), r.expect, r.stderrCounts, r.failure};
             };
             Task task{
                 mode, file,          tc.name,           varsWithLabel,
                 dir,  output,        tc.compileFailure, tc.runs ^ take(1) ^ map([&](const auto &r) { return expand(r, compileStore); }),
                 {},   tc.timeoutSecs};
             task.pchSpec = pchSpec;
             const auto steps = tc.runs ^ drop(1) ^ map([&](const auto &r) { return expand(r, store); });
             using Mirror = std::pair<std::string, std::vector<std::pair<std::string, std::string>>>;
             const auto mirrors = mode == Mode::Offload && physical
                                      ? std::vector<Mirror>{{"", {}}, // compiletime (default)
                                                            {"runtime", {{std::string(env::PolyrtMirror), "runtime"}}}, // generic SMA walk
                                                            {"usm", {{std::string(env::PolyrtMirror), "off"}}}}         // USM; 77 if no USM
                                      : std::vector<Mirror>{{"", {}}};
             task.variants = mirrors ^ map([&](const auto &suffix, const auto &mirrorEnv) {
                               return RunVariant{task.id(), arch, suffix, mirrorEnv, steps, task.display()};
                             });
             return task;
           });
  };

  const auto files =
      source.empty() ? cfg.testFiles : cfg.testFiles ^ filter([&](const auto &f) { return llvm::sys::path::filename(f) == source; });
  const auto all = files ^ flat_map([&](const auto &file) {
                     std::ifstream src(file, std::ios::in | std::ios::binary);
                     const auto cases = TestCase::parseTestCase(src, cfg.directive, {{cfg.archVar, targets}});
                     const auto shortName = extractTestName(file);
                     return cases ^ flat_map([&](const auto &tc) -> std::vector<Task> {
                              if (!matches(shortName, tc.name)) return {};
                              return tc.matrices ^ flat_map([&](const auto &vars) {
                                       return cfg.defaultsVariants ^ flat_map([&](const auto &label, const auto &value) {
                                                return tasksFor(file, shortName, tc, vars, label, value);
                                              });
                                     });
                            });
                   });

  return mergeByCompile(all ^ distinct_by([&](const Task &t) -> std::string {
                          if (t.mode != Mode::Passthrough) return t.id();
                          return fmt::format("P|{}|{}|{}", t.testFile, t.caseName,
                                             t.variables ^ mk_string("|", [&](const auto &k, const auto &v) {
                                               return k + "=" + (k == cfg.archVar ? std::string{} : v);
                                             }));
                        }),
                        cfg);
}

// NOLINTBEGIN(readability-redundant-member-init)
struct RunnerOptions {
  std::vector<std::string> caseFilters = {};
  std::string runTask = {};
  std::string compileTask = {};
  std::string runOnlyTask = {};
  std::string cleanupTask = {};
  std::string source = {};
  int compileJobs = -1;
  bool verbose = false;
  bool offload = true;
  bool passthrough = true;
  bool list = false;
  bool listIds = false;
  bool listShards = false;
  bool listBucketWeights = false;
  std::string extraWeights = {};
  bool listTargets = false;
  std::string emitFile = {};
  std::string emitEnv = {};
  std::string emitPrefix = {};
  std::string emitBinary = {};
  std::string emitWorkdir = {};
  std::string emitDistFile = {};
  std::string emitDistBinary = {};
  std::string emitDistEnv = {};
};
// NOLINTEND(readability-redundant-member-init)

// a task hashes into one of ShardBuckets buckets; a slice is a set of buckets of one shard, so a compile group never splits
inline constexpr unsigned ShardBuckets = 64;
// one run entry is one compile or a tenth of one, so run entries track compile count
inline constexpr std::size_t SliceRuns = 2000;

inline unsigned bucketOf(std::string_view source) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const unsigned char c : source)
    h = (h ^ c) * 0x100000001b3ULL;
  return static_cast<unsigned>((h ^ (h >> 32)) % ShardBuckets);
}

inline std::string sourceOf(const Task &t) { return std::string(llvm::sys::path::filename(t.testFile)); }

inline std::string sliceLabel(const std::string &shard, unsigned bucket) { return fmt::format("{}~{:02}", shard, bucket); }

inline unsigned sliceCount(std::size_t runs) {
  return static_cast<unsigned>(std::clamp<std::size_t>((runs + SliceRuns - 1) / SliceRuns, 1, ShardBuckets));
}

struct Slice {
  std::string id;
  std::string shard;
  std::string labels; // ctest -L regex selecting exactly this slice
};

// greedy longest-processing-time: heaviest bucket first into the lightest slice, ties to the lowest index
inline std::vector<std::vector<unsigned>> assignBuckets(const std::vector<std::size_t> &weights, unsigned count) {
  std::vector<unsigned> order(weights.size());
  for (unsigned b = 0; b < order.size(); ++b)
    order[b] = b;
  std::stable_sort(order.begin(), order.end(), [&](unsigned l, unsigned r) { return weights[l] > weights[r]; });
  std::vector<std::vector<unsigned>> slices(count);
  std::vector<std::size_t> load(count, 0);
  for (const auto b : order) {
    const auto lightest = static_cast<std::size_t>(std::min_element(load.begin(), load.end()) - load.begin());
    slices[lightest].push_back(b);
    load[lightest] += weights[b];
  }
  for (auto &bs : slices)
    std::sort(bs.begin(), bs.end());
  return slices;
}

inline Slice sliceOf(const std::string &shard, unsigned index, unsigned count, const std::vector<unsigned> &buckets) {
  if (count == 1) return {shard, shard, fmt::format("^{}$", shard)};
  return {fmt::format("{}.{}of{}", shard, index + 1, count), shard,
          fmt::format("^{}~({})$", shard, buckets ^ mk_string("|", [](unsigned b) { return fmt::format("{:02}", b); }))};
}

using BucketWeights = std::unordered_map<std::string, std::vector<std::size_t>>;

// run entries per bucket of each shard
inline BucketWeights bucketWeights(const std::vector<Task> &tasks, const DriverConfig &cfg) {
  BucketWeights weights;
  for (const auto &t : tasks) {
    if (detail::archFor(t, cfg).empty()) continue;
    auto &w = weights.try_emplace(ctestSafe(detail::compileTargetFor(t, cfg)), ShardBuckets, 0).first->second;
    w[bucketOf(sourceOf(t))] += t.variants.size();
  }
  return weights;
}

// `shard<TAB>bucket<TAB>runs` per non-empty bucket
inline std::string formatBucketWeights(const BucketWeights &weights) {
  std::vector<std::string> shards;
  for (const auto &[shard, w] : weights)
    shards.push_back(shard);
  std::sort(shards.begin(), shards.end());
  std::string out;
  for (const auto &shard : shards)
    for (unsigned b = 0; b < ShardBuckets; ++b)
      if (const auto runs = weights.at(shard)[b]; runs > 0) out += fmt::format("{}\t{}\t{}\n", shard, b, runs);
  return out;
}

inline void addBucketWeights(BucketWeights &weights, const std::string &text) {
  for (const auto &line : text ^ split('\n')) {
    const auto fields = line ^ split('\t');
    if (fields.size() != 3) continue;
    const auto bucket = static_cast<unsigned>(std::stoul(fields[1]));
    if (bucket >= ShardBuckets) throw std::runtime_error("polytest: bucket out of range in weights: " + line);
    weights.try_emplace(fields[0], ShardBuckets, 0).first->second[bucket] += std::stoull(fields[2]);
  }
}

// one slice per SliceRuns run entries of a shard, in profile order; whole source files go to the lightest slice
inline std::vector<Slice> listSlices(const std::vector<std::string> &targets, const BucketWeights &weights) {
  return targets ^ map([](const auto &t) { return ctestSafe(compileTargetOf(t)); }) ^ distinct() ^ flat_map([&](const auto &shard) {
           const auto it = weights.find(shard);
           const auto w = it == weights.end() ? std::vector<std::size_t>(ShardBuckets, 0) : it->second;
           const auto n = sliceCount(std::accumulate(w.begin(), w.end(), std::size_t{0}));
           const auto assigned = assignBuckets(w, n);
           std::vector<Slice> slices;
           for (unsigned i = 0; i < n; ++i)
             slices.push_back(sliceOf(shard, i, n, assigned[i]));
           return slices;
         });
}

inline std::string runLabels(const RunVariant &v, unsigned bucket) {
  if (v.target.empty()) return {};
  const auto device = ctestSafe(v.target);
  const auto compile = ctestSafe(compileTargetOf(v.target));
  const auto slice = sliceLabel(compile, bucket);
  return device == compile ? fmt::format("codegen;{};{}", device, slice) : fmt::format("codegen;{};{};{}", device, compile, slice);
}

inline void emitCtest(const std::vector<Task> &tasks, const DriverConfig &cfg, const std::string &file, const std::string &prefix,
                      const std::string &binary, const std::string &workdir, const std::string &env) {
  const auto entries =
      tasks ^ map([&](const auto &t) {
        const auto source = std::string(llvm::sys::path::filename(t.testFile));
        const auto timeout = t.timeoutSecs ? t.timeoutSecs : cfg.defaultTimeoutSecs;
        const auto target = detail::archFor(t, cfg);
        const auto bucket = bucketOf(source);
        if (t.variants.size() == 1 && t.variants[0].suffix.empty())
          return CtestEntry{t.id(), target.empty() ? std::string{} : runLabels(t.variants[0], bucket), {}, source, timeout};
        const auto compile = ctestSafe(compileTargetOf(target));
        const auto labels = target.empty() ? std::string{} : fmt::format("codegen;{};{}", compile, sliceLabel(compile, bucket));
        const auto vs =
            t.variants ^ map([&](const auto &v) {
              return CtestVariant{v.suffix, v.env ^ mk_string(";", [](const auto &k, const auto &val) { return k + "=" + val; }), v.id,
                                  runLabels(v, bucket)};
            });
        return CtestEntry{t.id(), labels, vs, source, timeout};
      });
  emitCtestFragment(file, prefix, binary, workdir, env, entries);
}

inline const char *statusTag(TaskStatus s) {
  switch (s) {
    case TaskStatus::Pass: return "PASS";
    case TaskStatus::Skip: return "SKIP";
    case TaskStatus::Fail: return "FAIL";
  }
  return "?";
}

// the device-level runs (the default mirror variant) of a task
inline std::vector<const RunVariant *> deviceRuns(const Task &t) {
  std::vector<const RunVariant *> runs;
  for (const auto &v : t.variants)
    if (v.suffix.empty()) runs.push_back(&v);
  return runs;
}

// a task id selects all its device runs, a run id selects that run alone
inline std::vector<Task> selectTasks(const std::vector<Task> &tasks, const std::string &id) {
  if (const auto whole = tasks ^ filter([&](const auto &t) { return t.id() == id; }); !whole.empty()) return whole;
  return tasks ^ collect([&](const auto &t) -> std::optional<Task> {
           const auto run = t.variants ^ filter([&](const auto &v) { return v.suffix.empty() && v.id == id; });
           if (run.empty()) return std::nullopt;
           auto only = t;
           only.variants = run;
           return only;
         });
}

inline int runTasks(const DriverConfig &cfg, const RunnerOptions &opts) {
  // XXX ROCr writes a ~150 MB gpucore.<pid> per GPU VM fault (honours RLIMIT_CORE)
  disableUncollectedCoreDumps();
  if (opts.listTargets) {
    loadTestTargets(cfg.profileDir) | for_each([](const auto &t) { std::fprintf(stdout, "%s\n", t.c_str()); });
    return 0;
  }
  if (opts.listShards || opts.listBucketWeights) {
    const auto tasks = enumerateTasks(cfg, opts.offload, opts.passthrough, opts.caseFilters, opts.source);
    auto weights = bucketWeights(tasks, cfg);
    if (opts.listBucketWeights) {
      std::fputs(formatBucketWeights(weights).c_str(), stdout);
      return 0;
    }
    if (!opts.extraWeights.empty()) addBucketWeights(weights, polyregion::read_string(opts.extraWeights));
    listSlices(loadTestTargets(cfg.profileDir), weights)
        | for_each([](const auto &s) { std::fprintf(stdout, "%s\t%s\t%s\n", s.id.c_str(), s.shard.c_str(), s.labels.c_str()); });
    return 0;
  }
  auto allTasks = enumerateTasks(cfg, opts.offload, opts.passthrough, opts.caseFilters, opts.source);
  // single-task compile / run / cleanup, driven by ctest fixtures for shared-compile tasks
  const auto single = [&]() -> std::string {
    if (!opts.compileTask.empty()) return opts.compileTask;
    if (!opts.runOnlyTask.empty()) return opts.runOnlyTask;
    return opts.cleanupTask;
  }();
  if (!single.empty()) {
    const bool compiles = !opts.compileTask.empty(), cleans = !opts.cleanupTask.empty();
    const auto selected =
        compiles || cleans ? allTasks ^ filter([&](const auto &t) { return t.id() == single; }) : selectTasks(allTasks, single);
    if (selected.empty()) return std::fprintf(stderr, "polytest: no task with id '%s'\n", single.c_str()), 1;
    const Task &t = selected.front();
    if (cleans) {
      detail::removeArtifacts(t.dir);
      return 0;
    }
    const auto o = compiles ? detail::compileTask(t, cfg) : detail::runTask(t, cfg, *deviceRuns(t).front());
    if (o.status == TaskStatus::Fail) {
      if (!o.failReason.empty()) std::fprintf(stderr, "  reason: %s\n", o.failReason.c_str());
      if (!o.cmdline.empty()) std::fprintf(stderr, "  cmd: %s\n", o.cmdline.c_str());
      std::fprintf(stderr, "  stdout:\n%s\n  stderr:\n%s\n", o.stdoutCapture.c_str(), o.stderrCapture.c_str());
    }
    if (o.status == TaskStatus::Skip) return reportSkip(t.id(), o.failReason), 77;
    return o.status == TaskStatus::Pass ? 0 : 1;
  }
  if (!opts.runTask.empty()) {
    allTasks = selectTasks(allTasks, opts.runTask);
    if (allTasks.empty()) {
      std::fprintf(stderr, "polytest: no task with id '%s'\n", opts.runTask.c_str());
      return 1;
    }
  }
  const auto &tasks = allTasks;
  if (tasks.empty()) {
    std::fprintf(stderr, "polytest: no tasks discovered (filters=%zu)\n", opts.caseFilters.size());
    return 0;
  }
  if (opts.listIds) {
    const auto ids = tasks ^ flat_map([](const auto &t) { return deviceRuns(t) ^ map([](const auto *v) { return v->id; }); });
    ids | for_each([](const auto &id) { std::fprintf(stdout, "%s\n", id.c_str()); });
    // XXX duplicate ids shadow tasks: ctest registers both names but --run-task resolves the first
    const auto groupedIds = ids ^ group_by([](const auto &id) { return id; });
    const auto dups = groupedIds | collect([](const auto &id, const auto &xs) -> std::optional<std::string> {
                        return xs.size() > 1 ? std::optional{id} : std::nullopt;
                      }) //
                      | to_vector();
    dups | for_each([](const auto &id) { std::fprintf(stderr, "polytest: duplicate task id '%s'\n", id.c_str()); });
    return dups.empty() ? 0 : 1;
  }
  if (!opts.emitFile.empty()) {
    emitCtest(tasks, cfg, opts.emitFile, opts.emitPrefix, opts.emitBinary, opts.emitWorkdir, opts.emitEnv);
    if (!opts.emitDistFile.empty()) emitCtest(tasks, cfg, opts.emitDistFile, opts.emitPrefix, opts.emitDistBinary, {}, opts.emitDistEnv);
    return 0;
  }
  if (opts.list) {
    tasks | for_each([](const auto &t) {
      deviceRuns(t) | for_each([](const auto *v) { std::fprintf(stdout, "%s\n", v->display.c_str()); });
    });
    return 0;
  }

  const auto requestedJobs =
      opts.compileJobs > 0 ? static_cast<std::size_t>(opts.compileJobs) : std::max(1U, std::thread::hardware_concurrency());
  const auto jobs = std::min(requestedJobs, tasks.size());
  const auto runCount = tasks ^ fold_left(std::size_t{0}, [](const auto &acc, const auto &t) { return acc + deviceRuns(t).size(); });
  std::fprintf(stderr, "polytest: %zu compiles, %zu runs; compile -j%zu, run -j1\n", tasks.size(), runCount, jobs);

  std::vector<TaskOutcome> compileOutcomes(tasks.size());
  std::atomic<std::size_t> nextIdx{0};
  std::atomic<std::size_t> compileDone{0};
  std::mutex logMtx;

  auto dumpDetails = [&](const TaskOutcome &o, bool includeStdout) {
    if (o.status != TaskStatus::Fail && !opts.verbose) return;
    if (!o.failReason.empty()) std::fprintf(stderr, "  reason: %s\n", o.failReason.c_str());
    if (!o.cmdline.empty()) std::fprintf(stderr, "  cmd: %s\n", o.cmdline.c_str());
    if (includeStdout) std::fprintf(stderr, "  stdout:\n%s\n", o.stdoutCapture.c_str());
    std::fprintf(stderr, "  stderr:\n%s\n", o.stderrCapture.c_str());
  };

  const auto compileStart = std::chrono::steady_clock::now();
  const auto worker = [&] {
    for (;;) {
      const auto i = nextIdx.fetch_add(1);
      if (i >= tasks.size()) return;
      auto &o = (compileOutcomes[i] = detail::compileTask(tasks[i], cfg));
      const auto done = ++compileDone;
      const std::scoped_lock lk(logMtx);
      std::fprintf(stderr, "[compile %zu/%zu] %s %s (%.2fs)\n", done, tasks.size(), statusTag(o.status), tasks[i].display().c_str(),
                   o.secs);
      // XXX include stdout on fail; MSVC link emits LNK1120 etc. to stdout, not stderr.
      dumpDetails(o, /*includeStdout*/ true);
      // XXX Drop captures on Pass to bound peak RSS over a 1500-task batch; only failures need to keep them for the summary.
      if (o.status == TaskStatus::Pass && !opts.verbose) std::string{}.swap(o.stdoutCapture), std::string{}.swap(o.stderrCapture);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(jobs);
  for (std::size_t t = 0; t < jobs; ++t)
    pool.emplace_back(worker);
  pool | for_each([](auto &t) { t.join(); });
  const auto compileSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - compileStart).count();

  std::vector<TaskOutcome> finalOutcomes;
  std::vector<std::string> finalDisplays;
  const auto runStart = std::chrono::steady_clock::now();
  std::size_t runIdx = 0;
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto &co = compileOutcomes[i];
    bool cleanable = true;
    for (const auto *variant : deviceRuns(tasks[i])) {
      ++runIdx;
      finalDisplays.push_back(variant->display);
      if (co.status != TaskStatus::Pass) {
        finalOutcomes.push_back(co);
        cleanable = false;
        if (co.status == TaskStatus::Skip) reportSkip(variant->id, co.failReason);
        const char *label = co.status == TaskStatus::Skip ? "compile-skip" : "compile-fail";
        std::fprintf(stderr, "[run %zu/%zu] %s %s (%s)\n", runIdx, runCount, statusTag(co.status), variant->display.c_str(), label);
        continue;
      }
      const auto out = detail::runTask(tasks[i], cfg, *variant);
      finalOutcomes.push_back(out);
      if (out.status == TaskStatus::Skip) reportSkip(variant->id, out.failReason);
      else if (out.status == TaskStatus::Fail) cleanable = false;
      std::fprintf(stderr, "[run %zu/%zu] %s %s (%.2fs)\n", runIdx, runCount, statusTag(out.status), variant->display.c_str(), out.secs);
      dumpDetails(out, /*includeStdout*/ true);
    }
    if (cfg.cleanupOnSuccess && cleanable) detail::removeArtifacts(tasks[i].dir);
  }
  const auto runSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - runStart).count();

  const auto countBy = [&](TaskStatus s) { return finalOutcomes | count([s](const auto &o) { return o.status == s; }); };
  const auto pass = countBy(TaskStatus::Pass);
  const auto skip = countBy(TaskStatus::Skip);
  const auto fail = countBy(TaskStatus::Fail);
  if (fail > 0) {
    const auto failedDisplay = finalDisplays                                                                      //
                               | zip(finalOutcomes)                                                               //
                               | filter([](const auto &, const auto &o) { return o.status == TaskStatus::Fail; }) //
                               | mk_string("\n", [](const auto &d, const auto &) { return "  - " + d; });
    std::fprintf(stderr, "polytest: failures (%zu):\n%s\n", fail, failedDisplay.c_str());
  }
  std::fprintf(stderr, "polytest: pass=%zu skip=%zu fail=%zu total=%zu compile=%.1fs run=%.1fs\n", pass, skip, fail, finalOutcomes.size(),
               compileSecs, runSecs);
  if (fail > 0) return 1;
  if (!opts.runTask.empty() && skip > 0) return 77;
  return 0;
}

namespace detail {
// XXX fire introspects `fired_main` by name with default-arg `fire::arg(...)` placeholders, so the
// driver config has to be threaded in through a side channel rather than as a normal parameter.
inline const DriverConfig *firedCfg = nullptr;
} // namespace detail

inline int fired_main( //
    fire::optional<std::string> caseFilter = fire::arg({"-c", "--case", "Filter by file shortname or case name"}),
    fire::optional<std::string> runTask = fire::arg({"--run-task", "Run exactly one task by id (see --list-ids)"}),
    fire::optional<std::string> compileTask = fire::arg({"--compile-task", "Compile one task by id, no run (ctest fixture setup)"}),
    fire::optional<std::string> runOnlyTask = fire::arg({"--run-only-task", "Run one already-compiled task by id, no compile"}),
    fire::optional<std::string> cleanupTask = fire::arg({"--cleanup-task", "Remove one task's binary (ctest fixture cleanup)"}),
    fire::optional<std::string> source = fire::arg({"--source",
                                                    "Enumerate only the test file with this name (with the single-task options)"}),
    int jobs = fire::arg({"-j", "--jobs", "Compile parallelism (default: hardware_concurrency)"}, -1),
    bool verbose = fire::arg({"-v", "--verbose", "Dump stdout/stderr for every task"}),
    bool offloadOnly = fire::arg({"--offload-only", "Skip passthrough mode"}),
    bool passthroughOnly = fire::arg({"--passthrough-only", "Skip offload mode"}),
    bool list = fire::arg({"-l", "--list", "Enumerate tasks (human-readable) and exit"}),
    bool listIds = fire::arg({"--list-ids", "Enumerate task ids (one per line) and exit"}),
    bool listShards = fire::arg({"--list-shards", "Enumerate slices (id, shard label, ctest -L regex; tab separated) and exit"}),
    bool listBucketWeights = fire::arg({"--list-bucket-weights", "Enumerate run entries per shard and bucket (tab separated) and exit"}),
    fire::optional<std::string> extraWeights = fire::arg({"--extra-weights", "Weights file of another suite to add (with --list-shards)"}),
    bool listTargets = fire::arg({"--list-targets", "Enumerate the profile's test targets (one per line) and exit"}),
    fire::optional<std::string> emitCtest = fire::arg({"--emit-ctest", "Write a CTestTestfile fragment to this path and exit"}),
    fire::optional<std::string> emitPrefix = fire::arg({"--emit-prefix", "ctest test-name prefix (with --emit-ctest)"}),
    fire::optional<std::string> emitBinary = fire::arg({"--emit-binary", "Binary ctest invokes (with --emit-ctest)"}),
    fire::optional<std::string> emitWorkdir = fire::arg({"--emit-workdir", "WORKING_DIRECTORY for emitted tests"}),
    fire::optional<std::string> emitEnv = fire::arg({"--emit-env", "ENVIRONMENT for emitted tests (with --emit-ctest)"}),
    fire::optional<std::string> emitDistFile = fire::arg({"--emit-dist-ctest", "Also write a dist CTestTestfile fragment here"}),
    fire::optional<std::string> emitDistBinary = fire::arg({"--emit-dist-binary", "Binary for the dist fragment"}),
    fire::optional<std::string> emitDistSubdir = fire::arg({"--emit-dist-subdir", "Dist test-files subdir; builds the dist ENVIRONMENT"})) {
  RunnerOptions opts{.compileJobs = jobs,
                     .verbose = verbose,
                     .offload = !passthroughOnly,
                     .passthrough = !offloadOnly,
                     .list = list,
                     .listIds = listIds,
                     .listShards = listShards,
                     .listBucketWeights = listBucketWeights,
                     .listTargets = listTargets};
  if (caseFilter) opts.caseFilters.push_back(caseFilter.value());
  if (runTask) opts.runTask = runTask.value();
  if (compileTask) opts.compileTask = compileTask.value();
  if (runOnlyTask) opts.runOnlyTask = runOnlyTask.value();
  if (cleanupTask) opts.cleanupTask = cleanupTask.value();
  if (source) opts.source = source.value();
  if (extraWeights) opts.extraWeights = extraWeights.value();
  if (emitCtest) opts.emitFile = emitCtest.value();
  if (emitPrefix) opts.emitPrefix = emitPrefix.value();
  if (emitBinary) opts.emitBinary = emitBinary.value();
  if (emitWorkdir) opts.emitWorkdir = emitWorkdir.value();
  if (emitEnv) opts.emitEnv = emitEnv.value();
  if (emitDistFile) opts.emitDistFile = emitDistFile.value();
  if (emitDistBinary) opts.emitDistBinary = emitDistBinary.value();
  if (emitDistSubdir) opts.emitDistEnv = distEnv(emitDistSubdir.value());
  return runTasks(*detail::firedCfg, opts);
}

inline int runMain(int argc, const char **argv, const DriverConfig &cfg) {
  const auto chdirTo = [](const char *what, llvm::StringRef dir) {
    if (const auto ec = llvm::sys::fs::create_directories(dir); ec) {
      std::fprintf(stderr, "polytest: failed to create %s='%s': %s\n", what, dir.str().c_str(), ec.message().c_str());
      return;
    }
    if (const auto ec = llvm::sys::fs::set_current_path(dir); ec)
      std::fprintf(stderr, "polytest: failed to chdir to %s='%s': %s\n", what, dir.str().c_str(), ec.message().c_str());
  };
  if (const char *const override = std::getenv(polyregion::env::PolytestWorkDir); override && *override)
    chdirTo(polyregion::env::PolytestWorkDir, override);
  else if (!cfg.workDir.empty()) chdirTo("workDir", cfg.workDir);
  defaultCacheDirsToCwd();
  detail::firedCfg = &cfg;
  constexpr const char *descr = "polytest runner: parallel compile, serial run";
  // NOLINTNEXTLINE(misc-const-correctness)
  PREPARE_FIRE_(argc, argv, false, fired_main, descr);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, descr));
  return FIRE_EXTRACT_1_PAD_(fired_main, descr)();
}

} // namespace polyregion::polytest
