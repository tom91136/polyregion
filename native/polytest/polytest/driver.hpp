#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "fmt/args.h"
#include "fmt/format.h"

#include "polyinvoke/device_lock.h"
#include "polyregion/env_keys.h"
#include "polyregion/io.hpp"

#include "fire.hpp"
#include "polytest/ctest_emit.hpp"
#include "polytest/lit.hpp"
#include "polytest/profile.hpp"

#ifndef _WIN32
  #include <sys/resource.h>
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
  std::pair<std::string, std::string> stdpar;
  std::string driverEnvVar;
  std::vector<std::string> passthroughEnvs;
  std::string outputPrefix;
  std::string tempPrefix;
  std::string directive;
  bool cleanupOnSuccess;
};

enum class Mode : std::uint8_t { Offload, Passthrough };
inline constexpr const char *modeName(Mode m) { return m == Mode::Offload ? "offload" : "passthrough"; }

struct RunVariant {
  std::string suffix;
  std::vector<std::pair<std::string, std::string>> env;
};

struct Task {
  Mode mode;
  std::string testFile;
  std::string caseName;
  std::vector<std::pair<std::string, std::string>> variables;
  std::string output;
  std::vector<TestCase::Run> runs; // templates already expanded against (variables + defaults + stdpar + input + output)
  std::vector<RunVariant> variants;

  std::string display() const {
    return fmt::format("[{}] {}/{} {}", modeName(mode), extractTestName(testFile), caseName,
                       variables | mk_string(" ", [](auto &k, auto &v) { return k + "=" + v; }));
  }

  // ctest-name-safe stable id: alnum + `_` + `-` only
  std::string id() const {
    auto safe = [](std::string_view s) {
      return std::string(s) ^ mk_string("", [](char c) {
               return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-'
                          ? std::string(1, c)
                          : fmt::format("_{:02x}", static_cast<unsigned char>(c));
             });
    };
    auto vars = variables | mk_string("", [&](auto &k, auto &v) { return "-" + safe(k) + "_" + safe(v); });
    return fmt::format("{}-{}-{}{}", modeName(mode), safe(extractTestName(testFile)), safe(caseName), vars);
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
  return t.variables                                                                                           //
         ^ collect_first([&](auto &k, auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; }) //
         ^ get_or_else(std::string{});
}

inline std::vector<std::string> baseEnvs(const Task &t, const DriverConfig &cfg, bool runStep) {
  std::vector<std::pair<std::string, std::string>> kvs;
  auto put = [&](std::string_view k, std::string_view v) {
    for (auto &e : kvs)
      if (e.first == k) {
        e.second = std::string(v);
        return;
      }
    kvs.emplace_back(std::string(k), std::string(v));
  };
  auto putKv = [&](const std::string &kv) {
    const auto eq = kv.find('=');
    if (eq != std::string::npos) put(kv.substr(0, eq), kv.substr(eq + 1));
  };
  if (t.mode == Mode::Passthrough)
    for (auto &kv : cfg.passthroughEnvs)
      putKv(kv);
  put(cfg.driverEnvVar, cfg.driverPath);
  put(polyregion::env::PolyrtPlatform, archFor(t, cfg));
  put(polyregion::env::PolyrtHostFallback, "0");
  put(polyregion::env::PolyrtStrictSelect, "1");
  // XXX NVIDIA's shader disk cache (~/.cache/nvidia/GLCache) bloats to 100k+ files across a sweep and the
  // driver stat-scans it on every init (~20s spin); one-shot test kernels gain nothing, so disable it
  put("__GL_SHADER_DISK_CACHE", "0");
  // XXX rusticl loads Mesa's LLVM in-process; the ICD loader pulls it in for ANY OpenCL target, clashing
  // with polydco/polystl's LLVM (two libLLVMSPIRVLib) -> bad_alloc. expose it only when it's selected
  const bool rusticlTarget = archFor(t, cfg).find("rusticl") != std::string::npos;
  for (auto &kv : loadProfileEnv(cfg.profileDir)) {
    if (!rusticlTarget && (kv ^ starts_with("RUSTICL_ENABLE="))) continue;
    putKv(kv);
  }
  if (const auto v = std::getenv(polyregion::env::PolyinvokeTestLock)) put(polyregion::env::PolyinvokeTestLock, v);
  else put(polyregion::env::PolyinvokeTestLock, "1");
  if (const auto v = std::getenv("ASAN_OPTIONS")) put("ASAN_OPTIONS", v);
  else
    put("ASAN_OPTIONS", "alloc_dealloc_mismatch=0,detect_leaks=0,protect_shadow_gap=0,verify_asan_link_order=0,strip_env=0"
#ifdef __APPLE__
                        ",detect_container_overflow=0"
#endif
    );
  if (const auto v = std::getenv("UBSAN_OPTIONS")) put("UBSAN_OPTIONS", v);
  else put("UBSAN_OPTIONS", "print_stacktrace=1");
  if (std::getenv(polyregion::env::PolytestDebug)) put(polyregion::env::PolyrtDebug, "2");
  // child env is replaced wholesale; forward loader-lib + link-thread overrides or they vanish
  for (const auto *name : {polyregion::env::PolycppLinkThreads, polyregion::env::PolyfcLinkThreads,           //
                           polyregion::env::PolyinvokeDisableSvm, polyregion::env::PolyinvokeDisableBackends, //
                           polyregion::env::PolyrtMirror, polyregion::env::PolyregionPassLog,                 //
                           polyregion::env::PolyregionDebug,                                                  //
                           "CUEW_LIB_PATH", "HIPEW_LIB_PATH", "HSAEW_LIB_PATH"})
    if (auto v = std::getenv(name)) put(name, v);
  if (const auto a = archFor(t, cfg); a ^ starts_with("opencl")) put(polyregion::env::PolyinvokeDisableSvm, "1");
  std::string lavapipeLd;
  if (archFor(t, cfg) == "llvmpipe")
    if (const auto *emu = std::getenv(polyregion::env::PolyregionEmulatorsHome)) {
      put("VK_ICD_FILENAMES", std::string(emu) + "/lavapipe/share/vulkan/icd.d/lvp_icd.x86_64.json");
      lavapipeLd = std::string(emu) + "/lib:" + emu + "/lavapipe/lib64:";
    }
  {
    std::string pathVal = std::getenv("PATH") ? std::getenv("PATH") : "";
#if defined(_WIN32)
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
#if defined(__APPLE__)
  // XXX forward DYLD_*; compiled test binaries can't find dist libs otherwise (SIGTRAP)
  for (const auto *name : {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "TMPDIR", "HOME"}) {
    if (auto v = std::getenv(name)) put(name, v);
  }
#elif defined(__linux__)
  // XXX forward LD_LIBRARY_PATH (dist-relative .so when rpath misses; lavapipe must precede) and a parent
  // LD_PRELOAD shim into the wholesale-replaced child env; the run-only ASan block layers its rt ahead
  for (const auto *name : {"TMPDIR", "HOME", "LD_PRELOAD", "POLYRT_DUMP_KERNEL"})
    if (auto v = std::getenv(name)) put(name, v);
  const auto ld = std::getenv("LD_LIBRARY_PATH");
  if (const auto joined = lavapipeLd + (ld ? ld : ""); !joined.empty()) put("LD_LIBRARY_PATH", joined);
#elif defined(_WIN32)
  // XXX CreateProcess needs SYSTEMROOT; clang needs TEMP/TMP; link.exe needs LIB/INCLUDE for
  // CRT and SDK lookup (flang does not auto-populate -libpath). Inherit a vcvars-style env.
  for (const auto *name : {"SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE", "WINDIR", "PATHEXT", "COMSPEC", "ProgramFiles",
                           "ProgramFiles(x86)", "ProgramData", "LOCALAPPDATA", "APPDATA", "LIB", "INCLUDE", "LIBPATH"}) {
    if (auto v = std::getenv(name)) put(name, v);
  }
#endif
  // XXX run-only: the child loads the instrumented runtime late, so force asan in first; compile steps
  // must not (would reach the arm64e system ld / 3rd-party ctest, which cannot load the runtime)
#if defined(__APPLE__) || defined(__linux__)
  if (runStep)
    if (const char *rt = std::getenv(polyregion::env::PolytestAsanPreload); rt && *rt) {
  #if defined(__APPLE__)
      put("DYLD_INSERT_LIBRARIES", rt);
  #else
      const auto prev = std::getenv("LD_PRELOAD");
      put("LD_PRELOAD", prev ? std::string(rt) + ":" + prev : std::string(rt));
  #endif
    }
#endif
  return kvs ^ map([](auto &k, auto &v) { return k + "=" + v; });
}

inline std::string resolveBin(const std::string &name, const DriverConfig &cfg) {
  // XXX Prefer cwd for generated test binaries: a stale copy in binaryDir from a previous build would otherwise shadow the fresh one.
  auto exists = [](llvm::StringRef p) -> std::optional<std::string> {
    if (llvm::sys::fs::exists(p)) return std::string(p);
#ifdef _WIN32
    auto withExe = std::string(p) + ".exe";
    if (llvm::sys::fs::exists(withExe)) return withExe;
#endif
    return std::nullopt;
  };
  auto toAbs = [](llvm::StringRef p) {
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

inline StepResult runStep(const Task &task, const DriverConfig &cfg, const std::string &command, bool isRunStep) {
  auto fragments = command ^ split(' ');
  auto [envBits, args] = fragments ^ span([](auto &x) { return x ^ contains('='); });

  auto envs = baseEnvs(task, cfg, isRunStep) ^ filter([&](auto &e) {
                return !(envBits ^ exists([&](auto &kv) { return e ^ starts_with(kv.substr(0, kv.find('=') + 1)); }));
              });
  envs ^= concat(envBits);

  const std::vector<llvm::StringRef> envRefs = envs ^ map([](auto &x) -> llvm::StringRef { return x; });
  const std::vector<llvm::StringRef> argRefs = args ^ map([](auto &x) -> llvm::StringRef { return x; });

  // XXX TempFile holds an exclusive Windows handle that blocks the child's redirect; use createUniqueFile + FileRemover
  auto makeTempPath = [&](std::string_view suffix) -> std::optional<std::string> {
    llvm::SmallString<256> p;
    if (llvm::sys::fs::createUniqueFile(cfg.tempPrefix + std::string(suffix) + "-%%-%%-%%-%%-%%", p)) return std::nullopt;
    return std::string(p);
  };
  auto outPath = makeTempPath("stdout");
  if (!outPath) return {-1, {}, "Cannot create stdout temp", command};
  auto errPath = makeTempPath("stderr");
  if (!errPath) return {-1, {}, "Cannot create stderr temp", command};
  llvm::FileRemover outRemover(*outPath), errRemover(*errPath);

  if (args.empty()) return {-1, {}, "empty command", command};
  auto resolved = resolveBin(args[0], cfg);
  std::string execErr;
  unsigned secondsToWait = 0;
  if (const char *t = std::getenv(polyregion::env::PolytestTimeout)) {
    char *end = nullptr;
    const unsigned long v = std::strtoul(t, &end, 10);
    if (end != t && *end == '\0') secondsToWait = static_cast<unsigned>(v);
  }
  auto code = llvm::sys::ExecuteAndWait(resolved, argRefs, envRefs, {std::nullopt, *outPath, *errPath}, secondsToWait, 0, &execErr);

  auto stdoutText = polyregion::read_string(*outPath);
  auto stderrText = polyregion::read_string(*errPath);

  // XXX -2 is LLVM's signal-killed sentinel; surface WTERMSIG via the ErrMsg out-param.
  if (code == -2 && !execErr.empty()) stderrText += "\n[polytest] signal: " + execErr + "\n";

  std::string cmdline = argRefs | drop(1) | prepend(llvm::StringRef(resolved)) | mk_string(" ", [](auto &s) { return s.str(); });
  return {code, std::move(stdoutText), std::move(stderrText), std::move(cmdline)};
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

inline TaskOutcome compileTask(const Task &task, const DriverConfig &cfg) {
  if (task.runs.empty()) return {};
  const auto start = std::chrono::steady_clock::now();
  auto r = runStep(task, cfg, task.runs[0].command, /*isRunStep=*/false);
  const auto exit = r.exitCode;
  if (std::getenv(polyregion::env::PolyregionPassLog) && !r.stderrText.empty())
    std::fprintf(stderr, "[%s]\n%s\n", task.display().c_str(), r.stderrText.c_str());
  TaskOutcome o;
  std::string reproReason;
  if (exit == 0)
    if (const char *e = std::getenv(polyregion::env::PolytestReproCheck); !(e && e[0] == '0')) {
      const auto &cmd = task.runs[0].command;
      const auto toks = cmd ^ split(' ');
      const auto out = toks ^ sliding(2, 1) ^ fold_left(std::string{}, [](auto acc, auto &w) { //
                         return w.size() == 2 && w[0] == "-o" ? w[1] : acc;
                       });
      if (!out.empty()) {
        const std::string out2 = out + ".repro";
        const auto cmd2 = toks ^ mk_string(" ", [&](auto &t) { return t == out ? out2 : t; });
        auto r2 = runStep(task, cfg, cmd2, /*isRunStep=*/false);
        if (r2.exitCode != 0) reproReason = fmt::format("repro recompile failed (exit {})", r2.exitCode);
        else if (polyregion::read_string(out) != polyregion::read_string(out2))
          reproReason = "non-reproducible build: two compiles of the same source produced different objects";
        llvm::sys::fs::remove(out2);
      }
    }
  absorbStep(o, std::move(r));
  o.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (exit == 77) o.failReason = "polyrt: no compatible target during compile (exit 77)", o.status = TaskStatus::Skip;
  else if (exit != 0) o.failReason = fmt::format("compile failed (exit {})", exit), o.status = TaskStatus::Fail;
  else if (!reproReason.empty()) o.failReason = reproReason, o.status = TaskStatus::Fail;
  return o;
}

inline TaskOutcome runTask(const Task &task, const DriverConfig &cfg) {
  TaskOutcome o;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 1; i < task.runs.size(); ++i) {
    auto r = runStep(task, cfg, task.runs[i].command, /*isRunStep=*/true);
    const auto exit = r.exitCode;
    absorbStep(o, std::move(r));
    if (exit == 77) {
      o.status = TaskStatus::Skip;
      o.failReason = "polyrt: no compatible target (exit 77)";
      break;
    }
    if (exit != 0) {
      o.status = TaskStatus::Fail;
      o.failReason = fmt::format("run step {} failed (exit {})", i, exit);
      break;
    }
    const auto lines = o.stdoutCapture ^ split('\n');
    if (!(task.runs[i].expect | forall([&](const auto &e) { return checkExpect(e, lines, o.stdoutCapture, o); }))) break;
  }
  o.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return o;
}

} // namespace detail

inline std::vector<Task> enumerateTasks(const DriverConfig &cfg, bool offload, bool passthrough,
                                        const std::vector<std::string> &caseFilters) {
  auto mkArgStore = [](const std::vector<std::pair<std::string, std::string>> &xs) {
    fmt::dynamic_format_arg_store<fmt::format_context> s;
    xs | for_each([&](auto &k, auto &v) { s.push_back(fmt::arg(k.c_str(), v)); });
    return s;
  };
  const auto matches = [&](const std::string &shortName, const std::string &caseName) {
    return caseFilters.empty() || (caseFilters | exists([&](auto &f) { return f == shortName || f == caseName; }));
  };
  const std::string libmFlag =
#if defined(_WIN32)
      "";
#else
      "-lm";
#endif
  std::vector<Mode> modes;
  if (offload) modes.emplace_back(Mode::Offload);
  if (passthrough) modes.emplace_back(Mode::Passthrough);
  const auto targets = loadTestTargets(cfg.profileDir);

  // one Task per mode for a (case, matrix-row, defaults-variant) combination
  const auto tasksFor = [&](const std::string &file, const std::string &shortName, const auto &tc, const auto &vars,
                            const std::string &label, const std::string &value) {
    const auto varsWithLabel = vars ^ append(std::pair{cfg.defaultsLabelVar, label});
    const auto augmented = varsWithLabel ^ concat(std::vector<std::pair<std::string, std::string>>{
                                               {cfg.defaultsVar, value},
                                               {cfg.stdpar.first, fmt::vformat(cfg.stdpar.second, mkArgStore(varsWithLabel))},
                                               {"input", file},
                                               {"libm", libmFlag}});
    const auto unevalStore = mkArgStore(augmented ^ append(std::pair{std::string("output"), std::string("<unevaluated>")}));
    const auto runsCmd = tc.runs ^ mk_string("", [&](auto &r) { return fmt::vformat(r.command, unevalStore); });
    const auto varsKey = varsWithLabel ^ mk_string("|", [](auto &k, auto &v) { return k + "=" + v; });
    const auto runsHash = std::hash<std::string>{}(runsCmd + "\0" + varsKey);
    const auto pidTag = std::to_string(static_cast<long long>(llvm::sys::Process::getProcessId()));
    const auto baseOutput = fmt::format("{}{}_{}_{:08x}", cfg.outputPrefix, shortName.empty() ? "anon" : shortName, pidTag,
                                        static_cast<std::uint32_t>(runsHash));
    const auto arch = varsWithLabel ^ collect_first([&](auto &k, auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; }) ^
                      get_or_else(std::string{});
    const bool physical = (arch ^ starts_with("cuda")) || (arch ^ starts_with("hsa")) || (arch ^ starts_with("hip"));
    const auto detBase =
        fmt::format("{}{}_{:08x}", cfg.outputPrefix, shortName.empty() ? "anon" : shortName, static_cast<std::uint32_t>(runsHash));
    return modes ^ map([&](const auto mode) -> Task {
             std::vector<RunVariant> variants;
             if (mode == Mode::Offload && physical)
               variants = {RunVariant{"", {}},                                                   // compiletime (default)
                           RunVariant{"runtime", {{std::string(env::PolyrtMirror), "runtime"}}}, // generic SMA walk
                           RunVariant{"usm", {{std::string(env::PolyrtMirror), "off"}}}};        // USM; 77 if no USM
             const auto output = fmt::format("{}_{}", variants.empty() ? baseOutput : detBase, modeName(mode));
             const auto store = mkArgStore(augmented ^ append(std::pair{std::string("output"), output}));
             auto runs = tc.runs ^ map([&](auto &r) { return TestCase::Run{fmt::vformat(r.command, store), r.expect}; });
             return {mode, file, tc.name, varsWithLabel, output, std::move(runs), std::move(variants)};
           });
  };

  const auto all = cfg.testFiles ^ flat_map([&](const std::string &file) {
                     std::ifstream src(file, std::ios::in | std::ios::binary);
                     const auto cases = TestCase::parseTestCase(src, cfg.directive, {{cfg.archVar, targets}});
                     const auto shortName = extractTestName(file);
                     return cases ^ flat_map([&](auto &tc) -> std::vector<Task> {
                              if (!matches(shortName, tc.name)) return {};
                              return tc.matrices ^ flat_map([&](auto &vars) {
                                       return cfg.defaultsVariants ^ flat_map([&](auto &label, auto &value) {
                                                return tasksFor(file, shortName, tc, vars, label, value);
                                              });
                                     });
                            });
                   });

  return all ^ distinct_by([&](const Task &t) -> std::string {
           if (t.mode != Mode::Passthrough) return t.id();
           return fmt::format("P|{}|{}|{}", t.testFile, t.caseName, t.variables ^ mk_string("|", [&](auto &k, auto &v) {
                                                                      return k + "=" + (k == cfg.archVar ? v.substr(0, v.find('@')) : v);
                                                                    }));
         });
}

struct RunnerOptions {
  std::vector<std::string> caseFilters = {};
  std::string runTask = {};
  std::string compileTask = {};
  std::string runOnlyTask = {};
  std::string cleanupTask = {};
  int compileJobs = -1;
  bool verbose = false;
  bool offload = true;
  bool passthrough = true;
  bool list = false;
  bool listIds = false;
  std::string emitFile = {};
  std::string emitPrefix = {};
  std::string emitBinary = {};
  std::string emitWorkdir = {};
  std::string emitDistFile = {};
  std::string emitDistBinary = {};
  std::string emitDistEnv = {};
};

inline void emitCtest(const std::vector<Task> &tasks, const std::string &file, const std::string &prefix, const std::string &binary,
                      const std::string &workdir, const std::string &env) {
  const auto entries = tasks ^ map([](const Task &t) {
                         const auto vs =
                             t.variants ^ map([](const RunVariant &v) {
                               return std::pair{v.suffix, v.env ^ mk_string(";", [](auto &k, auto &val) { return k + "=" + val; })};
                             });
                         return CtestEntry{t.id(), std::string{}, vs};
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

inline int runTasks(const DriverConfig &cfg, const RunnerOptions &opts) {
#ifndef _WIN32
  // XXX ROCr writes a ~150 MB gpucore.<pid> per GPU VM fault (honours RLIMIT_CORE)
  const rlimit noCore{0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
#endif
  auto allTasks = enumerateTasks(cfg, opts.offload, opts.passthrough, opts.caseFilters);
  // single-task compile / run / cleanup, driven by ctest fixtures for shared-compile variant tasks
  if (const auto &single = !opts.compileTask.empty()   ? opts.compileTask
                           : !opts.runOnlyTask.empty() ? opts.runOnlyTask
                                                       : opts.cleanupTask;
      !single.empty()) {
    auto it = allTasks ^ find([&](auto &t) { return t.id() == single; });
    if (!it) return std::fprintf(stderr, "polytest: no task with id '%s'\n", single.c_str()), 1;
    const Task &t = *it;
    if (!opts.cleanupTask.empty()) {
      llvm::sys::fs::remove(t.output);
      for (auto ext : {".exe", ".pdb", ".ilk", ".lib", ".exp"})
        llvm::sys::fs::remove(t.output + ext);
      return 0;
    }
    const bool isCompile = !opts.compileTask.empty();
    const auto o = isCompile ? detail::compileTask(t, cfg) : detail::runTask(t, cfg);
    if (o.status == TaskStatus::Fail) {
      if (!o.failReason.empty()) std::fprintf(stderr, "  reason: %s\n", o.failReason.c_str());
      if (!o.cmdline.empty()) std::fprintf(stderr, "  cmd: %s\n", o.cmdline.c_str());
      std::fprintf(stderr, "  stdout:\n%s\n  stderr:\n%s\n", o.stdoutCapture.c_str(), o.stderrCapture.c_str());
    }
    if (o.status == TaskStatus::Skip) {
      if (isCompile) return 77; // compile-skip cascades to dependents via the FIXTURES_SETUP SKIP_RETURN_CODE
      recordSkip(t.id(), o.failReason);
      return 0;
    }
    return o.status == TaskStatus::Pass ? 0 : 1;
  }
  if (!opts.runTask.empty()) {
    auto it = allTasks ^ find([&](auto &t) { return t.id() == opts.runTask; });
    if (!it) {
      std::fprintf(stderr, "polytest: no task with id '%s'\n", opts.runTask.c_str());
      return 1;
    }
    Task only = std::move(const_cast<Task &>(*it));
    allTasks.clear();
    allTasks.push_back(std::move(only));
  } else std::remove("polytest-skips.log");
  const auto &tasks = allTasks;
  if (tasks.empty()) {
    std::fprintf(stderr, "polytest: no tasks discovered (filters=%zu)\n", opts.caseFilters.size());
    return 0;
  }
  if (opts.listIds) {
    const auto ids = tasks ^ map([](auto &t) { return t.id(); });
    ids | for_each([](auto &id) { std::fprintf(stdout, "%s\n", id.c_str()); });
    // XXX duplicate ids shadow tasks: ctest registers both names but --run-task resolves the first
    const auto dups = ids ^ group_by([](auto &id) { return id; }) ^ filter([](auto &, auto &xs) { return xs.size() > 1; }) ^ keys();
    dups | for_each([](auto &id) { std::fprintf(stderr, "polytest: duplicate task id '%s'\n", id.c_str()); });
    return dups.empty() ? 0 : 1;
  }
  if (!opts.emitFile.empty()) {
    emitCtest(tasks, opts.emitFile, opts.emitPrefix, opts.emitBinary, opts.emitWorkdir, {});
    if (!opts.emitDistFile.empty()) emitCtest(tasks, opts.emitDistFile, opts.emitPrefix, opts.emitDistBinary, {}, opts.emitDistEnv);
    return 0;
  }
  if (opts.list) {
    tasks | for_each([](auto &t) { std::fprintf(stdout, "%s\n", t.display().c_str()); });
    return 0;
  }

  const int jobs = opts.compileJobs > 0 ? opts.compileJobs : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  std::fprintf(stderr, "polytest: %zu tasks; compile -j%d, run -j1\n", tasks.size(), jobs);

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
  auto worker = [&]() {
    for (;;) {
      auto i = nextIdx.fetch_add(1);
      if (i >= tasks.size()) return;
      auto &o = (compileOutcomes[i] = detail::compileTask(tasks[i], cfg));
      auto done = ++compileDone;
      std::lock_guard lk(logMtx);
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
  for (int t = 0; t < jobs; ++t)
    pool.emplace_back(worker);
  pool | for_each([](auto &t) { t.join(); });
  const auto compileSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - compileStart).count();

  std::vector<TaskOutcome> finalOutcomes(tasks.size());
  const auto runStart = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto &co = compileOutcomes[i];
    if (co.status != TaskStatus::Pass) {
      finalOutcomes[i] = co;
      if (co.status == TaskStatus::Skip) recordSkip(tasks[i].id(), co.failReason);
      const char *label = co.status == TaskStatus::Skip ? "compile-skip" : "compile-fail";
      std::fprintf(stderr, "[run %zu/%zu] %s %s (%s)\n", i + 1, tasks.size(), statusTag(co.status), tasks[i].display().c_str(), label);
      continue;
    }
    auto out = detail::runTask(tasks[i], cfg);
    finalOutcomes[i] = out;
    if (out.status == TaskStatus::Skip) recordSkip(tasks[i].id(), out.failReason);
    std::fprintf(stderr, "[run %zu/%zu] %s %s (%.2fs)\n", i + 1, tasks.size(), statusTag(out.status), tasks[i].display().c_str(), out.secs);
    dumpDetails(out, /*includeStdout*/ true);
    if (cfg.cleanupOnSuccess && (out.status == TaskStatus::Pass || out.status == TaskStatus::Skip)) {
      llvm::sys::fs::remove(tasks[i].output);
      // XXX Windows debug builds emit <output>.{exe,pdb,ilk,lib,exp} next to the bare name.
      for (auto ext : {".exe", ".pdb", ".ilk", ".lib", ".exp"})
        llvm::sys::fs::remove(tasks[i].output + ext);
    }
  }
  const auto runSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - runStart).count();

  const auto countBy = [&](TaskStatus s) { return finalOutcomes | aspartame::count([s](auto &o) { return o.status == s; }); };
  const auto pass = countBy(TaskStatus::Pass);
  const auto skip = countBy(TaskStatus::Skip);
  const auto fail = countBy(TaskStatus::Fail);
  if (fail > 0) {
    const auto failedDisplay = tasks                                                                  //
                               | zip(finalOutcomes)                                                   //
                               | filter([](auto &, auto &o) { return o.status == TaskStatus::Fail; }) //
                               | mk_string("\n", [](auto &t, auto &) { return "  - " + t.display(); });
    std::fprintf(stderr, "polytest: failures (%zu):\n%s\n", fail, failedDisplay.c_str());
  }
  std::fprintf(stderr, "polytest: pass=%zu skip=%zu fail=%zu total=%zu compile=%.1fs run=%.1fs\n", pass, skip, fail, tasks.size(),
               compileSecs, runSecs);
  return fail == 0 ? 0 : 1;
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
    int jobs = fire::arg({"-j", "--jobs", "Compile parallelism (default: hardware_concurrency)"}, -1),
    bool verbose = fire::arg({"-v", "--verbose", "Dump stdout/stderr for every task"}),
    bool offloadOnly = fire::arg({"--offload-only", "Skip passthrough mode"}),
    bool passthroughOnly = fire::arg({"--passthrough-only", "Skip offload mode"}),
    bool list = fire::arg({"-l", "--list", "Enumerate tasks (human-readable) and exit"}),
    bool listIds = fire::arg({"--list-ids", "Enumerate task ids (one per line) and exit"}),
    fire::optional<std::string> emitCtest = fire::arg({"--emit-ctest", "Write a CTestTestfile fragment to this path and exit"}),
    fire::optional<std::string> emitPrefix = fire::arg({"--emit-prefix", "ctest test-name prefix (with --emit-ctest)"}),
    fire::optional<std::string> emitBinary = fire::arg({"--emit-binary", "Binary ctest invokes (with --emit-ctest)"}),
    fire::optional<std::string> emitWorkdir = fire::arg({"--emit-workdir", "WORKING_DIRECTORY for emitted tests"}),
    fire::optional<std::string> emitDistFile = fire::arg({"--emit-dist-ctest", "Also write a dist CTestTestfile fragment here"}),
    fire::optional<std::string> emitDistBinary = fire::arg({"--emit-dist-binary", "Binary for the dist fragment"}),
    fire::optional<std::string> emitDistSubdir = fire::arg({"--emit-dist-subdir", "Dist test-files subdir; builds the dist ENVIRONMENT"})) {
  RunnerOptions opts{
      .compileJobs = jobs, .verbose = verbose, .offload = !passthroughOnly, .passthrough = !offloadOnly, .list = list, .listIds = listIds};
  if (caseFilter) opts.caseFilters.push_back(caseFilter.value());
  if (runTask) opts.runTask = runTask.value();
  if (compileTask) opts.compileTask = compileTask.value();
  if (runOnlyTask) opts.runOnlyTask = runOnlyTask.value();
  if (cleanupTask) opts.cleanupTask = cleanupTask.value();
  if (emitCtest) opts.emitFile = emitCtest.value();
  if (emitPrefix) opts.emitPrefix = emitPrefix.value();
  if (emitBinary) opts.emitBinary = emitBinary.value();
  if (emitWorkdir) opts.emitWorkdir = emitWorkdir.value();
  if (emitDistFile) opts.emitDistFile = emitDistFile.value();
  if (emitDistBinary) opts.emitDistBinary = emitDistBinary.value();
  if (emitDistSubdir) opts.emitDistEnv = distEnv(emitDistSubdir.value());
  return runTasks(*detail::firedCfg, opts);
}

inline int runMain(int argc, const char **argv, const DriverConfig &cfg) {
  auto chdirTo = [](const char *what, llvm::StringRef dir) {
    if (auto ec = llvm::sys::fs::create_directories(dir); ec) {
      std::fprintf(stderr, "polytest: failed to create %s='%s': %s\n", what, dir.str().c_str(), ec.message().c_str());
      return;
    }
    if (auto ec = llvm::sys::fs::set_current_path(dir); ec)
      std::fprintf(stderr, "polytest: failed to chdir to %s='%s': %s\n", what, dir.str().c_str(), ec.message().c_str());
  };
  if (const auto override = std::getenv(polyregion::env::PolytestWorkDir); override && *override)
    chdirTo(polyregion::env::PolytestWorkDir, override);
  else if (!cfg.workDir.empty()) chdirTo("workDir", cfg.workDir);
  detail::firedCfg = &cfg;
  constexpr const char *descr = "polytest runner: parallel compile, serial run";
  PREPARE_FIRE_(argc, argv, false, fired_main, descr);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, descr));
  return FIRE_EXTRACT_1_PAD_(fired_main, descr)();
}

} // namespace polyregion::polytest
