#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
#include "polyregion/io.hpp"

#include "fire.hpp"
#include "polytest/lit.hpp"
#include "polytest/profile.hpp"

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

struct Task {
  Mode mode;
  std::string testFile;
  std::string caseName;
  std::vector<std::pair<std::string, std::string>> variables;
  std::string output;
  std::vector<TestCase::Run> runs; // templates already expanded against (variables + defaults + stdpar + input + output)

  std::string display() const {
    return fmt::format("[{}] {}/{} {}", modeName(mode), extractTestName(testFile), caseName,
                       variables | mk_string(" ", [](auto &k, auto &v) { return k + "=" + v; }));
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
  return (t.variables                                                                                            //
          | collect_first([&](auto &k, auto &v) { return k == cfg.archVar ? std::optional{v} : std::nullopt; })) //
         ^ get_or_else(std::string{});
}

inline std::vector<std::string> baseEnvs(const Task &t, const DriverConfig &cfg) {
  std::vector<std::string> envs;
  if (t.mode == Mode::Passthrough) envs ^= concat_inplace(cfg.passthroughEnvs);
  envs.emplace_back(fmt::format("{}={}", cfg.driverEnvVar, cfg.driverPath));
  envs.emplace_back(fmt::format("POLYRT_PLATFORM={}", archFor(t, cfg)));
  envs.emplace_back("POLYRT_HOST_FALLBACK=0");
  envs.emplace_back(fmt::format("{}=1", polyregion::invoke::DeviceLockEnv));
  envs.emplace_back("ASAN_OPTIONS=alloc_dealloc_mismatch=0,detect_leaks=0");
  if (std::getenv("POLYTEST_DEBUG")) envs.emplace_back("POLYRT_DEBUG=2");
  if (auto p = std::getenv("PATH")) envs.emplace_back(std::string("PATH=") + p);
#if defined(__APPLE__)
  // XXX forward DYLD_*; compiled test binaries can't find dist libs otherwise (SIGTRAP).
  for (const auto *name : {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "TMPDIR", "HOME"}) {
    if (auto v = std::getenv(name)) envs.emplace_back(std::string(name) + "=" + v);
  }
#elif defined(__linux__)
  // XXX forward LD_LIBRARY_PATH for dist-relative .so when rpath misses.
  for (const auto *name : {"LD_LIBRARY_PATH", "LD_PRELOAD", "TMPDIR", "HOME"}) {
    if (auto v = std::getenv(name)) envs.emplace_back(std::string(name) + "=" + v);
  }
#elif defined(_WIN32)
  // XXX CreateProcess needs SYSTEMROOT; clang needs TEMP/TMP; link.exe needs LIB/INCLUDE for
  // CRT and SDK lookup (flang does not auto-populate -libpath). Inherit a vcvars-style env.
  for (const auto *name : {"SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE", "WINDIR", "PATHEXT", "COMSPEC", "ProgramFiles",
                           "ProgramFiles(x86)", "ProgramData", "LOCALAPPDATA", "APPDATA", "LIB", "INCLUDE", "LIBPATH"}) {
    if (auto v = std::getenv(name)) envs.emplace_back(std::string(name) + "=" + v);
  }
#endif
  return envs;
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
  // XXX BinaryDir is baked at build time. The CI unit-tests job downloads only the dist
  // artefact (no build tree), so the BinaryDir path doesn't exist there. Fall back to a
  // PATH lookup so the dist's bin/ (set in workflow PATH) resolves.
  if (auto p = llvm::sys::findProgramByName(name)) return *p;
  return name;
}

struct StepResult {
  int exitCode = 0;
  std::string stdoutText;
  std::string stderrText;
  std::string cmdline;
};

inline StepResult runStep(const Task &task, const DriverConfig &cfg, const std::string &command) {
  auto fragments = command ^ split(' ');
  auto [envBits, args] = fragments ^ span([](auto &x) { return x.find('=') != std::string::npos; });

  auto envs = baseEnvs(task, cfg);
  envs ^= concat_inplace(envBits);

  const std::vector<llvm::StringRef> envRefs = envs | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector();
  const std::vector<llvm::StringRef> argRefs = args | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector();

  // XXX TempFile holds an exclusive Windows handle, blocking ExecuteAndWait's child from
  // opening it for redirect; use path-only createUniqueFile + FileRemover.
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
  auto code = llvm::sys::ExecuteAndWait(resolved, argRefs, envRefs, {std::nullopt, *outPath, *errPath});

  auto stdoutText = polyregion::read_string(*outPath);
  auto stderrText = polyregion::read_string(*errPath);

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
  auto r = runStep(task, cfg, task.runs[0].command);
  const auto exit = r.exitCode;
  TaskOutcome o;
  absorbStep(o, std::move(r));
  o.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (exit == 77) o.failReason = "polyrt: no compatible target during compile (exit 77)", o.status = TaskStatus::Skip;
  else if (exit != 0) o.failReason = fmt::format("compile failed (exit {})", exit), o.status = TaskStatus::Fail;
  return o;
}

inline TaskOutcome runTask(const Task &task, const DriverConfig &cfg) {
  TaskOutcome o;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 1; i < task.runs.size(); ++i) {
    auto r = runStep(task, cfg, task.runs[i].command);
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
    for (auto &[k, v] : xs)
      s.push_back(fmt::arg(k.c_str(), v));
    return s;
  };
  const auto matches = [&](const std::string &shortName, const std::string &caseName) {
    return caseFilters.empty() || (caseFilters | exists([&](auto &f) { return f == shortName || f == caseName; }));
  };
  std::vector<Mode> modes;
  if (offload) modes.emplace_back(Mode::Offload);
  if (passthrough) modes.emplace_back(Mode::Passthrough);
  const auto targets = loadTestTargets(cfg.profileDir);

  std::vector<Task> tasks;
  for (const auto &file : cfg.testFiles) {
    std::ifstream src(file, std::ios::in | std::ios::binary);
    const auto cases = TestCase::parseTestCase(src, cfg.directive, {{cfg.archVar, targets}});
    const auto shortName = extractTestName(file);
    for (const auto &tc : cases) {
      if (!matches(shortName, tc.name)) continue;
      for (const auto &vars : tc.matrices) {
        for (const auto &[label, value] : cfg.defaultsVariants) {
          const auto varsWithLabel = vars | append(std::pair{cfg.defaultsLabelVar, label}) | to_vector();
          const auto augmented = varsWithLabel                                                                                     //
                                 | append(std::pair{cfg.defaultsVar, value})                                                       //
                                 | append(std::pair{cfg.stdpar.first, fmt::vformat(cfg.stdpar.second, mkArgStore(varsWithLabel))}) //
                                 | append(std::pair{std::string("input"), file})                                                   //
                                 | to_vector();
          const auto unevalStore =
              mkArgStore(augmented | append(std::pair{std::string("output"), std::string("<unevaluated>")}) | to_vector());
          const auto runsCmd = tc.runs | mk_string("", [&](auto &r) { return fmt::vformat(r.command, unevalStore); });
          const auto varsKey = varsWithLabel | mk_string("|", [](auto &k, auto &v) { return k + "=" + v; });
          const auto runsHash = std::hash<std::string>{}(runsCmd + "\0" + varsKey);
          const auto pidTag = std::to_string(static_cast<long long>(llvm::sys::Process::getProcessId()));
          const auto baseOutput = fmt::format("{}{}_{}_{:08x}", cfg.outputPrefix, shortName.empty() ? "anon" : shortName, pidTag,
                                              static_cast<std::uint32_t>(runsHash));
          for (const auto mode : modes) {
            const auto output = fmt::format("{}_{}", baseOutput, modeName(mode));
            const auto store = mkArgStore(augmented | append(std::pair{std::string("output"), output}) | to_vector());
            auto resolvedRuns =
                tc.runs | map([&](auto &r) { return TestCase::Run{fmt::vformat(r.command, store), r.expect}; }) | to_vector();
            tasks.push_back({mode, file, tc.name, varsWithLabel, output, std::move(resolvedRuns)});
          }
        }
      }
    }
  }
  return tasks;
}

struct RunnerOptions {
  std::vector<std::string> caseFilters;
  int compileJobs = -1;
  bool verbose = false;
  bool offload = true;
  bool passthrough = true;
  bool list = false;
};

inline const char *statusTag(TaskStatus s) {
  switch (s) {
    case TaskStatus::Pass: return "PASS";
    case TaskStatus::Skip: return "SKIP";
    case TaskStatus::Fail: return "FAIL";
  }
  return "?";
}

inline int runTasks(const DriverConfig &cfg, const RunnerOptions &opts) {
  const auto tasks = enumerateTasks(cfg, opts.offload, opts.passthrough, opts.caseFilters);
  if (tasks.empty()) {
    std::fprintf(stderr, "polytest: no tasks discovered (filters=%zu)\n", opts.caseFilters.size());
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
      const char *label = co.status == TaskStatus::Skip ? "compile-skip" : "compile-fail";
      std::fprintf(stderr, "[run %zu/%zu] %s %s (%s)\n", i + 1, tasks.size(), statusTag(co.status), tasks[i].display().c_str(), label);
      continue;
    }
    auto out = detail::runTask(tasks[i], cfg);
    finalOutcomes[i] = out;
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
    int jobs = fire::arg({"-j", "--jobs", "Compile parallelism (default: hardware_concurrency)"}, -1),
    bool verbose = fire::arg({"-v", "--verbose", "Dump stdout/stderr for every task"}),
    bool offloadOnly = fire::arg({"--offload-only", "Skip passthrough mode"}),
    bool passthroughOnly = fire::arg({"--passthrough-only", "Skip offload mode"}),
    bool list = fire::arg({"-l", "--list", "Enumerate tasks and exit without running"})) {
  RunnerOptions opts{.compileJobs = jobs, .verbose = verbose, .offload = !passthroughOnly, .passthrough = !offloadOnly, .list = list};
  if (caseFilter) opts.caseFilters.push_back(caseFilter.value());
  return runTasks(*detail::firedCfg, opts);
}

inline int runMain(int argc, const char **argv, const DriverConfig &cfg) {
  if (const auto override = std::getenv("POLYTEST_WORK_DIR"); override && *override) {
    std::error_code ec;
    std::filesystem::create_directories(override, ec);
    std::filesystem::current_path(override, ec);
    if (ec) std::fprintf(stderr, "polytest: failed to chdir to POLYTEST_WORK_DIR='%s': %s\n", override, ec.message().c_str());
  } else if (!cfg.workDir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(cfg.workDir, ec);
    std::filesystem::current_path(cfg.workDir, ec);
    if (ec) std::fprintf(stderr, "polytest: failed to chdir to workDir='%s': %s\n", cfg.workDir.c_str(), ec.message().c_str());
  }
  detail::firedCfg = &cfg;
  constexpr const char *descr = "polytest runner: parallel compile, serial run";
  PREPARE_FIRE_(argc, argv, false, fired_main, descr);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, descr));
  return FIRE_EXTRACT_1_PAD_(fired_main, descr)();
}

} // namespace polyregion::polytest
