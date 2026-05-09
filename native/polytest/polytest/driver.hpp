#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fmt/args.h"
#include "fmt/format.h"

#include "polyregion/io.hpp"

#include "polytest/lit.hpp"
#include "polytest/profile.hpp"

namespace polyregion::polytest {

using namespace aspartame;

struct DriverConfig {
  std::string driverPath;                       // path to clang++ / flang-new
  std::string binaryDir;                        // build dir to resolve test binaries against
  std::vector<std::string> testFiles;           // LIT-style source files to scan
  std::string profileDir;                       // directory holding `<host>.env` test profiles
  std::string archVar;                          // matrix variable, e.g. "polycpp_arch", "polyfc_arch"
  std::pair<std::string, std::string> defaults; // {name, value} format-string pair injected as a variable
  std::pair<std::string, std::string> stdpar;   // {name, template} -- template may reference `{archVar}`
  std::string driverEnvVar;                     // env var to set the wrapper's driver-binary path
  std::vector<std::string> passthroughEnvs;     // envs added when `passthrough` is true
  std::string outputPrefix;                     // generated binary name prefix
  std::string tempPrefix;                       // tempfile name prefix
  std::string directive;                        // line prefix for LIT directives (e.g. "#pragma region")
  bool cleanupOnSuccess;                        // whether to remove generated binaries after passing runs
};

inline void runTestSuite(const DriverConfig &cfg, bool passthrough) {
  auto run = [&](TestCase &case_, const std::string &input, const std::vector<std::pair<std::string, std::string>> &variables) {
    const auto mkArgStore = [](auto &&xs) {
      fmt::dynamic_format_arg_store<fmt::format_context> s;
      for (auto &&[k, v] : xs)
        s.push_back(fmt::arg(k.c_str(), v));
      return s;
    };

    const auto userArgs = variables ^ map([&](auto &k, auto &v) { return std::pair{k, v}; });
    const auto augmentedArgs = userArgs                                                            //
                               | append(std::pair{"input", input})                                 //
                               | append(std::pair{cfg.defaults.first, cfg.defaults.second})        //
                               | append(std::pair{cfg.stdpar.first,                                //
                                                  fmt::vformat(cfg.stdpar.second,                  //
                                                               variables ^ and_then(mkArgStore))}) //
                               | to_vector();

    const bool runtimeDebug = std::getenv("POLYTEST_DEBUG") != nullptr;

    const auto unevaluatedStore = augmentedArgs ^ append(std::pair{"output", "<unevaluated>"}) ^ and_then(mkArgStore);
    const auto testName = extractTestName(input);
    const auto runsHash =
        std::hash<std::string>()(case_.runs ^ mk_string("", [&](auto &x) { return fmt::vformat(x.command, unevaluatedStore); }));
    const auto output = fmt::format("{}{}_{:08x}", cfg.outputPrefix, testName.empty() ? "anon" : testName, static_cast<uint32_t>(runsHash));
    const auto evaluatedStore = augmentedArgs ^ append(std::pair{"output", output}) ^ and_then(mkArgStore);

    // XXX Catch2 reruns this lambda from scratch for each leaf section, so a local flag would reset
    // between the compile and run passes. The static keeps state across re-entries.
    static std::unordered_map<std::string, bool> stepFailureBy;
    for (size_t i = 0; i < case_.runs.size(); ++i) {
      const auto &[rawCommand, expect] = case_.runs[i];
      const auto command = fmt::vformat(rawCommand, evaluatedStore);
      DYNAMIC_SECTION("do: " << command) {
        if (stepFailureBy[output]) {
          SKIP("earlier step in this case failed; not running '" << command << "'");
          return;
        }
        auto fragments = command ^ split(' ');
        auto [envs, args] = fragments ^ span([](auto &x) { return x.find('=') != std::string::npos; });

        if (passthrough) envs ^= concat_inplace(cfg.passthroughEnvs);

        envs.emplace_back(fmt::format("{}={}", cfg.driverEnvVar, cfg.driverPath));
        envs.emplace_back(fmt::format("POLYRT_PLATFORM={}", fmt::vformat("{" + cfg.archVar + "}", evaluatedStore)));
        envs.emplace_back("POLYRT_HOST_FALLBACK=0");

        envs.emplace_back("ASAN_OPTIONS=alloc_dealloc_mismatch=0,detect_leaks=0");
        if (runtimeDebug) envs.emplace_back("POLYRT_DEBUG=2");

        if (auto path = std::getenv("PATH"); path) envs.emplace_back(std::string("PATH=") + path);

        if (args.empty()) throw std::logic_error("Bad command: " + command);

        std::vector<llvm::StringRef> envs_ = envs ^ map([&](auto &x) { return llvm::StringRef(x); });
        std::vector<llvm::StringRef> args_ = args ^ map([&](auto &x) { return llvm::StringRef(x); });

        auto stdoutFile = llvm::sys::fs::TempFile::create(cfg.tempPrefix + "stdout-%%-%%-%%-%%-%%");
        if (auto e = stdoutFile.takeError()) FAIL("Cannot create stdout:" << toString(std::move(e)));
        auto stderrFile = llvm::sys::fs::TempFile::create(cfg.tempPrefix + "stderr-%%-%%-%%-%%-%%");
        if (auto e = stderrFile.takeError()) FAIL("Cannot create stderr:" << toString(std::move(e)));

        // Resolve `<outputPrefix>*` from cwd first so a stale copy in BinaryDir from a previous
        // build can't shadow the freshly-produced one.
        auto resolveBin = [&](llvm::StringRef name) -> std::string {
          const bool isTestBin = name.starts_with(cfg.outputPrefix);
          if (isTestBin && llvm::sys::fs::exists(name)) {
            llvm::SmallString<256> abs(name);
            llvm::sys::fs::make_absolute(abs);
            return std::string(abs);
          }
          auto inBinaryDir = fmt::format("{}/{}", cfg.binaryDir, std::string(name));
          if (llvm::sys::fs::exists(inBinaryDir)) return inBinaryDir;
          if (llvm::sys::fs::exists(name)) {
            llvm::SmallString<256> abs(name);
            llvm::sys::fs::make_absolute(abs);
            return std::string(abs);
          }
          return std::string(name);
        };
        const auto resolved = resolveBin(args[0]);
        auto exitCode = llvm::sys::ExecuteAndWait(resolved, args_, envs_, {std::nullopt, stdoutFile->TmpName, stderrFile->TmpName});

        auto stdout_ = polyregion::read_string(stdoutFile->TmpName);
        auto stderr_ = polyregion::read_string(stderrFile->TmpName);
        consumeError(stdoutFile->discard());
        consumeError(stderrFile->discard());

        // INFO over WARN: only flushes on REQUIRE/CHECK failure, so passing tests stay quiet.
        WARN("cmdline: " << (args_ | drop(1) | prepend(fmt::format("{}/{}", cfg.binaryDir, args[0])) //
                             | mk_string(" ", [](auto &s) { return s.str(); })));
        WARN("envs: " << (envs_ ^ mk_string(" ", [](auto &s) { return s.str(); })));
        WARN("stderr:\n" << stderr_ << "[EOF]");
        WARN("stdout:\n" << stdout_ << "[EOF]");
        // XXX exit 77 is the autotools "skipped" convention; polyrt emits this when no
        // compatible target is available and host fallback is disabled.
        if (exitCode == 77) {
          stepFailureBy[output] = true;
          SKIP("polyrt reports no compatible target for this iteration (exit 77)");
        }
        if (exitCode != 0) stepFailureBy[output] = true;
        REQUIRE(exitCode == 0);
        auto stdoutLines = stdout_ ^ split('\n');
        for (auto &[line, expected] : expect) {
          DYNAMIC_SECTION("requires: " << (line ? std::to_string(*line) : "*") << "==" << expected) {
            if (line) {
              INFO(stdoutLines.size());
              auto idx = *line < 0 ? stdoutLines.size() + *line : *line;
              CHECK(stdoutLines[idx] == expected);
            } else {
              CHECK(stdout_ == expected);
            }
          }
        }
        if (cfg.cleanupOnSuccess && i == case_.runs.size() - 1) {
          if (auto ec = llvm::sys::fs::remove(output)) WARN("Cannot remove binary " << output << ": " << ec.message());
        }
      }
    }
  };

  for (const auto &test : cfg.testFiles) {
    DYNAMIC_SECTION(extractTestName(test)) {
      std::ifstream source(test, std::ios::in | std::ios::binary);

      auto cases = TestCase::parseTestCase(source, cfg.directive, {{cfg.archVar, loadTestTargets(cfg.profileDir)}});

      if (cases.empty()) FAIL("No test cases found");

      for (auto &testCase : cases) {
        DYNAMIC_SECTION("case: " << testCase.name) {
          if (testCase.matrices.empty()) FAIL("Test matrix yielded zero tests");

          for (const auto &variables : testCase.matrices) {
            auto name = variables ^ mk_string(" ", [](auto &k, auto &v) { return k + "=" + v; });
            DYNAMIC_SECTION("using: " << name) {
              CAPTURE(test + ":1"); // XXX glue a fake line number so that the test runner can turn it into a URL
              run(testCase, test, variables);
            }
          }
        }
      }
    }
  }
}

} // namespace polyregion::polytest
