
#include <fstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_range.hpp"

#include "fmt/args.h"
#include "polyfront/lit.hpp"
#include "polyregion/io.hpp"
#include "test_all.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"

using namespace aspartame;

void testAll(bool passthrough) {

  auto run = [passthrough](polyregion::polyfront::TestCase &case_, const std::string &input,
                           const std::vector<std::pair<std::string, std::string>> &variables) {
    const auto mkArgStore = [](auto &&xs) {
      fmt::dynamic_format_arg_store<fmt::format_context> s;
      for (auto &&[k, v] : xs)
        s.push_back(fmt::arg(k.c_str(), v));
      return s;
    };

    const auto userArgs = variables ^ map([&](auto &k, auto &v) { return std::pair{k, v}; });
    const auto augmentedArgs =
        userArgs                                                                                          //
        | append(std::pair{"input", input})                                                               //
        | append(std::pair{"polycpp_defaults", "-fno-crash-diagnostics -O1 -g3 -Wall -Wextra -pedantic"}) //
        | append(std::pair{
              "polycpp_stdpar",
              fmt::vformat("-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=dynamic -v",
                           variables ^ and_then(mkArgStore))}) //
        | to_vector();

    const auto unevaluatedStore = augmentedArgs ^ append(std::pair{"output", "<unevaluated>"}) ^ and_then(mkArgStore);
    const auto output = fmt::format("polycpp_test_{:x}", std::hash<std::string>()(case_.runs ^ mk_string("", [&](auto &x) {
                                                                                    return fmt::vformat(x.command, unevaluatedStore);
                                                                                  })));
    const auto evaluatedStore = augmentedArgs ^ append(std::pair{"output", output}) ^ and_then(mkArgStore);

    for (size_t i = 0; i < case_.runs.size(); ++i) {
      const auto &[rawCommand, expect] = case_.runs[i];
      const auto command = fmt::vformat(rawCommand, evaluatedStore);
      DYNAMIC_SECTION("do: " << command) {
        auto fragments = command ^ split(' ');
        auto [envs, args] = fragments ^ span([](auto &x) { return x.find('=') != std::string::npos; });

        if (passthrough) {
          envs.emplace_back("POLYCPP_NO_REWRITE=1");
          envs.emplace_back("POLYSTL_NO_OFFLOAD=1");
        }

        envs.emplace_back(fmt::format("POLYCPP_DRIVER={}", ClangDriver));
        envs.emplace_back(fmt::format("POLYRT_PLATFORM={}", fmt::vformat("{polycpp_arch}", evaluatedStore)));
        envs.emplace_back("POLYRT_HOST_FALLBACK=0");

        // if host, + -fsanitize=address,undefined
        envs.emplace_back("ASAN_OPTIONS=alloc_dealloc_mismatch=0,detect_leaks=0");
        envs.emplace_back("POLYRT_DEBUG=2");
        //        envs.emplace_back("LD_PRELOAD=/usr/bin/../lib/clang/18/lib/x86_64-redhat-linux-gnu/libclang_rt.asan.so");

        if (auto path = std::getenv("PATH"); path) envs.emplace_back(std::string("PATH=") + path);

        if (args.empty()) throw std::logic_error("Bad command: " + command);

        std::vector<llvm::StringRef> envs_ = envs ^ map([&](auto &x) { return llvm::StringRef(x); });
        std::vector<llvm::StringRef> args_ = args ^ map([&](auto &x) { return llvm::StringRef(x); });

        auto stdoutFile = llvm::sys::fs::TempFile::create("polycpp_stdout-%%-%%-%%-%%-%%");
        if (auto e = stdoutFile.takeError()) FAIL("Cannot create stdout:" << toString(std::move(e)));
        auto stderrFile = llvm::sys::fs::TempFile::create("polycpp_stderr-%%-%%-%%-%%-%%");
        if (auto e = stderrFile.takeError()) FAIL("Cannot create stderr:" << toString(std::move(e)));

        // ExecuteAndWait does PATH but not cwd; args[0] is either "polycpp" (BinaryDir) or a
        // just-compiled binary in cwd, so resolve both without touching PATH or chdir.
        auto resolveBin = [&](llvm::StringRef name) -> std::string {
          auto inBinaryDir = fmt::format("{}/{}", BinaryDir, std::string(name));
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

        WARN("cmdline: " << (args_ | drop(1) | prepend(fmt::format("{}/{}", BinaryDir, args[0])) //
                             | mk_string(" ", [](auto &s) { return s.str(); })));
        WARN("envs: " << (envs_ ^ mk_string(" ", [](auto &s) { return s.str(); })));
        WARN("stderr:\n" << stderr_ << "[EOF]");
        WARN("stdout:\n" << stdout_ << "[EOF]");
        REQUIRE(exitCode == 0);
        //              CHECK(stderr_.empty());
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
        // Drop the compiled binary so the cwd stays clean across a sweep. Only reached when
        // every REQUIRE in this section passed; failed runs leave the binary for triage.
        if (i == case_.runs.size() - 1) {
          if (auto ec = llvm::sys::fs::remove(output)) WARN("Cannot remove binary " << output << ": " << ec.message());
        }
      }
    }
  };

  for (const auto &test : TestFiles) {
    DYNAMIC_SECTION(polyregion::polyfront::extractTestName(test)) {
      std::ifstream source(test, std::ios::in | std::ios::binary);

      auto cases = polyregion::polyfront::TestCase::parseTestCase(
          source, "#pragma region", {{"polycpp_arch", polyregion::polyfront::loadTestTargets(POLYREGION_TEST_PROFILE_DIR)}});

      if (cases.empty()) FAIL("No test cases found");

      for (auto &testCase : cases) {

        DYNAMIC_SECTION("case: " << testCase.name) {

          // testCase.matrices.emplace_back();

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

TEST_CASE("offload") { testAll(false); }

TEST_CASE("passthrough") { testAll(true); }