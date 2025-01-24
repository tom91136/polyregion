
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
    const auto augmentedArgs = userArgs                                                                                          //
                               | append(std::pair{"input", input})                                                               //
                               | append(std::pair{"polycpp_defaults", "-fno-crash-diagnostics -O1 -g3 -Wall -Wextra -pedantic"}) //
                               | append(std::pair{"polycpp_stdpar", fmt::vformat("-fstdpar -fstdpar-arch={polycpp_arch} -fuse-ld=lld",
                                                                                 variables ^ and_then(mkArgStore))}) //
                               | to_vector();

    const auto unevaluatedStore = augmentedArgs ^ append(std::pair{"output", "<unevaluated>"}) ^ and_then(mkArgStore);
    const auto output = fmt::format(
        "{:x}", std::hash<std::string>()(case_.runs ^ mk_string("", [&](auto &x) { return fmt::vformat(x.command, unevaluatedStore); })));
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
        envs.emplace_back(fmt::vformat("POLYRT_PLATFORM={polycpp_arch}", evaluatedStore));
        envs.emplace_back("POLYSTL_HOST_FALLBACK=0");

        // if host, + -fsanitize=address,undefined
        envs.emplace_back("ASAN_OPTIONS=alloc_dealloc_mismatch=0,detect_leaks=0");
        //        envs.emplace_back("LD_PRELOAD=/usr/bin/../lib/clang/18/lib/x86_64-redhat-linux-gnu/libclang_rt.asan.so");

        if (auto path = std::getenv("PATH"); path) envs.emplace_back(std::string("PATH=") + path);

        if (args.empty()) throw std::logic_error("Bad command: " + command);

        std::vector<llvm::StringRef> envs_ = envs ^ map([&](auto &x) { return llvm::StringRef(x); });
        std::vector<llvm::StringRef> args_ = args ^ map([&](auto &x) { return llvm::StringRef(x); });

        auto stdoutFile = llvm::sys::fs::TempFile::create("polycpp_stdout-%%-%%-%%-%%-%%");
        if (auto e = stdoutFile.takeError()) FAIL("Cannot create stdout:" << toString(std::move(e)));
        auto stderrFile = llvm::sys::fs::TempFile::create("polycpp_stderr-%%-%%-%%-%%-%%");
        if (auto e = stderrFile.takeError()) FAIL("Cannot create stderr:" << toString(std::move(e)));

        auto exitCode = llvm::sys::ExecuteAndWait(args[0], args_, envs_, {std::nullopt, stdoutFile->TmpName, stderrFile->TmpName});

        auto stdout_ = polyregion::read_string(stdoutFile->TmpName);
        auto stderr_ = polyregion::read_string(stderrFile->TmpName);
        consumeError(stdoutFile->discard());
        consumeError(stderrFile->discard());

        WARN("exe:  " << BinaryDir << "/" << args[0]);
        WARN("args: " << (args_ ^ mk_string(" ", [](auto &s) { return s.str(); })));
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
        if (i == case_.runs.size() - 1) {
          // if (llvm::sys::fs::remove(binaryName)) INFO("Removed binary: " << binaryName);
          // else WARN("Cannot remove binary: " << binaryName);
        }
      }
    }
  };

  for (const auto &test : TestFiles) {
    DYNAMIC_SECTION(polyregion::polyfront::extractTestName(test)) {
      std::ifstream source(test, std::ios::in | std::ios::binary);

      auto cases = polyregion::polyfront::TestCase::parseTestCase(
          source, "#pragma region",
          {
#if defined(__linux__)
              {"polycpp_arch", {"cuda@sm_89", "hip@gfx1036", "hsa@gfx1036", "host@native"}} //
#elif defined(__APPLE__)
              {"polycpp_arch", {"host@apple-m2"}} //
#elif defined(_WIN32)
              {"polycpp_arch", {}} //
#else
  #error "Unsupported platform"
#endif
          });

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