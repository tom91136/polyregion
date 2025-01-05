
#include <fstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_range.hpp"

#include "fmt/args.h"
#include "polyregion/io.hpp"
#include "test_all.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"

using namespace aspartame;

std::string extractTestName(const std::string &path) {
  std::string prefix = "check_";
  size_t prefixPos = path.find(prefix);
  size_t extPos = path.find_last_of('.');
  if (prefixPos != std::string::npos && extPos != std::string::npos) {
    return path.substr(prefixPos + prefix.size(), extPos - (prefixPos + prefix.size()));
  } else return "";
}

struct TestCase {
  struct Run {
    using Expect = std::pair<std::optional<int>, std::string>;
    std::string command;
    std::vector<Expect> expect;
  };
  using Variable = std::pair<std::string, std::vector<std::string>>;
  std::string name;
  std::vector<std::vector<std::pair<std::string, std::string>>> matrices;
  std::vector<Run> runs;

  static std::vector<TestCase> parseTestCase(std::ifstream &file,          //
                                             const std::string &directive, //
                                             const std::vector<Variable> &extraMatrices = {}) {
    TestCase testCase;

    auto parseNormalised = [&]<typename F>(std::ifstream &s, F f) {
      auto pos = s.tellg();
      std::vector<typename std::invoke_result_t<F, std::string &>::value_type> xs;
      for (std::string line; std::getline(s, line);) {
        line = line ^ trim();
        if (!(line ^ starts_with(directive))) continue;
        line = line ^ replace_all(directive, "");
        if (auto t = f(line)) {
          xs.emplace_back(*t);
          pos = s.tellg();
        } else break;
      }
      s.seekg(pos); // backtrack on failure
      return xs;
    };

    auto parseRight = [](const std::string &prefix, const std::string &line) -> std::optional<std::string> {
      const auto pos = line.find(prefix);
      return pos != std::string::npos ? std::optional{line.substr(pos + prefix.size())} : std::nullopt;
    };

    auto parseExpects = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<Run::Expect> {
        return parseRight("requires", line) ^ map([](auto &expect) {
                 const auto delimIdx = expect.find(':', 0);
                 auto lineNum = expect ^ starts_with("@") ? std::optional{std::stoi(expect.substr(1, delimIdx))} : std::nullopt;
                 return Run::Expect{lineNum, expect.substr(delimIdx + 1) ^ trim()};
               });
      });
    };

    auto parseRuns = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<Run> {
        return parseRight("do:", line) ^ map([&](auto &runLine) { return Run{runLine ^ trim(), parseExpects()}; });
      });
    };

    auto parseMatrices = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<std::vector<Variable>> {
        return parseRight("using:", line) ^ map([](auto &matrixLine) {
                 return matrixLine ^ trim() ^ split(' ') ^ map([](auto &v) {
                          const auto delimIdx = v.find('=', 0);
                          const auto vs = v.substr(delimIdx + 1) ^ split(',');
                          return std::pair{v.substr(0, delimIdx), vs};
                        });
               });
      });
    };

    return parseNormalised(file, [&](std::string &line) -> std::optional<TestCase> {
      return parseRight("case:", line) ^ map([&](auto &c) {
               return TestCase{
                   .name = c ^ trim(),
                   .matrices = parseMatrices()                                                                                           //
                               ^ flatten()                                                                                               //
                               ^ concat(extraMatrices)                                                                                   //
                               ^ map([](auto &name, auto &values) { return values ^ map([&](auto &v) { return std::pair{name, v}; }); }) //
                               ^ sequence(),
                   .runs = parseRuns()};
             });
    });
  }
};

void testAll(bool passthrough) {

  auto run = [passthrough](TestCase &case_, const std::string &input, const std::vector<std::pair<std::string, std::string>> &variables) {
    const auto mkArgStore = [](auto &&xs) {
      fmt::dynamic_format_arg_store<fmt::format_context> s;
      for (auto &&[k, v] : xs)
        s.push_back(fmt::arg(k.c_str(), v));
      return s;
    };

    const auto userArgs = variables ^ map([&](auto &k, auto &v) { return std::pair{k, v}; });
    const auto augmentedArgs =
        userArgs                                                                                                                       //
        | append(std::pair{"input", input})                                                                                            //
        | append(std::pair{"polycpp_defaults", "-fno-crash-diagnostics -O1 -g3 -Wall -Wextra -pedantic"})                              //
        | append(std::pair{"polycpp_stdpar", fmt::vformat("-fstdpar -fstdpar-arch={polycpp_arch}", variables ^ and_then(mkArgStore))}) //
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
        envs.emplace_back(fmt::vformat("POLYSTL_PLATFORM={polycpp_arch}", evaluatedStore));
        envs.emplace_back("POLYSTL_HOST_FALLBACK=0");

        // if host, + -fsanitize=address,undefined
        envs.emplace_back("ASAN_OPTIONS=alloc_dealloc_mismatch=0");
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
    DYNAMIC_SECTION(extractTestName(test)) {
      std::ifstream source(test, std::ios::in | std::ios::binary);

      auto cases = TestCase::parseTestCase(source, "#pragma region",
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