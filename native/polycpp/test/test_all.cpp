#include <fstream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "fmt/args.h"
#include "io.hpp"
#include "test_all.h"
#include "utils.hpp"
#include "llvm/Support/Program.h"

std::string extractTestName(const std::string &path) {
  std::string prefix = "check_";
  size_t prefixPos = path.find(prefix);
  size_t extPos = path.find_last_of('.');
  if (prefixPos != std::string::npos && extPos != std::string::npos) {
    return path.substr(prefixPos + prefix.size(), extPos - (prefixPos + prefix.size()));
  } else
    return "";
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

  static std::vector<TestCase> parseTestCase(std::ifstream &file) {
    TestCase testCase;

    auto parseNormalised = [](std::ifstream &file, auto f) {
      auto pos = file.tellg();
      std::vector<typename std::invoke_result_t<decltype(f), std::string &>::value_type> xs;
      for (std::string line; std::getline(file, line);) {
        polyregion::trimInplace(line);
        if (!line.starts_with("//")) continue;
        polyregion::replaceInPlace(line, "//", "");
        if (auto t = f(line); t) {
          xs.emplace_back(*t);
          pos = file.tellg();
        } else
          break;
      }
      file.seekg(pos); // backtrack on failure
      return xs;
    };

    auto parseRight = [](const std::string &prefix, const std::string &line) -> std::optional<std::string> {
      auto pos = line.find(prefix);
      return pos != std::string::npos ? std::optional{line.substr(pos + prefix.size())} : std::nullopt;
    };

    auto parseExpects = [&]() {
      return parseNormalised(file, [&](std::string &line) -> std::optional<TestCase::Run::Expect> {
        if (auto expect = parseRight("#EXPECT", line); expect) {
          auto delim = expect->find(':', 0);
          auto lineNum = expect->starts_with("@") ? std::optional{std::stoi(expect->substr(1, delim))} : std::nullopt;
          return TestCase::Run::Expect{lineNum, polyregion::trim(expect->substr(delim + 1))};
        } else
          return {};
      });
    };

    auto parseRuns = [&]() {
      return parseNormalised(file, [&](std::string &line) -> std::optional<TestCase::Run> {
        auto runLine = parseRight("#RUN:", line);
        return runLine ? std::optional{Run{polyregion::trimInplace(*runLine), parseExpects()}} : std::nullopt;
      });
    };

    auto parseMatrices = [&]() {
      return parseNormalised(file, [&](std::string &line) -> std::optional<std::vector<TestCase::Variable>> {
        if (auto matrixLine = parseRight("#MATRIX:", line); matrixLine) {
          auto variables = polyregion::split(polyregion::trimInplace(*matrixLine), ' ');
          std::vector<Variable> xs;
          std::transform(variables.begin(), variables.end(), std::back_inserter(xs), [](std::string &v) {
            auto delim = v.find('=', 0);
            auto vs = polyregion::split(v.substr(delim + 1), ',');
            return std::pair{v.substr(0, delim), vs};
          });
          return xs;
        } else
          return {};
      });
    };

    return parseNormalised(file, [&](std::string &line) -> std::optional<TestCase> {
      auto caseLine = parseRight("#CASE:", line);
      auto matrices_ = polyregion::flatten(parseMatrices());
      auto runs_ = parseRuns();

      decltype(TestCase::matrices) xs;
      for (auto [v, ys] : matrices_) {
        std::vector<std::pair<std::string, std::string>> row(ys.size());
        std::transform(ys.begin(), ys.end(), row.begin(), [v_ = v](auto &x) { return std::pair{v_, x}; });
        xs.push_back(row);
      }
      return caseLine ? std::optional{TestCase{polyregion::trimInplace(*caseLine), polyregion::cartesin_product(xs), runs_}} : std::nullopt;
    });
  }
};

void testAll(bool passthrough){

  auto run = [passthrough](TestCase &case_, const std::string &binaryName, const std::function<std::string(std::string &)> &mkCommand) {
    for (size_t i = 0; i < case_.runs.size(); ++i) {
      auto &run = case_.runs[i];
      auto command = mkCommand(run.command);
      DYNAMIC_SECTION("RUN " <<  run.command) {
        auto fragments = polyregion::split(command, ' ');
        auto [envs, args] = polyregion::take_while(fragments, [](auto &x) { return x.find('=') != std::string::npos; });
        args.emplace_back("-fsanitize=address");

        if(passthrough){
          envs.emplace_back("POLYCPP_NO_REWRITE=1");
          envs.emplace_back("POLYSTL_NO_OFFLOAD=1");
        }

        envs.emplace_back("POLYSTL_LIB=" + PolySTLLib);
        envs.emplace_back("POLYSTL_INCLUDE=" + PolySTLInclude);
        envs.emplace_back("POLYC_BIN=" + PolyCBin);


        envs.emplace_back("LD_LIBRARY_PATH=" + PolySTLLDLibraryPath);
        if (auto path = std::getenv("PATH"); path) envs.emplace_back(std::string("PATH=") + path);

        if (args.empty()) throw std::logic_error("Bad command: " + command);
        std::vector<llvm::StringRef> envs_;
        std::transform(envs.begin(), envs.end(), std::back_inserter(envs_), [](auto &x) { return llvm::StringRef(x); });

        std::vector<llvm::StringRef> args_;
        std::transform(args.begin(), args.end(), std::back_inserter(args_), [](auto &x) { return llvm::StringRef(x); });

        auto stdoutFile = llvm::sys::fs::TempFile::create("polycpp_stdout-%%-%%-%%-%%-%%");
        if (auto e = stdoutFile.takeError()) FAIL("Cannot create stdout:" << toString(std::move(e)));
        auto stderrFile = llvm::sys::fs::TempFile::create("polycpp_stderr-%%-%%-%%-%%-%%");
        if (auto e = stderrFile.takeError()) FAIL("Cannot create stderr:" << toString(std::move(e)));

        auto exitCode = llvm::sys::ExecuteAndWait(args[0], args_, envs_, {std::nullopt, stdoutFile->TmpName, stderrFile->TmpName});

        auto stdout_ = polyregion::read_string(stdoutFile->TmpName);
        auto stderr_ = polyregion::read_string(stderrFile->TmpName);
        consumeError(stdoutFile->discard());
        consumeError(stderrFile->discard());

        WARN("exe:  " << args[0] << " (wd=" << BinaryDir << ")");
        WARN("args: " << polyregion::mk_string<llvm::StringRef>(args_, &llvm::StringRef::str, " "));
        WARN("envs: " << polyregion::mk_string<llvm::StringRef>(envs_, &llvm::StringRef::str, " "));
        WARN("stderr:\n" << stderr_ << "[EOF]");
        WARN("stdout:\n" << stdout_ << "[EOF]");
        REQUIRE(exitCode == 0);
        //              CHECK(stderr_.empty());
        auto stdoutLines = polyregion::split(stdout_, '\n');
        for (auto &[line, expected] : run.expect) {
          DYNAMIC_SECTION("EXPECT " << (line ? std::to_string(*line) : "*") << "==" << expected) {
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
          if (llvm::sys::fs::remove(binaryName)) INFO("Removed binary: " << binaryName);
          else
            WARN("Cannot remove binary: " << binaryName);
        }
      }
    }
  };

  for (const auto &test : TestFiles) {
    DYNAMIC_SECTION(extractTestName(test)) {
      std::ifstream s(test, std::ios::in | std::ios::binary);
      auto tc = TestCase::parseTestCase(s);

      for (auto &case_ : tc) {

        DYNAMIC_SECTION("CASE " << case_.name) {

          for (const auto &variables : case_.matrices) {
            auto name = polyregion::mk_string<std::pair<std::string, std::string>>(
                variables, [](auto &p) { return p.first + "=" + p.second; }, " ");
            DYNAMIC_SECTION("MATRIX " <<name  ) {

              auto binaryName = polyregion::hex(polyregion::hash(polyregion::mk_string<TestCase::Run>(
                  case_.runs, [](auto &x) { return x.command; }, "")));
              fmt::dynamic_format_arg_store<fmt::format_context> store;
              store.push_back(fmt::arg("input", test));
              store.push_back(fmt::arg("output", binaryName));
              for (const auto &[k, v] : variables)
                store.push_back(fmt::arg(k.c_str(), v));
              CAPTURE(test);
              run(case_, binaryName, [&](std::string &command) { return fmt::vformat(command, store); });
            }
          }
        }
      }
    }
  }
}


TEST_CASE("offload") {
  testAll(false);
}

TEST_CASE("passthrough") {
  testAll(true);
}