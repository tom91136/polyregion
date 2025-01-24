#pragma once

#include <fstream>
#include <optional>

#include "aspartame/all.hpp"

namespace polyregion::polyfront {

using namespace aspartame;

inline std::string extractTestName(const std::string &path, const std::string &prefix = "check_") {
  const size_t prefixPos = path.find(prefix);
  const size_t extPos = path.find_last_of('.');
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
} // namespace polyregion::polyfront