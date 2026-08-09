#pragma once

#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"

namespace polyregion::polytest {

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
  bool offloadOnly = false;
  std::optional<std::string> compileFailure;
  std::vector<Run> runs;

  static std::vector<TestCase> parseTestCase(std::ifstream &file,          //
                                             const std::string &directive, //
                                             const std::vector<Variable> &extraMatrices = {}) {
    TestCase testCase;

    auto parseNormalised = [&]<typename F>(std::ifstream &s, F f) {
      auto pos = s.tellg();
      std::vector<typename std::invoke_result_t<F, std::string &>::value_type> xs;
      for (const auto &[rawLine, nextPos] : istream_lines_with_position(s)) {
        auto line = rawLine ^ trim();
        if (!(line ^ starts_with(directive))) continue;
        line = line ^ replace_all(directive, "");
        if (auto t = f(line)) {
          xs.emplace_back(*t);
          pos = nextPos;
        } else break;
      }
      s.seekg(pos); // backtrack on failure
      return xs;
    };

    auto parseRight = [](const std::string &prefix, const std::string &line) -> std::optional<std::string> {
      if (const auto pair = line ^ split_once(prefix)) return pair->second;
      return std::nullopt;
    };

    auto parseExpects = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<Run::Expect> {
        return parseRight("requires", line) ^ map([](const auto &expect) {
                 const auto delimIdx = expect.find(':', 0);
                 const auto [location, message] = expect ^ split_at(delimIdx);
                 auto lineNum = location ^ starts_with("@") ? std::optional{std::stoi(location ^ drop(1))} : std::nullopt;
                 return Run::Expect{lineNum, message ^ drop(1) ^ trim()};
               });
      });
    };

    auto parseRuns = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<Run> {
        return parseRight("do:", line) ^ map([&](const auto &runLine) { return Run{runLine ^ trim(), parseExpects()}; });
      });
    };

    auto parseFlag = [&](const std::string &name) {
      return !parseNormalised(file, [&](const std::string &line) -> std::optional<bool> {
                return (line ^ trim()) == name ? std::optional{true} : std::nullopt;
              }).empty();
    };

    auto parseValue = [&](const std::string &name) -> std::optional<std::string> {
      auto values = parseNormalised(file, [&](const std::string &line) -> std::optional<std::string> {
        return parseRight(name, line) ^ map([](const auto &value) { return value ^ trim(); });
      });
      return values.empty() ? std::nullopt : std::optional{values.front()};
    };

    auto parseMatrices = [&]() {
      return parseNormalised(file, [&](const std::string &line) -> std::optional<std::vector<Variable>> {
        return parseRight("using:", line) ^ map([](const auto &matrixLine) {
                 return matrixLine ^ trim() ^ split(' ') ^ map([](const auto &v) {
                          auto [name, values] = (v ^ split_once('=')).value_or(std::pair{v, std::string{}});
                          return std::pair{std::move(name), values ^ split(',')};
                        });
               });
      });
    };

    return parseNormalised(file, [&](std::string &line) -> std::optional<TestCase> {
      return parseRight("case:", line) ^ map([&](const auto &c) {
               return TestCase{.name = c ^ trim(),
                               .matrices = (parseMatrices()         //
                                            | flatten()             //
                                            | concat(extraMatrices) //
                                            | map([](const auto &name, const auto &values) {
                                                return values ^ map([&](const auto &v) { return std::pair{name, v}; });
                                              })           //
                                            | to_vector()) //
                                           ^ cartesian_product(),
                               .offloadOnly = parseFlag("offload-only"),
                               .compileFailure = parseValue("compile-fails:"),
                               .runs = parseRuns()};
             });
    });
  }
};

} // namespace polyregion::polytest
