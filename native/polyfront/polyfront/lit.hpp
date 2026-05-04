#pragma once

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <climits>
  #include <unistd.h>
  #ifndef _POSIX_HOST_NAME_MAX
    #define _POSIX_HOST_NAME_MAX 255
  #endif
#endif

#include "aspartame/all.hpp"

namespace polyregion::polyfront {

using namespace aspartame;

inline std::optional<std::string> hostname() {
#ifdef _WIN32
  DWORD n = 0;
  GetComputerNameExA(ComputerNameDnsHostname, nullptr, &n); // probes required size, returns false
  std::string s(n, '\0');
  if (!GetComputerNameExA(ComputerNameDnsHostname, s.data(), &n)) return {};
  s.resize(n);
  return s;
#else
  std::string s(_POSIX_HOST_NAME_MAX + 1, '\0');
  if (gethostname(s.data(), s.size()) != 0) return {};
  s.resize(std::strlen(s.c_str()));
  return s;
#endif
}

// `<backend>@<uarch>` test targets, in order: $POLYREGION_TEST_TARGETS (verbatim, colon-split),
// $POLYREGION_TEST_PROFILE.env, <hostname>.env, default.env, {}. Profile files concatenate every
// `POLYREGION_TEST_TARGETS=...` line; `#` and blank lines are skipped.
inline std::vector<std::string> loadTestTargets(const std::string &profileDir) {
  auto split = [](const std::string &s, std::vector<std::string> &out) {
    for (size_t i = 0, j = 0; i <= s.size(); ++i)
      if (i == s.size() || s[i] == ':') {
        if (i > j) out.emplace_back(s.substr(j, i - j));
        j = i + 1;
      }
  };
  if (auto v = std::getenv("POLYREGION_TEST_TARGETS")) {
    std::vector<std::string> xs;
    split(v, xs);
    return xs;
  }
  auto readKey = [&](const std::filesystem::path &file) -> std::optional<std::vector<std::string>> {
    if (!std::filesystem::exists(file)) return {};
    std::ifstream is(file);
    std::vector<std::string> targets;
    bool any = false;
    for (std::string line; std::getline(is, line);) {
      size_t s = 0;
      while (s < line.size() && (line[s] == ' ' || line[s] == '\t'))
        ++s;
      if (s == line.size() || line[s] == '#') continue;
      // Tolerate the shell-sourcable forms: `export K=v`, `K+=v`, and a leading `:` on appends.
      if (line.compare(s, 7, "export ") == 0) s += 7;
      const auto eq = line.find('=', s);
      if (eq == std::string::npos) continue;
      auto keyEnd = eq;
      if (keyEnd > s && line[keyEnd - 1] == '+') --keyEnd;
      if (line.compare(s, keyEnd - s, "POLYREGION_TEST_TARGETS") != 0) continue;
      any = true;
      auto valStart = eq + 1;
      if (valStart < line.size() && line[valStart] == ':') ++valStart;
      split(line.substr(valStart), targets);
    }
    if (!any) return {};
    return targets;
  };
  if (auto v = std::getenv("POLYREGION_TEST_PROFILE"))
    if (auto t = readKey(std::filesystem::path(profileDir) / (std::string(v) + ".env"))) return *t;
  if (auto h = hostname()) {
    if (const auto dot = h->find('.'); dot != std::string::npos) h->resize(dot);
    if (auto t = readKey(std::filesystem::path(profileDir) / (*h + ".env"))) return *t;
  }
  if (auto t = readKey(std::filesystem::path(profileDir) / "default.env")) return *t;
  return {};
}

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