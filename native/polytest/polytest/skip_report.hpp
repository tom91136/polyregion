#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"

namespace polyregion::polytest {

using namespace aspartame;

inline constexpr std::string_view SkipPrefix = "[SKIP] ";

inline void reportSkip(const std::string &id, const std::string &reason) {
  fmt::print(stderr, "{}{}{}{}\n", SkipPrefix, id, reason.empty() ? "" : ": ", reason);
}

struct SkippedTest {
  std::string name, reason;
  bool recorded = false;
};

struct SkipSummary {
  std::string text;
  size_t skipped = 0, unexplained = 0;
};

namespace detail {

inline std::string xmlUnescape(std::string_view s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '&') {
      out += s[i];
      continue;
    }
    const auto semi = s.find(';', i);
    if (semi == std::string_view::npos) {
      out += s[i];
      continue;
    }
    const auto entity = s.substr(i + 1, semi - i - 1);
    if (entity == "amp") out += '&';
    else if (entity == "lt") out += '<';
    else if (entity == "gt") out += '>';
    else if (entity == "quot") out += '"';
    else if (entity == "apos") out += '\'';
    else if (entity.size() > 1 && entity[0] == '#') {
      const bool hex = entity[1] == 'x';
      const auto code = std::strtoul(std::string(entity.substr(hex ? 2 : 1)).c_str(), nullptr, hex ? 16 : 10);
      out += code < 0x80 ? static_cast<char>(code) : '?';
    } else {
      out += s[i];
      continue;
    }
    i = semi;
  }
  return out;
}

inline std::string reasonOf(std::string_view line) {
  const auto body = line.substr(SkipPrefix.size());
  const auto sep = body.find(": ");
  return sep == std::string_view::npos ? std::string{} : std::string(body.substr(sep + 2));
}

} // namespace detail

// ctest writes the output of a skipped test to system-out; the reason is the last `[SKIP] <id>: <reason>` line in it
inline std::vector<SkippedTest> skippedTests(const std::string &junit) {
  std::vector<SkippedTest> out;
  for (size_t pos = junit.find("<testcase "); pos != std::string::npos;) {
    const auto next = junit.find("<testcase ", pos + 1);
    const std::string_view section = std::string_view(junit).substr(pos, next == std::string::npos ? std::string::npos : next - pos);
    pos = next;
    if (section.find("<skipped") == std::string_view::npos) continue;
    const auto nameAt = section.find("name=\"");
    if (nameAt == std::string_view::npos) continue;
    const auto nameEnd = section.find('"', nameAt + 6);
    SkippedTest test;
    test.name = detail::xmlUnescape(section.substr(nameAt + 6, nameEnd - nameAt - 6));
    const auto outStart = section.find("<system-out>");
    const auto outEnd = section.find("</system-out>");
    if (outStart != std::string_view::npos && outEnd != std::string_view::npos && outEnd > outStart) {
      const auto output = detail::xmlUnescape(section.substr(outStart + 12, outEnd - outStart - 12));
      if (const auto line = (output ^ split('\n')) ^ find_last([](const auto &l) { return l ^ starts_with(SkipPrefix); })) {
        test.reason = detail::reasonOf(*line);
        test.recorded = !test.reason.empty();
      }
    }
    out.push_back(std::move(test));
  }
  return out;
}

inline SkipSummary summariseSkips(const std::vector<SkippedTest> &tests, const size_t sampleNames = 3) {
  SkipSummary summary;
  summary.skipped = tests.size();
  if (tests.empty()) return summary;
  std::map<std::string, std::vector<std::string>> byReason;
  std::vector<std::string> unexplained;
  for (const auto &t : tests)
    (t.recorded ? byReason[t.reason] : unexplained).push_back(t.name);
  summary.unexplained = unexplained.size();

  std::vector<std::pair<std::string, std::vector<std::string>>> groups(byReason.begin(), byReason.end());
  std::stable_sort(groups.begin(), groups.end(), [](const auto &l, const auto &r) { return l.second.size() > r.second.size(); });

  const auto names = [&](const std::vector<std::string> &xs) {
    const auto shown = xs ^ take(sampleNames) ^ mk_string(", ");
    return xs.size() > sampleNames ? fmt::format("{}, ... (+{} more)", shown, xs.size() - sampleNames) : shown;
  };
  summary.text = fmt::format("polytest: {} skipped test(s), {} distinct reason(s)\n", tests.size(), groups.size());
  for (const auto &[reason, xs] : groups)
    summary.text += fmt::format("  {:>6} x {}\n           e.g. {}\n", xs.size(), reason, names(xs));
  if (!unexplained.empty())
    summary.text += fmt::format("polytest: {} skipped test(s) recorded no [SKIP] reason: {}\n", unexplained.size(), names(unexplained));
  return summary;
}

} // namespace polyregion::polytest
