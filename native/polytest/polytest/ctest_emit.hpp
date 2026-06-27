#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/env_keys.h"

#ifndef _WIN32
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace polyregion::polytest {

// out-of-band skip log: ctest's SKIP_RETURN_CODE is an O(n^2) name scan at parse, so a skip exits 0
// and appends here instead; shared by both runners
inline void recordSkip(const std::string &id, const std::string &reason) {
  fmt::print(stderr, "[SKIP] {}{}{}\n", id, reason.empty() ? "" : ": ", reason);
#ifndef _WIN32
  const auto line = fmt::format("{}\t{}\n", id, reason);
  if (const int fd = ::open("polytest-skips.log", O_WRONLY | O_CREAT | O_APPEND, 0644); fd >= 0) {
    [[maybe_unused]] const auto n = ::write(fd, line.data(), line.size()); // short line => atomic append
    ::close(fd);
  }
#endif
}

// empty `variants` => a self-contained `--run-task`; non-empty => one FIXTURES_SETUP compile, a
// FIXTURES_REQUIRED run per variant (name suffix + extra ENVIRONMENT), and a FIXTURES_CLEANUP
struct CtestEntry {
  std::string id;
  std::string labels;
  std::vector<std::pair<std::string, std::string>> variants;
};

inline void emitCtestFragment(const std::string &file, const std::string &prefix, const std::string &binary,
                              const std::string & /*workdir*/, const std::string &env, const std::vector<CtestEntry> &tasks) {
  using namespace aspartame;
  // set_tests_properties is an O(n^2) name scan at parse, so emit it only where needed: the runner
  // chdirs itself and reports run-skips out-of-band, so a plain test is a bare add_test. only the
  // FIXTURES_SETUP keeps SKIP_RETURN_CODE, to cascade a compile-skip to its runs
  const auto joinEnv = [&](const std::string &extra) { return env.empty() ? extra : extra.empty() ? env : env + ";" + extra; };
  const auto envProp = [&](const std::string &extra) {
    const auto e = joinEnv(extra);
    return e.empty() ? std::string{} : fmt::format(" ENVIRONMENT \"{}\"", e);
  };

  unsigned long long seed = 0x9e3779b97f4a7c15ull;
  if (const char *s = std::getenv(env::PolytestSeed)) {
    char *end = nullptr;
    if (const auto v = std::strtoull(s, &end, 10); end != s && *end == '\0') seed = v;
  }

  std::ofstream(file, std::ios::binary)
      << (tasks                                         //
          ^ sort_by([](const auto &t) { return t.id; }) //
          ^ shuffle(std::mt19937_64(seed))              //
          ^ mk_string("", [&](const CtestEntry &t) {
              const auto name = prefix.empty() ? t.id : prefix + "-" + t.id;
              if (t.variants.empty()) {
                auto line = fmt::format("add_test(\"{}\" \"{}\" --run-task \"{}\")\n", name, binary, t.id);
                const auto labelsProp = t.labels.empty() ? std::string{} : fmt::format(" LABELS \"{}\"", t.labels);
                if (const auto p = labelsProp + envProp({}); !p.empty())
                  line += fmt::format("set_tests_properties(\"{}\" PROPERTIES{})\n", name, p);
                return line;
              }
              // shared-compile: one setup, a run per variant, one cleanup, gated on a per-task fixture
              const auto fix = "fix-" + t.id;
              auto frag = fmt::format("add_test(\"compile-{}\" \"{}\" --compile-task \"{}\")\n", name, binary, t.id) +
                          fmt::format("set_tests_properties(\"compile-{}\" PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_SETUP \"{}\"{})\n", name,
                                      fix, envProp({}));
              frag += t.variants ^ mk_string("", [&](const auto &suffix, const auto &extraEnv) {
                        const auto vname = suffix.empty() ? name : name + "-" + suffix;
                        return fmt::format("add_test(\"{}\" \"{}\" --run-only-task \"{}\")\n", vname, binary, t.id) +
                               fmt::format("set_tests_properties(\"{}\" PROPERTIES FIXTURES_REQUIRED \"{}\"{})\n", vname, fix,
                                           envProp(extraEnv));
                      });
              return frag + fmt::format("add_test(\"cleanup-{}\" \"{}\" --cleanup-task \"{}\")\n", name, binary, t.id) +
                     fmt::format("set_tests_properties(\"cleanup-{}\" PROPERTIES FIXTURES_CLEANUP \"{}\"{})\n", name, fix, envProp({}));
            }));
}

inline std::string distEnv(const std::string &filesSubdir) {
  return fmt::format("{}=${{CMAKE_CURRENT_LIST_DIR}}/../{};{}=${{CMAKE_CURRENT_LIST_DIR}}/../test/profiles;{}=${{CMAKE_CURRENT_LIST_DIR}}",
                     env::PolytestFilesDir, filesSubdir, env::PolytestProfileDir, env::PolytestWorkDir);
}

} // namespace polyregion::polytest
