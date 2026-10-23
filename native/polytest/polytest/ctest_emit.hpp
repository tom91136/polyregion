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

namespace polyregion::polytest {

using namespace aspartame;

inline void reportSkip(const std::string &id, const std::string &reason) {
  fmt::print(stderr, "[SKIP] {}{}{}\n", id, reason.empty() ? "" : ": ", reason);
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
  // set_tests_properties scans every test per property at parse (O(n^2)), so emit only the properties a
  // test needs; the runner chdirs itself, so no WORKING_DIRECTORY
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
                const auto labelsProp = t.labels.empty() ? std::string{} : fmt::format(" LABELS \"{}\"", t.labels);
                return fmt::format("add_test(\"{}\" \"{}\" --run-task \"{}\")\n", name, binary, t.id)
                       + fmt::format("set_tests_properties(\"{}\" PROPERTIES SKIP_RETURN_CODE 77{}{})\n", name, labelsProp, envProp({}));
              }
              // shared-compile: one setup, a run per variant, one cleanup, gated on a per-task fixture
              const auto fix = "fix-" + name;
              const auto labelsProp = t.labels.empty() ? std::string{} : fmt::format(" LABELS \"{}\"", t.labels);
              auto frag = fmt::format("add_test(\"compile-{}\" \"{}\" --compile-task \"{}\")\n", name, binary, t.id)
                          + fmt::format("set_tests_properties(\"compile-{}\" PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_SETUP \"{}\"{}{})\n",
                                        name, fix, labelsProp, envProp({}));
              frag += t.variants ^ mk_string("", [&](const auto &suffix, const auto &extraEnv) {
                        const auto vname = suffix.empty() ? name : name + "-" + suffix;
                        return fmt::format("add_test(\"{}\" \"{}\" --run-only-task \"{}\")\n", vname, binary, t.id)
                               + fmt::format("set_tests_properties(\"{}\" PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_REQUIRED \"{}\" "
                                             "RESOURCE_LOCK \"{}\"{}{})\n",
                                             vname, fix, fix, labelsProp, envProp(extraEnv));
                      });
              return frag + fmt::format("add_test(\"cleanup-{}\" \"{}\" --cleanup-task \"{}\")\n", name, binary, t.id)
                     + fmt::format("set_tests_properties(\"cleanup-{}\" PROPERTIES FIXTURES_CLEANUP \"{}\"{}{})\n", name, fix, labelsProp,
                                   envProp({}));
            }));
}

inline std::string distEnv(const std::string &filesSubdir) {
  return fmt::format("{}=${{CMAKE_CURRENT_LIST_DIR}}/../{};{}=${{CMAKE_CURRENT_LIST_DIR}}/../test/profiles;{}=${{CMAKE_CURRENT_LIST_DIR}}",
                     env::PolytestFilesDir, filesSubdir, env::PolytestProfileDir, env::PolytestWorkDir);
}

} // namespace polyregion::polytest
