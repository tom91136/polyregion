#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/env_keys.h"

namespace polyregion::polytest {

inline void emitCtestFragment(const std::string &file, const std::string &prefix, const std::string &binary, const std::string &workdir,
                              const std::string &env, const std::vector<std::pair<std::string, std::string>> &tasks) {
  using namespace aspartame;
  const auto out = tasks ^ mk_string("", [&](auto &id, auto &extraLabels) {
                     const auto name = prefix.empty() ? id : prefix + "-" + id;
                     std::string labels = id ^ starts_with("offload-") ? "device" : "";
                     if (!extraLabels.empty()) labels = labels.empty() ? extraLabels : labels + ";" + extraLabels;
                     return fmt::format("add_test(\"{}\" \"{}\" --run-task \"{}\")\n"
                                        "set_tests_properties(\"{}\" PROPERTIES SKIP_RETURN_CODE 77{}{}{})\n",
                                        name, binary, id, name,                                                   //
                                        workdir.empty() ? "" : fmt::format(" WORKING_DIRECTORY \"{}\"", workdir), //
                                        labels.empty() ? "" : fmt::format(" LABELS \"{}\"", labels),              //
                                        env.empty() ? "" : fmt::format(" ENVIRONMENT \"{}\"", env));
                   });
  std::ofstream(file, std::ios::binary) << out;
}

inline std::string distEnv(const std::string &filesSubdir) {
  return fmt::format("{}=${{CMAKE_CURRENT_LIST_DIR}}/../{};{}=${{CMAKE_CURRENT_LIST_DIR}}/../test/profiles;{}=${{CMAKE_CURRENT_LIST_DIR}}",
                     env::PolytestFilesDir, filesSubdir, env::PolytestProfileDir, env::PolytestWorkDir);
}

} // namespace polyregion::polytest
