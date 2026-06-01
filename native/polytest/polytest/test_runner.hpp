#pragma once

#include <string_view>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polytest/test_case.hpp"

namespace polyregion::polytest::cases {

// Modes:
//   --list-ids               one id per line on stdout
//   --run-task <id>          run a single task; exit 0/77/1
//   (no args)                run all tasks in registration order
inline int runMain(int argc, char **argv) {
  std::string_view mode = "all";
  std::string_view target;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--list-ids") mode = "list";
    else if (a == "--run-task" && i + 1 < argc) {
      mode = "run";
      target = argv[++i];
    } else if (a == "--help" || a == "-h") {
      fmt::print("Usage: {} [--list-ids | --run-task <id>]\n", argv[0]);
      return 0;
    } else {
      fmt::print(stderr, "Unknown arg: {}\n", a);
      return 2;
    }
  }

  using namespace aspartame;

  auto all = ::polyregion::polytest::cases::discoverers() ^ flat_map([](auto &d) { return d(); });

  if (mode == "list") {
    for (auto &t : all) {
      if (t.labels.empty()) fmt::print("{}\n", t.id);
      else fmt::print("{}\t{}\n", t.id, t.labels);
    }
    return 0;
  }

  auto runOne = [](const Task &t) -> int {
    fmt::print(stderr, "[RUN ] {}\n", t.id);
    int rc;
    try {
      rc = t.run();
    } catch (const SkipRequested &) {
      fmt::print(stderr, "[DONE] {} -> skip\n", t.id);
      return 77;
    } catch (const RequireFailed &) {
      fmt::print(stderr, "[DONE] {} -> fail (REQUIRE)\n", t.id);
      return 1;
    } catch (const std::exception &e) {
      fmt::print(stderr, "[DONE] {} -> fail (exception: {})\n", t.id, e.what());
      return 1;
    } catch (...) {
      fmt::print(stderr, "[DONE] {} -> fail (unknown exception)\n", t.id);
      return 1;
    }
    fmt::print(stderr, "[DONE] {} -> {}\n", t.id, rc == 0 ? "pass" : rc == 77 ? "skip" : "fail");
    return rc;
  };

  if (mode == "run") {
    if (auto found = all ^ find([&](auto &t) { return t.id == target; })) return runOne(*found);
    fmt::print(stderr, "No task with id `{}`\n", target);
    return 2;
  }

  int worst = 0;
  for (auto &t : all) {
    const int rc = runOne(t);
    if (rc != 0 && rc != 77 && worst == 0) worst = rc;
  }
  return worst;
}

} // namespace polyregion::polytest::cases
