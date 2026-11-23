#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
  #include <unistd.h>

  #include "polyregion/dl.h"

  #include "polytest/test_case.hpp"
  #include <sys/wait.h>

using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

std::string procSelfExe() {
  auto *const resolved = realpath("/proc/self/exe", nullptr);
  if (!resolved) return {};
  std::string path(resolved);
  free(resolved);
  return path;
}

int runMainExecutable() {
  Context ctx;
  const auto path = polyregion_dl_module_path(reinterpret_cast<const void *>(&runMainExecutable));
  POLYTEST_CHECK_S(ctx, !path.empty() && path == procSelfExe(), "module path '{}' is not the main executable", path);
  return ctx.failed ? 1 : 0;
}

int runBareArgv0() {
  Context ctx;
  std::vector<std::string> args{"dl-module-path", "--run-task", "dl-module-path-main-executable"};
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);

  fflush(nullptr);
  const pid_t child = fork();
  POLYTEST_REQUIRE_S(ctx, child >= 0, "fork failed");
  if (child == 0) {
    execv("/proc/self/exe", argv.data());
    _exit(127);
  }
  int status = 0;
  POLYTEST_REQUIRE_S(ctx, waitpid(child, &status, 0) == child, "waitpid failed");
  POLYTEST_CHECK_S(ctx, WIFEXITED(status) && WEXITSTATUS(status) == 0,
                   "main executable run under bare argv0 did not resolve to itself (status {})", status);
  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() {
  return {
      Task{"dl-module-path-main-executable", "", &runMainExecutable},
      Task{"dl-module-path-bare-argv0", "", &runBareArgv0},
  };
}

} // namespace

POLYTEST_DISCOVER(discoverAll)

#endif
