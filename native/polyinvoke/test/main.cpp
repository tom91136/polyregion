#include "polyregion/env.h"

#include "polytest/profile.hpp"
#include "polytest/test_runner.hpp"

int main(int argc, char **argv) {
  polyregion::env::put("__GL_SHADER_DISK_CACHE", "0", false);
  polyregion::env::put("POCL_KERNEL_CACHE", "0", false);
  polyregion::polytest::applyProfileEnvironment(POLYREGION_TEST_PROFILE_DIR);
  return polyregion::polytest::cases::runMain(argc, argv);
}
