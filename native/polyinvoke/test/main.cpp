#include "polytest/profile.hpp"
#include "polytest/test_runner.hpp"

int main(int argc, char **argv) {
  polyregion::polytest::applyProfileEnvironment(POLYREGION_TEST_PROFILE_DIR);
  return polyregion::polytest::cases::runMain(argc, argv);
}
