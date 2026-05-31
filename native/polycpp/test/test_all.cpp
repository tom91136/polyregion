#include "test_all.h"

#include "polyregion/env_keys.h"

#include "polytest/driver.hpp"

int main(int argc, const char **argv) {
  using namespace polyregion::polytest;
  return runMain(
      argc, argv,
      DriverConfig{
          .driverPath = Driver,
          .binaryDir = BinaryDir,
          .workDir = WorkDir,
          .testFiles = TestFiles,
          .profileDir = envOr(polyregion::env::PolytestProfileDir, POLYREGION_TEST_PROFILE_DIR),
          .archVar = "polycpp_arch",
          .defaultsVar = "polycpp_defaults",
          .defaultsLabelVar = "opt",
          .defaultsVariants = {{"O0", "-fno-crash-diagnostics -O0 -g3 -Wall -Wextra -pedantic -std=c++17"},
                               {"O3", "-fno-crash-diagnostics -O3 -g3 -Wall -Wextra -pedantic -std=c++17"}},
          .stdpar = {"polycpp_stdpar",
#ifdef _WIN32
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=static -v"
#else
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=dynamic -v"
#endif
          },
          .driverEnvVar = "POLYCPP_DRIVER",
          .passthroughEnvs = {std::string(polyregion::env::PolycppNoRewrite) + "=1", std::string(polyregion::env::PolystlNoOffload) + "=1"},
          .outputPrefix = "polycpp_test_",
          .tempPrefix = "polycpp_",
          .directive = "#pragma region",
          .cleanupOnSuccess = true,
      });
}
