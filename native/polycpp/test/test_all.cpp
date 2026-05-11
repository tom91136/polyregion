#include "test_all.h"

#include "polytest/driver.hpp"

int main(int argc, const char **argv) {
  using namespace polyregion::polytest;
  return runMain(argc, argv,
                 DriverConfig{
                     .driverPath = ClangDriver,
                     .binaryDir = BinaryDir,
                     .testFiles = TestFiles,
                     .profileDir = POLYREGION_TEST_PROFILE_DIR,
                     .archVar = "polycpp_arch",
                     .defaults = {"polycpp_defaults", "-fno-crash-diagnostics -O1 -g3 -Wall -Wextra -pedantic"},
                     .stdpar = {"polycpp_stdpar",
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=dynamic -v"},
                     .driverEnvVar = "POLYCPP_DRIVER",
                     .passthroughEnvs = {"POLYCPP_NO_REWRITE=1", "POLYSTL_NO_OFFLOAD=1"},
                     .outputPrefix = "polycpp_test_",
                     .tempPrefix = "polycpp_",
                     .directive = "#pragma region",
                     .cleanupOnSuccess = false,
                 });
}
