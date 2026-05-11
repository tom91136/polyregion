#include "test_all.h"

#include "polytest/driver.hpp"

int main(int argc, const char **argv) {
  using namespace polyregion::polytest;
  return runMain(argc, argv,
                 DriverConfig{
                     .driverPath = FlangDriver,
                     .binaryDir = BinaryDir,
                     .testFiles = TestFiles,
                     .profileDir = POLYREGION_TEST_PROFILE_DIR,
                     .archVar = "polyfc_arch",
                     .defaults = {"polyfc_defaults", "-O1 -g -cpp"},
                     .stdpar = {"polyfc_stdpar",
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -lstdc++ -fstdpar-rt=dynamic"},
                     .driverEnvVar = "POLYFC_DRIVER",
                     .passthroughEnvs = {"POLYFC_NO_REWRITE=1", "POLYDCO_NO_OFFLOAD=1"},
                     .outputPrefix = "polyfc_test_",
                     .tempPrefix = "polyfc_",
                     .directive = "!CHECK",
                     .cleanupOnSuccess = true,
                 });
}
