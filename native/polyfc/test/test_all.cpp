#include "test_all.h"

#include "catch2/catch_test_macros.hpp"

#include "polytest/driver.hpp"

static polyregion::polytest::DriverConfig mkConfig() {
  return {
      .driverPath = FlangDriver,
      .binaryDir = BinaryDir,
      .testFiles = TestFiles,
      .profileDir = POLYREGION_TEST_PROFILE_DIR,
      .archVar = "polyfc_arch",
      .defaults = {"polyfc_defaults", "-O1 -g -cpp"},
      .stdpar = {"polyfc_stdpar", "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -lstdc++ -fstdpar-rt=dynamic"},
      .driverEnvVar = "POLYFC_DRIVER",
      .passthroughEnvs = {"POLYFC_NO_REWRITE=1", "POLYDCO_NO_OFFLOAD=1"},
      .outputPrefix = "polyfc_test_",
      .tempPrefix = "polyfc_",
      .directive = "!CHECK",
      .cleanupOnSuccess = true,
  };
}

TEST_CASE("offload") { polyregion::polytest::runTestSuite(mkConfig(), false); }

TEST_CASE("passthrough") { polyregion::polytest::runTestSuite(mkConfig(), true); }
