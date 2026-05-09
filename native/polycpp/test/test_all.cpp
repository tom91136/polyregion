#include "test_all.h"

#include "catch2/catch_test_macros.hpp"

#include "polytest/driver.hpp"

static polyregion::polytest::DriverConfig mkConfig() {
  return {
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
  };
}

TEST_CASE("offload") { polyregion::polytest::runTestSuite(mkConfig(), false); }

TEST_CASE("passthrough") { polyregion::polytest::runTestSuite(mkConfig(), true); }
