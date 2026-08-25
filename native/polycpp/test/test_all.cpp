#include "test_all.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "polyregion/env_keys.h"

#include "polytest/driver.hpp"

int main(int argc, const char **argv) {
  using namespace polyregion::polytest;
  llvm::SmallString<256> siblingFixture(argv[0]);
  llvm::sys::fs::make_absolute(siblingFixture);
  llvm::sys::path::remove_filename(siblingFixture);
  llvm::sys::path::append(siblingFixture, llvm::sys::path::filename(PackageFixture));
  const auto packageFixture = llvm::sys::fs::exists(siblingFixture) ? siblingFixture.str().str() : std::string(PackageFixture);
  llvm::SmallString<256> siblingPolyc(argv[0]);
  llvm::sys::fs::make_absolute(siblingPolyc);
  llvm::sys::path::remove_filename(siblingPolyc);
#ifdef _WIN32
  llvm::sys::path::append(siblingPolyc, "..", "polyc", "polyc.exe");
#else
  llvm::sys::path::append(siblingPolyc, "..", "polyc", "polyc");
#endif
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
          .defaultsVariants = {{"O0", POLYTEST_APPLE_TARGET_FLAG "-fno-crash-diagnostics -O0 -g3 -Wall -Wextra -pedantic -std=c++17"},
                               {"O3", POLYTEST_APPLE_TARGET_FLAG "-fno-crash-diagnostics -O3 -g3 -Wall -Wextra -pedantic -std=c++17"}},
          .extraVars = {{"package_fixture", packageFixture}, {"polypackage_emit", siblingPolyc.str().str()}},
          .stdpar = {"polycpp_stdpar",
#ifdef _WIN32
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=static -v"
#else
                     POLYTEST_APPLE_TARGET_FLAG
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polycpp_arch} -fstdpar-mem=reflect -fstdpar-rt=dynamic -v"
#endif
          },
          .driverEnvVar = polyregion::env::PolycppDriver,
          .passthroughEnvs = {std::string(polyregion::env::PolycppNoRewrite) + "=1", std::string(polyregion::env::PolystlNoOffload) + "=1"},
          .outputPrefix = "polycpp_test_",
          .tempPrefix = "polycpp_",
          .directive = "#pragma region",
          .cleanupOnSuccess = true,
      });
}
