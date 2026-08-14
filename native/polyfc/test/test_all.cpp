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
  return runMain(
      argc, argv,
      DriverConfig{
          .driverPath = Driver,
          .binaryDir = BinaryDir,
          .workDir = WorkDir,
          .testFiles = TestFiles,
          .profileDir = envOr(polyregion::env::PolytestProfileDir, POLYREGION_TEST_PROFILE_DIR),
          .archVar = "polyfc_arch",
          .defaultsVar = "polyfc_defaults",
          .defaultsLabelVar = "opt",
          .defaultsVariants = {{"O0", POLYTEST_APPLE_TARGET_FLAG "-O0 -g -cpp"}, {"O3", POLYTEST_APPLE_TARGET_FLAG "-O3 -g -cpp"}},
          .extraVars = {{"package_fixture", packageFixture}},
          .stdpar = {"polyfc_stdpar",
#ifdef _WIN32
                     // XXX Windows CUDA/HIP have no HMM, so plain heap pointers can't reach
                     // the GPU. Use mem=interpose to route Fortran allocations through
                     // polyrt_usm_* (cuMemAllocManaged / hipMallocManaged) so kernels see USM.
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fstdpar-mem=interpose -fstdpar-rt=static"
#elif defined(__APPLE__)
                     // no -lstdc++: flang's libc++ has no rpath; libpolydco covers it.
                     POLYTEST_APPLE_TARGET_FLAG
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -fstdpar-rt=dynamic"
#else
                     "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -lstdc++ -fstdpar-rt=dynamic"
#endif
          },
          .driverEnvVar = polyregion::env::PolyfcDriver,
          .passthroughEnvs = {std::string(polyregion::env::PolyfcNoRewrite) + "=1"},
          .outputPrefix = "polyfc_test_",
          .tempPrefix = "polyfc_",
          .directive = "!CHECK",
          .cleanupOnSuccess = true,
      });
}
