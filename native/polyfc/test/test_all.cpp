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
          .archVar = "polyfc_arch",
          .defaultsVar = "polyfc_defaults",
          .defaultsLabelVar = "opt",
          .defaultsVariants = {{"O0", POLYTEST_APPLE_TARGET_FLAG "-O0 -g -cpp"}, {"O3", POLYTEST_APPLE_TARGET_FLAG "-O3 -g -cpp"}},
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
