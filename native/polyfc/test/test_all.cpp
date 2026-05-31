#include "test_all.h"

#include "polyregion/env_keys.h"

#include "polytest/driver.hpp"

int main(int argc, const char **argv) {
  using namespace polyregion::polytest;
  return runMain(argc, argv,
                 DriverConfig{
                     .driverPath = Driver,
                     .binaryDir = BinaryDir,
                     .workDir = WorkDir,
                     .testFiles = TestFiles,
                     .profileDir = envOr(polyregion::env::PolytestProfileDir, POLYREGION_TEST_PROFILE_DIR),
                     .archVar = "polyfc_arch",
                     .defaultsVar = "polyfc_defaults",
                     .defaultsLabelVar = "opt",
                     .defaultsVariants = {{"O0", "-O0 -g -cpp"}, {"O3", "-O3 -g -cpp"}},
                     .stdpar = {"polyfc_stdpar",
#ifdef _WIN32
                                // polyreflect-plugin is not built on Windows; force mem=direct + static rt
                                // to exercise the static-fold compile path.
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fstdpar-mem=direct -fstdpar-rt=static"
#elif defined(__APPLE__)
                                // no -lstdc++: flang's libc++ has no rpath; libpolydco covers it.
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -fstdpar-rt=dynamic"
#else
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -lstdc++ -fstdpar-rt=dynamic"
#endif
                     },
                     .driverEnvVar = "POLYFC_DRIVER",
                     .passthroughEnvs = {std::string(polyregion::env::PolyfcNoRewrite) + "=1"},
                     .outputPrefix = "polyfc_test_",
                     .tempPrefix = "polyfc_",
                     .directive = "!CHECK",
                     .cleanupOnSuccess = true,
                 });
}
