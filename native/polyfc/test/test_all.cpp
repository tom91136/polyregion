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
                     .defaultsVar = "polyfc_defaults",
                     .defaultsLabelVar = "opt",
                     .defaultsVariants = {{"O0", "-O0 -g -cpp"}, {"O3", "-O3 -g -cpp"}},
                     .stdpar = {"polyfc_stdpar",
#ifdef _WIN32
                                // polyreflect-plugin is not built on Windows; force mem=direct + static rt
                                // to exercise the static-fold compile path.
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fstdpar-mem=direct -fstdpar-rt=static"
#else
                                "-fstdpar -fstdpar-verbose=debug -fstdpar-arch={polyfc_arch} -fuse-ld=lld -lstdc++ -fstdpar-rt=dynamic"
#endif
                     },
                     .driverEnvVar = "POLYFC_DRIVER",
                     .passthroughEnvs = {"POLYFC_NO_REWRITE=1", "POLYDCO_NO_OFFLOAD=1"},
                     .outputPrefix = "polyfc_test_",
                     .tempPrefix = "polyfc_",
                     .directive = "!CHECK",
                     .cleanupOnSuccess = true,
                 });
}
