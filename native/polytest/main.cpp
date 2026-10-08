#include <cstdio>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "polyregion/env.h"
#include "polyregion/env_keys.h"

#include "polytest/driver.hpp"

using namespace polyregion::polytest;

namespace {

struct Suite {
  std::vector<std::pair<std::string, std::string>> values;

  std::string get(const std::string &key, const std::string &fallback = {}) const {
    std::string value = fallback;
    for (const auto &[k, v] : values)
      if (k == key) value = v;
    return value;
  }
  std::vector<std::string> all(const std::string &key) const {
    std::vector<std::string> out;
    for (const auto &[k, v] : values)
      if (k == key) out.emplace_back(v ^ starts_with(":") ? v ^ drop(1) : v);
    return out;
  }
  std::vector<std::pair<std::string, std::string>> pairs(const std::string &key) const {
    return all(key) ^ collect([](const auto &v) { return v ^ split_once('='); });
  }
};

std::string distPath(const std::string &root, std::initializer_list<const char *> parts) {
  llvm::SmallString<256> p(root);
  for (const auto *part : parts)
    llvm::sys::path::append(p, part);
  return std::string(p);
}

void prependEnv(const char *name, const std::string &value, char separator) {
  const char *current = std::getenv(name);
  const auto joined = current && *current ? value + separator + current : value;
  polyregion::env::put(name, joined.c_str(), true);
}

} // namespace

int main(int argc, const char **argv) {
  const char *suitePath = std::getenv(polyregion::env::PolytestSuite);
  if (!suitePath || !*suitePath) {
    std::fprintf(stderr, "polytest: %s must name a suite file\n", polyregion::env::PolytestSuite);
    return 2;
  }
  if (!llvm::sys::fs::exists(suitePath)) {
    std::fprintf(stderr, "polytest: suite file does not exist: %s\n", suitePath);
    return 2;
  }
  const Suite suite{fileLines(suitePath) | collect(parseEnvLine) | to_vector()};

  const auto exe = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&distPath));
  const std::string bin(llvm::sys::path::parent_path(exe));
  const std::string root(llvm::sys::path::parent_path(bin));

  for (const auto &[key, value] : suite.pairs("POLYTEST_DEFAULT_ENV"))
    polyregion::env::put(key.c_str(), value.c_str(), false);

#if defined(_WIN32)
  prependEnv("PATH", distPath(root, {"lib", "polycpp", "lib"}) + ";" + distPath(root, {"lib"}) + ";" + bin, ';');
  constexpr auto exeSuffix = ".exe";
  constexpr auto stdparRt = "static";
#elif defined(__APPLE__)
  prependEnv("DYLD_FALLBACK_LIBRARY_PATH", distPath(root, {"lib"}) + ":" + distPath(root, {"lib", "polycpp", "lib"}), ':');
  constexpr auto exeSuffix = "";
  constexpr auto stdparRt = "dynamic";
#else
  prependEnv("LD_LIBRARY_PATH", distPath(root, {"lib"}) + ":" + distPath(root, {"lib", "polycpp", "lib"}), ':');
  constexpr auto exeSuffix = "";
  constexpr auto stdparRt = "dynamic";
#endif

  const char *profileOverride = std::getenv(polyregion::env::PolytestProfileDir);
  const auto profileDir = profileOverride && *profileOverride
                              ? std::string(profileOverride)
                              : suite.get("POLYTEST_PROFILE_DIR", distPath(root, {"share", "polyregion", "test-profiles"}));
  const auto stdpar = suite.get("POLYTEST_STDPAR")           //
                      ^ replace_all("{stdpar_rt}", stdparRt) //
                      ^ replace_all("{apple_target}", POLYTEST_APPLE_TARGET_FLAG);
  return runMain(argc, argv,
                 DriverConfig{
                     .driverPath = suite.get("POLYTEST_DRIVER", distPath(root, {"bin", "clang++"}) + exeSuffix),
                     .binaryDir = bin,
                     .workDir = suite.get("POLYTEST_WORK_DIR"),
                     .testFiles = suite.all("POLYTEST_FILES"),
                     .profileDir = profileDir,
                     .archVar = suite.get("POLYTEST_ARCH_VAR", "arch"),
                     .defaultsVar = "defaults",
                     .defaultsLabelVar = "variant",
                     .defaultsVariants = {{"default", ""}},
                     .extraVars = suite.pairs("POLYTEST_VAR"),
                     .stdpar = {suite.get("POLYTEST_STDPAR_VAR", "stdpar"), stdpar},
                     .driverEnvVar = suite.get("POLYTEST_DRIVER_ENV", polyregion::env::PolycppDriver),
                     .passthroughEnvs = {},
                     .outputPrefix = suite.get("POLYTEST_OUTPUT_PREFIX", "polytest_"),
                     .tempPrefix = suite.get("POLYTEST_TEMP_PREFIX", "polytest_"),
                     .directive = suite.get("POLYTEST_DIRECTIVE", "#pragma region"),
                     .cleanupOnSuccess = true,
                     .targetVar = suite.get("POLYTEST_TARGET_VAR"),
                     .targetValues = suite.pairs("POLYTEST_TARGET_VALUE"),
                 });
}
