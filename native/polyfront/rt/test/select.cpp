#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include <catch2/catch_test_macros.hpp>

#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyrt/rt.h"

namespace {

void selfAnchor() {}

struct ProbeResult {
  int code;
  std::string stderrText;
};

ProbeResult probe(const std::string &testName, const std::vector<std::pair<std::string, std::string>> &env) {
  std::deque<polyregion::env::ScopedEnv> scoped;
  for (const auto &[name, value] : env)
    scoped.emplace_back(name, value.c_str());
  llvm::SmallString<128> errorPath;
  REQUIRE_FALSE(llvm::sys::fs::createTemporaryFile("polyrt-select-probe", "txt", errorPath));
  const auto self = llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&selfAnchor));
  const std::vector<llvm::StringRef> args{self, testName};
  std::string executionError;
  const int code =
      llvm::sys::ExecuteAndWait(self, args, std::nullopt, {std::nullopt, llvm::StringRef(), errorPath.str()}, 0, 0, &executionError);
  std::string errors = executionError;
  if (const auto buffer = llvm::MemoryBuffer::getFile(errorPath)) errors += (*buffer)->getBuffer().str();
  llvm::sys::fs::remove(errorPath);
  return {code, std::move(errors)};
}

std::vector<std::pair<std::string, std::string>> failingBackend(const char *strict) {
  return {{polyregion::env::PolyrtPlatform, "cuda@*"},
          {polyregion::env::PolyinvokeDisableBackends, "CUDA"},
          {polyregion::env::PolyrtStrictSelect, strict},
          {polyregion::env::PolyrtDebug, "0"}};
}

} // namespace

TEST_CASE("select probe: initialise", "[.probe]") { polyregion::polyrt::initialise(); }

TEST_CASE("strict selection skips when the selected backend fails to initialise") {
  const auto [code, errors] = probe("select probe: initialise", failingBackend("1"));
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("Backend CUDA selected by POLYRT_PLATFORM failed to initialise (strict), exiting 77 (skip)") != std::string::npos);
}

TEST_CASE("non-strict selection continues without a platform when the backend fails to initialise") {
  const auto [code, errors] = probe("select probe: initialise", failingBackend("0"));
  INFO(errors);
  CHECK(code == 0);
}

TEST_CASE("strict selection leaves an unrecognised platform to the fallback") {
  auto env = failingBackend("1");
  env.front().second = "none";
  const auto [code, errors] = probe("select probe: initialise", env);
  INFO(errors);
  CHECK(code == 0);
}
