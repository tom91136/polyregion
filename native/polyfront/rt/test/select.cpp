#include <cstdlib>
#include <deque>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include <catch2/catch_test_macros.hpp>

#ifdef POLYRT_TESTS_POLYDCO
  #include "polydco/polydco.h"
#endif
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyrt/rt.h"

namespace {

void selfAnchor() {}

struct ProbeResult {
  int code;
  std::string stderrText;
};

ProbeResult probe(const std::string &testName, const std::vector<std::pair<std::string, std::string>> &env, unsigned secondsToWait = 0) {
  std::deque<polyregion::env::ScopedEnv> scoped;
  for (const auto &[name, value] : env)
    scoped.emplace_back(name, value.c_str());
  llvm::SmallString<128> errorPath;
  REQUIRE_FALSE(llvm::sys::fs::createTemporaryFile("polyrt-select-probe", "txt", errorPath));
  const auto self = llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&selfAnchor));
  const std::vector<llvm::StringRef> args{self, testName};
  std::string executionError;
  const int code = llvm::sys::ExecuteAndWait(self, args, std::nullopt, {std::nullopt, llvm::StringRef(), errorPath.str()}, secondsToWait, 0,
                                             &executionError);
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
  CHECK(errors.find("polyrt: no compatible target: Backend CUDA selected by POLYRT_PLATFORM failed to initialise (strict)")
        != std::string::npos);
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

TEST_CASE("skip names the features the device lacks, once per feature") {
  const auto [code, errors] = probe("skip probe: device lacks required features", {{polyregion::env::PolyrtDebug, "0"}});
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("polyrt: no compatible target: stub lacks int64, fp64, fp16\n") != std::string::npos);
  CHECK(
      errors.find(
          R"("module":"lacking","kind":"Managed","format":"Object","features":["int64","fp64","fp16"],"outcome":"lacks","detail":["int64","fp64","fp16"])")
      != std::string::npos);
}

TEST_CASE("skip names the image format the device cannot load") {
  const auto [code, errors] = probe("skip probe: device format differs from every image", {{polyregion::env::PolyrtDebug, "0"}});
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("polyrt: no compatible target: stub cannot load Object images\n") != std::string::npos);
  CHECK(errors.find(R"("outcome":"format","detail":["Object"])") != std::string::npos);
}

TEST_CASE("skip without any offered image says so") {
  const auto [code, errors] = probe("skip probe: nothing was offered", {{polyregion::env::PolyrtDebug, "0"}});
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("polyrt: no compatible target: no image offered\n") != std::string::npos);
  CHECK(errors.find(R"(polyrt-skip: {"device":null,"attempts":[]})") != std::string::npos);
}

TEST_CASE("skip without a device says so") {
  const auto [code, errors] = probe("skip probe: no device", {{polyregion::env::PolyrtDebug, "0"}});
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("polyrt: no compatible target: no device\n") != std::string::npos);
  CHECK(errors.find(R"("outcome":"no-device")") != std::string::npos);
}

TEST_CASE("select probe: exit during transaction", "[.probe]") {
  auto *context = polyregion::polyrt::currentContext();
  REQUIRE(context);
  REQUIRE(polyrt_context_acquire(context));
  std::thread([] { std::exit(9); }).join();
}

TEST_CASE("process exit does not wait on a transaction held by another thread") {
  const auto [code, errors] = probe("select probe: exit during transaction", {}, 20);
  INFO(errors);
  CHECK(code == 9);
}

#ifdef POLYRT_TESTS_POLYDCO
TEST_CASE("select probe: polydco host fallback", "[.probe]") { polydco_host_fallback(); }

TEST_CASE("polydco host fallback skips when host fallback is disabled") {
  const auto [code, errors] = probe("select probe: polydco host fallback", {{polyregion::env::PolyrtHostFallback, "0"}});
  INFO(errors);
  CHECK(code == 77);
  CHECK(errors.find("polyrt: no compatible target: polydco_host_fallback: no backend ran the loop and host fallback is disabled")
        != std::string::npos);
}

TEST_CASE("polydco host fallback runs the host loop by default") {
  const auto [code, errors] = probe("select probe: polydco host fallback", {{polyregion::env::PolyrtHostFallback, "1"}});
  INFO(errors);
  CHECK(code == 0);
}
#endif
