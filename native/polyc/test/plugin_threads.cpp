#include <string>
#include <thread>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include "catch2/catch_test_macros.hpp"

#include "polyregion/polypackage_symbols.h"

#include "dso_runner.h"
#include "polypass_locate.h"

using namespace polyregion;

namespace {

constexpr auto ProbeTag = "[plugin-thread-probe]";
const char anchor = 0;

std::string pluginPath() {
  std::string error;
  for (const auto &plugin : polypass::resolvePlugins(error))
    if (plugin.kind == polypass::PluginKind::Dso) return plugin.path;
  return {};
}

} // namespace

TEST_CASE("plugin thread probe", "[.][plugin-thread-probe]") {
  const auto path = pluginPath();
  REQUIRE_FALSE(path.empty());
  polypass::DsoPassRunner runner(path);
  std::string loadError;
  std::thread([&] { loadError = runner.load(); }).join();
  REQUIRE(loadError.empty());
  for (int i = 0; i < 8; ++i) {
    std::string error;
    runner.runPackage(polypackage::abi::LinkPackage, {}, error);
    CHECK_FALSE(error.empty());
  }
}

TEST_CASE("a plugin first loaded on a finished thread survives later collections", "[plugin-thread]") {
  const auto exe = llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(const_cast<char *>(&anchor)));
  REQUIRE_FALSE(exe.empty());
  CHECK(llvm::sys::ExecuteAndWait(exe, {exe, ProbeTag}) == 0);
}
