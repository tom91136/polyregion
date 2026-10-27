#include "polytest/driver.hpp"

#include <algorithm>
#include <fstream>
#include <string>

#include "llvm/Support/Process.h"

#include <catch2/catch_test_macros.hpp>

namespace {

struct TempDir {
  std::string path;
  TempDir() {
    llvm::SmallString<256> p;
    llvm::sys::fs::createUniqueDirectory("polytest-driver-test", p);
    path = std::string(p);
  }
  ~TempDir() { llvm::sys::fs::remove_directories(path); }
  std::string file(const std::string &name) const { return path + "/" + name; }
  void touch(const std::string &name) const { std::ofstream(file(name)) << "x"; }
};

polyregion::polytest::DriverConfig testConfig(const std::string &dir) {
  return {.driverPath = "driver",
          .binaryDir = dir,
          .workDir = dir,
          .testFiles = {},
          .profileDir = dir,
          .archVar = "arch",
          .defaultsVar = "defaults",
          .defaultsLabelVar = "opt",
          .defaultsVariants = {{"O0", ""}},
          .extraVars = {},
          .stdpar = {"stdpar", ""},
          .driverEnvVar = "TEST_DRIVER",
          .passthroughEnvs = {},
          .outputPrefix = "out_",
          .directive = "#pragma region",
          .cleanupOnSuccess = true};
}

} // namespace

TEST_CASE("Artifact removal clears the output and its dotted siblings only") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto output = dir.file("out_a_offload");
  dir.touch("out_a_offload");
  dir.touch("out_a_offload.o");
  dir.touch("out_a_offload.polyast");
  dir.touch("out_a_offload.stdout");
  llvm::sys::fs::create_directories(output + ".packages/nested");
  dir.touch("out_a_offload.packages/nested/blob");
  dir.touch("out_a_offload_other.o");
  dir.touch("out_a_offloadx");
  dir.touch("module-cache");

  detail::removeArtifacts(output);

  CHECK_FALSE(llvm::sys::fs::exists(output));
  CHECK_FALSE(llvm::sys::fs::exists(output + ".o"));
  CHECK_FALSE(llvm::sys::fs::exists(output + ".polyast"));
  CHECK_FALSE(llvm::sys::fs::exists(output + ".stdout"));
  CHECK_FALSE(llvm::sys::fs::exists(output + ".packages"));
  CHECK(llvm::sys::fs::exists(dir.file("out_a_offload_other.o")));
  CHECK(llvm::sys::fs::exists(dir.file("out_a_offloadx")));
  CHECK(llvm::sys::fs::exists(dir.file("module-cache")));
}

TEST_CASE("Artifact removal through a relative output name clears its siblings") {
  using namespace polyregion::polytest;
  TempDir dir;
  dir.touch("out_b_offload");
  dir.touch("out_b_offload.pdb");
  llvm::sys::fs::create_directories(dir.file("out_b_offload.packages/nested"));
  dir.touch("out_b_offload.packages/nested/blob");
  llvm::SmallString<256> cwd;
  REQUIRE_FALSE(llvm::sys::fs::current_path(cwd));
  REQUIRE_FALSE(llvm::sys::fs::set_current_path(dir.path));

  detail::removeArtifacts("out_b_offload");

  REQUIRE_FALSE(llvm::sys::fs::set_current_path(cwd));
  CHECK_FALSE(llvm::sys::fs::exists(dir.file("out_b_offload")));
  CHECK_FALSE(llvm::sys::fs::exists(dir.file("out_b_offload.pdb")));
  CHECK_FALSE(llvm::sys::fs::exists(dir.file("out_b_offload.packages")));
}

TEST_CASE("Task output names are stable across processes") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host");
  std::ofstream(dir.file("check_one.cpp")) << "#pragma region case: first\n#pragma region using: v=a\n#pragma region do: echo {output}\n"
                                              "#pragma region case: second\n#pragma region using: v=a\n#pragma region do: echo {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_one.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  REQUIRE(tasks.size() == 2);
  CHECK(tasks[0].output != tasks[1].output);
  for (const auto &t : tasks) {
    CHECK(t.output.find(std::to_string(llvm::sys::Process::getProcessId())) == std::string::npos);
    CHECK(t.output.starts_with("out_one_"));
  }
}

TEST_CASE("Cache and library locations reach the child environment") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv cache(polyregion::env::PolyregionCacheDir, "/polytest/cache");
  polyregion::env::ScopedEnv xdg("XDG_CACHE_HOME", "/polytest/xdg");
  polyregion::env::ScopedEnv pocl("POCL_CACHE_DIR", "/polytest/pocl");
  const auto cfg = testConfig(dir.path);
  const Task task{Mode::Offload, "check_x.cpp", "c", {}, "out", std::nullopt, {}, {}};
  const auto envs = detail::baseEnvs(task, cfg, false);
  const auto has = [&](const std::string &kv) { return std::find(envs.begin(), envs.end(), kv) != envs.end(); };
  CHECK(has("POLYREGION_CACHE_DIR=/polytest/cache"));
  CHECK(has("XDG_CACHE_HOME=/polytest/xdg"));
  CHECK(has("POCL_CACHE_DIR=/polytest/pocl"));
}

TEST_CASE("Skip reason is the last runtime reason line on stderr") {
  using namespace polyregion::polytest;
  const std::string fallback = "polyrt: no compatible target (exit 77)";
  CHECK(detail::skipReason("noise\npolyrt: no compatible target: SwiftShader Device lacks int64\r\ntrailer\n", fallback)
        == "polyrt: no compatible target: SwiftShader Device lacks int64");
  CHECK(detail::skipReason("polyrt: no compatible target: no device\npolyrt: no compatible target: X lacks fp64\n", fallback)
        == "polyrt: no compatible target: X lacks fp64");
  CHECK(detail::skipReason("", fallback) == fallback);
  CHECK(detail::skipReason("unrelated skip\n", fallback) == fallback);
}

#ifndef _WIN32
TEST_CASE("A run step exiting 77 reports the runtime reason as the skip reason") {
  using namespace polyregion::polytest;
  TempDir dir;
  std::ofstream(dir.file("skip.sh")) << "#!/bin/sh\necho 'polyrt: no compatible target: SwiftShader Device lacks int64' >&2\nexit 77\n";
  llvm::sys::fs::setPermissions(dir.file("skip.sh"), llvm::sys::fs::perms::all_all);
  const auto cfg = testConfig(dir.path);
  const Task task{
      Mode::Offload, "check_x.cpp", "c", {}, dir.file("out"), std::nullopt, {{"compile", {}, {}}, {dir.file("skip.sh"), {}, {}}}, {}};
  const auto outcome = detail::runTask(task, cfg);
  CHECK(outcome.status == TaskStatus::Skip);
  CHECK(outcome.failReason == "polyrt: no compatible target: SwiftShader Device lacks int64");
}
#endif
