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

#ifndef _WIN32
namespace {

constexpr const char *SwiftShaderInventory =
    R"({"backend":"Vulkan","platform":"Vulkan","kind":"Managed","devices":[{"name":"SwiftShader Device","format":"SPIRV_GLCompute","features":["vulkan","spirv_glcompute","unknown"]}]})";

constexpr const char *Selector = "spirv_glcompute@!spirv_glcompute@*SwiftShader*";

void script(const TempDir &dir, const std::string &name, const std::string &body) {
  std::ofstream(dir.file(name)) << "#!/bin/sh\n" << body << "\n";
  llvm::sys::fs::setPermissions(dir.file(name), llvm::sys::fs::perms::all_all);
}

polyregion::polytest::Task skippingTask(const TempDir &dir, const std::string &record, const std::string &manifest,
                                        const std::string &selector = Selector) {
  using namespace polyregion::polytest;
  script(dir, "polyinfo", std::string("cat <<'EOF'\n") + SwiftShaderInventory + "\nEOF");
  script(dir, "skip.sh", "cat <<'EOF' >&2\n" + record + "\nEOF\nexit 77");
  std::ofstream(dir.file("out.images")) << manifest;
  return {Mode::Offload,
          "check_x.cpp",
          "c",
          {{"arch", selector}},
          dir.file("out"),
          std::nullopt,
          {{"compile", {}, {}}, {dir.file("skip.sh"), {}, {}}},
          {}};
}

constexpr const char *Fp64Manifest = R"({"module":"m","images":[{"kind":"Managed","format":"SPIRV_GLCompute","features":["fp64"]}]})";
constexpr const char *PlainManifest = R"({"module":"m","images":[{"kind":"Managed","format":"SPIRV_GLCompute","features":[]}]})";

std::string record(const std::string &device, const std::string &kind, const std::string &format, const std::string &features,
                   const std::string &outcome, const std::string &detail) {
  return R"(polyrt-skip: {"device":{"platform":"Vulkan","name":")" + device + R"(","kind":"Managed","format":"SPIRV_GLCompute"},)"
         + R"("attempts":[{"module":"m","kind":")" + kind + R"(","format":")" + format + R"(","features":[)" + features + R"(],"outcome":")"
         + outcome + R"(","detail":[)" + detail + R"(]}]})";
}

} // namespace

TEST_CASE("A skip whose device lacks the feature its image requires stays a skip") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task =
      skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""), Fp64Manifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  INFO(outcome.failReason);
  CHECK(outcome.status == TaskStatus::Skip);
  CHECK(outcome.failReason.find("SwiftShader Device lacks a feature its image requires") != std::string::npos);
}

TEST_CASE("A skip on a device that has every required feature fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "", "lacks", "\"fp64\""), PlainManifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("has every feature") != std::string::npos);
}

TEST_CASE("A skip whose selector matches no device fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""),
                                 Fp64Manifest, "spirv_glcompute@!spirv_glcompute@*NoSuchGpu*");
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("matched 0 of 1") != std::string::npos);
}

TEST_CASE("A skip with no image for the device format fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const std::string objectManifest = R"({"module":"m","images":[{"kind":"HostThreaded","format":"Object","features":["fp64"]}]})";
  const auto task =
      skippingTask(dir, record("SwiftShader Device", "HostThreaded", "Object", "\"fp64\"", "format", "\"Object\""), objectManifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("no image for Managed/SPIRV_GLCompute") != std::string::npos);
}

TEST_CASE("A skip without a runtime record fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, "polyrt: no compatible target: Selector 'x' matched none (strict)", Fp64Manifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("without a skip record") != std::string::npos);
}

TEST_CASE("A skip with no manifest from the compile step fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""), "");
  const auto outcome = detail::runTask(task, testConfig(dir.path));
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("no compile-time image manifest") != std::string::npos);
}

TEST_CASE("A module with no image for the device does not mask a justified module") {
  using namespace polyregion::polytest;
  const auto manifest =
      skip::parseManifest(R"({"module":"a","images":[{"kind":"Managed","format":"SPIRV_GLCompute","features":["int64"]}]})"
                          "\n"
                          R"({"module":"b","images":[{"kind":"HostThreaded","format":"Source","features":[]}]})");
  const auto attempts =
      R"(polyrt-skip: {"device":{"platform":"Vulkan","name":"SwiftShader Device","kind":"Managed","format":"SPIRV_GLCompute"},"attempts":[)"
      R"({"module":"a","kind":"Managed","format":"SPIRV_GLCompute","features":["int64"],"outcome":"lacks","detail":["int64"]},)"
      R"({"module":"b","kind":"HostThreaded","format":"Source","features":[],"outcome":"format","detail":["Source"]}]})";
  const auto inventory = skip::parseInventory(SwiftShaderInventory);
  CHECK(skip::judge(Selector, inventory, manifest, attempts).justified);
  const auto onlyB = skip::parseManifest(R"({"module":"b","images":[{"kind":"HostThreaded","format":"Source","features":[]}]})");
  const auto bAttempt =
      R"(polyrt-skip: {"device":{"platform":"Vulkan","name":"SwiftShader Device","kind":"Managed","format":"SPIRV_GLCompute"},"attempts":[)"
      R"({"module":"b","kind":"HostThreaded","format":"Source","features":[],"outcome":"format","detail":["Source"]}]})";
  CHECK_FALSE(skip::judge(Selector, inventory, onlyB, bAttempt).justified);
}

TEST_CASE("A compile step exiting 77 fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  script(dir, "compile.sh", "exit 77");
  const Task task{Mode::Offload, "check_x.cpp", "c", {}, dir.file("out"), std::nullopt, {{dir.file("compile.sh"), {}, {}}}, {}};
  CHECK(detail::compileTask(task, testConfig(dir.path)).status == TaskStatus::Fail);
}
#endif
