#include "polytest/driver.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
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

TEST_CASE("Artifact removal clears the task directory only") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto taskDir = dir.file("out_a_offload");
  llvm::sys::fs::create_directories(taskDir + "/out.packages/nested");
  dir.touch("out_a_offload/out");
  dir.touch("out_a_offload/out.o");
  dir.touch("out_a_offload/out.stdout");
  dir.touch("out_a_offload/out.packages/nested/blob");
  dir.touch("out_a_offload_other");
  dir.touch("module-cache");

  detail::removeArtifacts(taskDir);

  CHECK_FALSE(llvm::sys::fs::exists(taskDir));
  CHECK(llvm::sys::fs::exists(dir.file("out_a_offload_other")));
  CHECK(llvm::sys::fs::exists(dir.file("module-cache")));
}

TEST_CASE("Artifact removal through a relative directory clears it") {
  using namespace polyregion::polytest;
  TempDir dir;
  llvm::sys::fs::create_directories(dir.file("out_b_offload/out.packages/nested"));
  dir.touch("out_b_offload/out");
  dir.touch("out_b_offload/out.pdb");
  dir.touch("out_b_offload/out.packages/nested/blob");
  llvm::SmallString<256> cwd;
  REQUIRE_FALSE(llvm::sys::fs::current_path(cwd));
  REQUIRE_FALSE(llvm::sys::fs::set_current_path(dir.path));

  detail::removeArtifacts("out_b_offload");

  REQUIRE_FALSE(llvm::sys::fs::set_current_path(cwd));
  CHECK_FALSE(llvm::sys::fs::exists(dir.file("out_b_offload")));
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
    CHECK(t.dir.starts_with("out_one_"));
    CHECK(t.output == t.dir + "/" + OutputName);
  }
}

TEST_CASE("A case limited to listed targets expands for those targets only") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native;cuda@sm_89;arm@native");
  std::ofstream(dir.file("check_two.cpp"))
      << "#pragma region case: anywhere\n#pragma region do: echo {output}\n"
         "#pragma region case: cpu\n#pragma region targets: native arm\n#pragma region do: echo {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_two.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  const auto archesOf = [&](const std::string &name) {
    using namespace aspartame;
    return tasks                                                       //
           ^ filter([&](const auto &t) { return t.caseName == name; }) //
           ^ flat_map([](const auto &t) {
               return t.variables ^ collect([](const auto &k, const auto &v) { return k == "arch" ? std::optional{v} : std::nullopt; });
             }) //
           ^ sort();
  };
  CHECK(archesOf("anywhere") == std::vector<std::string>{"arm@native", "cuda@sm_89", "host@native"});
  CHECK(archesOf("cpu") == std::vector<std::string>{"arm@native", "host@native"});
}

TEST_CASE("Passthrough runs once however many targets are listed") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native;cuda@sm_89;arm@native");
  std::ofstream(dir.file("check_three.cpp")) << "#pragma region case: any\n#pragma region do: echo {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_three.cpp")};
  const auto tasks = enumerateTasks(cfg, true, true, {});
  CHECK((tasks ^ count([](const auto &t) { return t.mode == Mode::Offload; })) == 3);
  const auto passthrough = tasks ^ filter([](const auto &t) { return t.mode == Mode::Passthrough; });
  REQUIRE(passthrough.size() == 1);
  CHECK(detail::archFor(passthrough[0], cfg) == "host@native");
}

TEST_CASE("A source name limits enumeration to that file") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_a.cpp")) << "#pragma region case: a\n#pragma region do: echo {output}\n";
  std::ofstream(dir.file("check_b.cpp")) << "#pragma region case: b\n#pragma region timeout: 700\n#pragma region do: echo {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_a.cpp"), dir.file("check_b.cpp")};
  CHECK(enumerateTasks(cfg, true, false, {}).size() == 2);
  const auto tasks = enumerateTasks(cfg, true, false, {}, "check_b.cpp");
  REQUIRE(tasks.size() == 1);
  CHECK(tasks[0].caseName == "b");
  CHECK(tasks[0].timeoutSecs == 700U);
  CHECK(enumerateTasks(cfg, true, false, {}, "check_c.cpp").empty());
}

TEST_CASE("A directive the case grammar cannot place is rejected") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_misplaced.cpp")) << "#pragma region case: a\n#pragma region do: echo {output}\n"
                                                    "#pragma region compile-fails: late\n#pragma region do: echo again\n"
                                                    "#pragma region case: b\n#pragma region do: echo {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_misplaced.cpp")};
  CHECK_THROWS_AS(enumerateTasks(cfg, true, false, {}), std::invalid_argument);
}

TEST_CASE("A directive name inside a message stays part of that message") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_names.cpp")) << "#pragma region case: a\n#pragma region compile-fails: requires do: x\n"
                                                "#pragma region do: echo {output}\n"
                                                "#pragma region case: b\n#pragma region do: echo {output}\n"
                                                "#pragma region do: echo {output}\n#pragma region fails-with: requires stderr-count: 1\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_names.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  REQUIRE(tasks.size() == 2);
  CHECK(tasks[0].compileFailure == "requires do: x");
  REQUIRE(tasks[1].variants.size() == 1);
  REQUIRE(tasks[1].variants[0].runs.size() == 1);
  CHECK(tasks[1].variants[0].runs[0].expect.empty());
  CHECK(tasks[1].variants[0].runs[0].failure == "requires stderr-count: 1");
}

TEST_CASE("Requires and stderr counts interleave after a step") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_interleave.cpp"))
      << "#pragma region case: a\n#pragma region do: echo {output}\n#pragma region do: echo {output}\n"
         "#pragma region stderr-count: 1 alpha\n#pragma region requires: one\n"
         "#pragma region stderr-count: 0 beta\n#pragma region requires: two\n"
         "#pragma region do: echo {output}\n#pragma region stderr-count: 2 gamma\n"
         "#pragma region requires: three\n#pragma region fails-with: nope\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_interleave.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].variants.size() == 1);
  const auto &runs = tasks[0].variants[0].runs;
  REQUIRE(runs.size() == 2);
  CHECK(runs[0].expect == std::vector<TestCase::Run::Expect>{{std::nullopt, "one"}, {std::nullopt, "two"}});
  CHECK(runs[0].stderrCounts == std::vector<TestCase::Run::StderrCount>{{1, "alpha"}, {0, "beta"}});
  CHECK(!runs[0].failure);
  CHECK(runs[1].expect == std::vector<TestCase::Run::Expect>{{std::nullopt, "three"}});
  CHECK(runs[1].stderrCounts == std::vector<TestCase::Run::StderrCount>{{2, "gamma"}});
  CHECK(runs[1].failure == std::optional<std::string>{"nope"});
}

TEST_CASE("Requires before stderr counts after a step still parse") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_order.cpp"))
      << "#pragma region case: a\n#pragma region do: echo {output}\n#pragma region do: echo {output}\n"
         "#pragma region requires: one\n#pragma region stderr-count: 1 alpha\n"
         "#pragma region do: echo {output}\n#pragma region requires: two\n"
         "#pragma region stderr-count: 0 beta\n#pragma region fails-with: nope\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_order.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].variants.size() == 1);
  const auto &runs = tasks[0].variants[0].runs;
  REQUIRE(runs.size() == 2);
  CHECK(runs[0].expect == std::vector<TestCase::Run::Expect>{{std::nullopt, "one"}});
  CHECK(runs[0].stderrCounts == std::vector<TestCase::Run::StderrCount>{{1, "alpha"}});
  CHECK(!runs[0].failure);
  CHECK(runs[1].expect == std::vector<TestCase::Run::Expect>{{std::nullopt, "two"}});
  CHECK(runs[1].stderrCounts == std::vector<TestCase::Run::StderrCount>{{0, "beta"}});
  CHECK(runs[1].failure == std::optional<std::string>{"nope"});
}

TEST_CASE("A requires after fails-with is rejected") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_late.cpp")) << "#pragma region case: a\n#pragma region do: echo {output}\n"
                                               "#pragma region do: echo {output}\n#pragma region fails-with: nope\n"
                                               "#pragma region requires: late\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_late.cpp")};
  CHECK_THROWS_AS(enumerateTasks(cfg, true, false, {}), std::invalid_argument);
}

TEST_CASE("An expected failure on the first step is rejected") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets, "host@native");
  std::ofstream(dir.file("check_first.cpp")) << "#pragma region case: a\n#pragma region do: echo {output}\n"
                                                "#pragma region fails-with: nope\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_first.cpp")};
  CHECK_THROWS_AS(enumerateTasks(cfg, true, false, {}), std::invalid_argument);
}

TEST_CASE("Cache and library locations reach the child environment") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv cache(polyregion::env::PolyregionCacheDir, "/polytest/cache");
  polyregion::env::ScopedEnv xdg("XDG_CACHE_HOME", "/polytest/xdg");
  polyregion::env::ScopedEnv pocl("POCL_CACHE_DIR", "/polytest/pocl");
  const auto cfg = testConfig(dir.path);
  const Task task{Mode::Offload, "check_x.cpp", "c", {}, "out_x", "out_x/out", std::nullopt, {}, {}, std::nullopt};
  const auto envs = detail::baseEnvs(task, cfg, false, "");
  const auto has = [&](const std::string &kv) { return std::find(envs.begin(), envs.end(), kv) != envs.end(); };
  CHECK(has("POLYREGION_CACHE_DIR=/polytest/cache"));
  CHECK(has("XDG_CACHE_HOME=/polytest/xdg"));
  CHECK(has("POCL_CACHE_DIR=/polytest/pocl"));
}

TEST_CASE("Targets that differ in device alone share one compile") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets,
                                     "spirv_glcompute@!spirv_glcompute@*llvmpipe*;spirv_glcompute@!spirv_glcompute@*SwiftShader*;"
                                     "host@native!host@*;cuda@sm_35!cuda@*");
  std::ofstream(dir.file("check_group.cpp")) << "#pragma region case: g\n#pragma region do: build {arch} {output}\n"
                                                "#pragma region do: run {arch} {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_group.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  REQUIRE(tasks.size() == 3);

  const auto named = [&](const std::string &arch) {
    return tasks ^ filter([&](const auto &t) { return detail::archFor(t, cfg) == arch; });
  };
  const auto merged = named("spirv_glcompute@");
  REQUIRE(merged.size() == 1);
  CHECK(merged[0].runs.size() == 1);
  CHECK(merged[0].runs[0].command == "build spirv_glcompute@ " + merged[0].output);
  REQUIRE(merged[0].variants.size() == 2);
  CHECK(merged[0].variants[0].id != merged[0].variants[1].id);
  CHECK((merged[0].variants ^ map([](const auto &v) { return v.target; }) ^ sort())
        == std::vector<std::string>{"spirv_glcompute@!spirv_glcompute@*SwiftShader*", "spirv_glcompute@!spirv_glcompute@*llvmpipe*"});
  for (const auto &v : merged[0].variants) {
    REQUIRE(v.runs.size() == 1);
    CHECK(v.runs[0].command == "run " + v.target + " " + merged[0].output);
  }
  {
    polyregion::env::ScopedEnv solo(polyregion::env::PolyregionTestTargets, "spirv_glcompute@!spirv_glcompute@*llvmpipe*");
    const auto alone = enumerateTasks(cfg, true, false, {});
    REQUIRE(alone.size() == 1);
    CHECK(alone[0].id() == merged[0].id());
  }

  const auto host = named("host@native");
  REQUIRE(host.size() == 1);
  REQUIRE(host[0].variants.size() == 1);
  CHECK(host[0].variants[0].target == "host@native!host@*");
  CHECK(host[0].runs[0].command == "build host@native " + host[0].output);

  const auto cuda = named("cuda@sm_35");
  REQUIRE(cuda.size() == 1);
  CHECK((cuda[0].variants ^ map([](const auto &v) { return v.suffix; })) == std::vector<std::string>{"", "runtime", "usm"});
  CHECK((cuda[0].variants ^ map([](const auto &v) { return v.id; }) ^ distinct()).size() == 1);
}

TEST_CASE("Merged tasks list one run per device and emit one compile") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets,
                                     "spirv_glcompute@!spirv_glcompute@*llvmpipe*;spirv_glcompute@!spirv_glcompute@*SwiftShader*;"
                                     "host@native!host@*");
  std::ofstream(dir.file("check_emit.cpp")) << "#pragma region case: e\n#pragma region do: build {arch} {output}\n"
                                               "#pragma region do: run {arch} {output}\n";
  auto cfg = testConfig(dir.path);
  cfg.testFiles = {dir.file("check_emit.cpp")};
  const auto tasks = enumerateTasks(cfg, true, false, {});
  const auto fragment = dir.file("fragment.cmake");
  emitCtest(tasks, cfg, fragment, "p", "runner", {}, {});
  const auto text = polyregion::read_string(fragment);
  const auto count = [&](const std::string &needle) {
    std::size_t n = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
      ++n;
    return n;
  };
  CHECK(count("--compile-task") == 1);
  CHECK(count("--run-only-task") == 2);
  CHECK(count("--run-task") == 1);
  const auto slice = [&](const std::string &shard, const Task &t) { return sliceLabel(shard, bucketOf(sourceOf(t))); };
  const auto merged = tasks ^ find([&](const auto &t) { return detail::archFor(t, cfg) == "spirv_glcompute@"; });
  const auto host = tasks ^ find([&](const auto &t) { return detail::archFor(t, cfg) == "host@native!host@*"; });
  REQUIRE((merged && host));
  const auto glcompute = slice("spirv_glcompute_40", *merged);
  CHECK(count("LABELS \"codegen;spirv_glcompute_40;" + glcompute + "\"") == 2);
  CHECK(count("LABELS \"codegen;spirv_glcompute_40_21spirv_glcompute_40_2allvmpipe_2a;spirv_glcompute_40;" + glcompute + "\"") == 1);
  CHECK(count("LABELS \"codegen;spirv_glcompute_40_21spirv_glcompute_40_2aSwiftShader_2a;spirv_glcompute_40;" + glcompute + "\"") == 1);
  CHECK(count("LABELS \"codegen;host_40native_21host_40_2a;host_40native;" + slice("host_40native", *host) + "\"") == 1);
}

TEST_CASE("A bucket is stable and every task of a source file shares it") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  CHECK(bucketOf("check_a.cpp") == 53);
  CHECK(bucketOf("") == bucketOf(""));
  CHECK(sliceLabel("host_40native", 7) == "host_40native~07");

  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets,
                                     "spirv_glcompute@!spirv_glcompute@*llvmpipe*;spirv_glcompute@!spirv_glcompute@*SwiftShader*;"
                                     "cuda@sm_35!cuda@*");
  auto cfg = testConfig(dir.path);
  for (int f = 0; f < 12; ++f) {
    const auto file = dir.file("check_many" + std::to_string(f) + ".cpp");
    std::ofstream src(file);
    for (int i = 0; i < 4; ++i)
      src << "#pragma region case: c" << i << "\n#pragma region do: build {arch} {output}\n#pragma region do: run {arch} {output}\n";
    cfg.testFiles.push_back(file);
  }
  const auto tasks = enumerateTasks(cfg, true, false, {});
  const auto fragment = dir.file("fragment.cmake");
  emitCtest(tasks, cfg, fragment, "p", "runner", {}, {});
  const auto text = polyregion::read_string(fragment);

  std::set<unsigned> used;
  for (const auto &t : tasks) {
    const auto shard = ctestSafe(compileTargetOf(detail::archFor(t, cfg)));
    const auto label = sliceLabel(shard, bucketOf(sourceOf(t)));
    used.insert(bucketOf(sourceOf(t)));
    std::size_t labelled = 0;
    for (auto at = text.find(label + "\""); at != std::string::npos; at = text.find(label + "\"", at + 1))
      ++labelled;
    const auto sharing =
        tasks ^ count([&](const auto &o) {
          return ctestSafe(compileTargetOf(detail::archFor(o, cfg))) == shard && bucketOf(sourceOf(o)) == bucketOf(sourceOf(t));
        });
    const auto entries = (t.variants.size() == 1 ? 1 : t.variants.size() + 2);
    CHECK(labelled == sharing * entries);
  }
  CHECK(used.size() > 1);
  for (const auto &t : tasks)
    for (const auto &o : tasks)
      if (sourceOf(t) == sourceOf(o)) CHECK(bucketOf(sourceOf(t)) == bucketOf(sourceOf(o)));
}

TEST_CASE("Buckets are assigned to slices by greedy longest-processing-time") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  CHECK(sliceCount(0) == 1);
  CHECK(sliceCount(SliceRuns) == 1);
  CHECK(sliceCount(SliceRuns + 1) == 2);
  CHECK(sliceCount(1000 * SliceRuns) == ShardBuckets);

  CHECK(sliceOf("host_40native", 0, 1, {}).labels == "^host_40native$");
  CHECK(sliceOf("host_40native", 0, 1, {}).id == "host_40native");
  CHECK(sliceOf("host_40native", 1, 3, {1, 4}).id == "host_40native.2of3");
  CHECK(sliceOf("host_40native", 1, 3, {1, 4}).labels == "^host_40native~(01|04)$");

  using Buckets = std::vector<std::vector<unsigned>>;
  CHECK(assignBuckets({5, 3, 3, 2, 2, 1}, 2) == Buckets{{0, 3, 5}, {1, 2, 4}});
  CHECK(assignBuckets({4, 4, 4, 4}, 3) == Buckets{{0, 3}, {1}, {2}});
  CHECK(assignBuckets({0, 0, 7, 0, 7}, 2) == Buckets{{0, 1, 2, 3}, {4}});
  CHECK(assignBuckets({9, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 3) == Buckets{{0}, {1, 3, 5, 7, 9}, {2, 4, 6, 8}});

  std::vector<std::size_t> weights(ShardBuckets);
  for (unsigned b = 0; b < ShardBuckets; ++b)
    weights[b] = (b * 37U) % 11U;
  for (const unsigned n : {2U, 3U, 5U, 8U, 64U}) {
    const auto slices = assignBuckets(weights, n);
    CHECK(assignBuckets(weights, n) == slices);
    std::vector<unsigned> seen;
    for (const auto &bs : slices)
      seen.insert(seen.end(), bs.begin(), bs.end());
    std::sort(seen.begin(), seen.end());
    std::vector<unsigned> all(ShardBuckets);
    for (unsigned b = 0; b < ShardBuckets; ++b)
      all[b] = b;
    CHECK(seen == all);
  }
}

TEST_CASE("A shard slices by its run entries and keeps profile order") {
  using namespace polyregion::polytest;
  using namespace aspartame;
  TempDir dir;
  polyregion::env::ScopedEnv targets(polyregion::env::PolyregionTestTargets,
                                     "host@native!host@*;cuda@sm_35!cuda@*;spirv_glcompute@!spirv_glcompute@*llvmpipe*;"
                                     "spirv_glcompute@!spirv_glcompute@*SwiftShader*;c11@native!c11@*");
  auto cfg = testConfig(dir.path);
  for (int f = 0; f < 40; ++f) {
    const auto file = dir.file("check_weight" + std::to_string(f) + ".cpp");
    std::ofstream src(file);
    for (int i = 0; i < 20; ++i)
      src << "#pragma region case: c" << i << "\n#pragma region do: build {arch} {output}\n#pragma region do: run {arch} {output}\n";
    cfg.testFiles.push_back(file);
  }
  const auto tasks = enumerateTasks(cfg, true, false, {});
  const auto slices = listSlices(loadTestTargets(cfg.profileDir), bucketWeights(tasks, cfg));
  const auto idsOf = [&](const std::string &shard) {
    return slices ^ filter([&](const auto &s) { return s.shard == shard; }) ^ map([](const auto &s) { return s.id; });
  };
  CHECK((slices ^ map([](const auto &s) { return s.shard; }) ^ distinct())
        == std::vector<std::string>{"host_40native", "cuda_40sm_35", "spirv_glcompute_40", "c11_40native"});
  CHECK(idsOf("host_40native") == std::vector<std::string>{"host_40native"});
  CHECK(idsOf("c11_40native") == std::vector<std::string>{"c11_40native"});
  CHECK(idsOf("cuda_40sm_35") == std::vector<std::string>{"cuda_40sm_35.1of2", "cuda_40sm_35.2of2"});
  CHECK(idsOf("spirv_glcompute_40") == std::vector<std::string>{"spirv_glcompute_40"});

  const auto cuda = slices ^ filter([](const auto &s) { return s.shard == "cuda_40sm_35"; });
  REQUIRE(cuda.size() == 2);
  const auto again = listSlices(loadTestTargets(cfg.profileDir), bucketWeights(tasks, cfg));
  REQUIRE(again.size() == slices.size());
  for (std::size_t i = 0; i < slices.size(); ++i)
    CHECK(again[i].labels == slices[i].labels);
  const auto bucketsOf = [](const std::string &labels) {
    const auto open = labels.find('(');
    return labels.substr(open + 1, labels.size() - open - 3) ^ split('|');
  };
  auto both = bucketsOf(cuda[0].labels);
  both ^= concat(bucketsOf(cuda[1].labels));
  std::sort(both.begin(), both.end());
  CHECK(both.size() == ShardBuckets);
  CHECK((both ^ distinct()).size() == ShardBuckets);

  const auto weights = bucketWeights(tasks, cfg);
  BucketWeights summed;
  addBucketWeights(summed, formatBucketWeights(weights));
  CHECK(formatBucketWeights(summed) == formatBucketWeights(weights));
  addBucketWeights(summed, formatBucketWeights(weights));
  const auto doubled = listSlices(loadTestTargets(cfg.profileDir), summed);
  CHECK((doubled ^ count([](const auto &s) { return s.shard == "cuda_40sm_35"; })) == 3);
  CHECK((doubled ^ count([](const auto &s) { return s.shard == "host_40native"; })) == 1);
}

#ifndef _WIN32
TEST_CASE("A run selects its own device in the child environment") {
  using namespace polyregion::polytest;
  TempDir dir;
  dir.touch("rusticl.icd");
  dir.touch("pocl.icd");
  polyregion::env::ScopedEnv vk("VK_DRIVER_FILES", "/icd/lvp_icd.json:/icd/swiftshader.json");
  polyregion::env::ScopedEnv ocl("OCL_ICD_VENDORS", dir.path.c_str());
  polyregion::env::ScopedEnv rusticl("RUSTICL_ENABLE", "llvmpipe");
  const auto cfg = testConfig(dir.path);
  const Task task{Mode::Offload, "check_x.cpp", "c", {{"arch", "spirv_glcompute@"}}, "out_x", "out_x/out", std::nullopt, {}, {},
                  std::nullopt};
  const auto has = [](const std::vector<std::string> &envs, const std::string &kv) {
    return std::find(envs.begin(), envs.end(), kv) != envs.end();
  };
  const auto hasKey = [](const std::vector<std::string> &envs, const std::string &key) {
    return std::any_of(envs.begin(), envs.end(), [&](const auto &kv) { return kv.rfind(key + "=", 0) == 0; });
  };

  const std::string swift = "spirv_glcompute@!spirv_glcompute@*SwiftShader*", lvp = "spirv_glcompute@!spirv_glcompute@*llvmpipe*";
  const auto swiftEnv = detail::baseEnvs(task, cfg, false, swift);
  CHECK(has(swiftEnv, "VK_DRIVER_FILES=/icd/swiftshader.json"));
  CHECK(has(swiftEnv, "POLYRT_PLATFORM=" + swift));
  const auto lvpEnv = detail::baseEnvs(task, cfg, false, lvp);
  CHECK(has(lvpEnv, "VK_DRIVER_FILES=/icd/lvp_icd.json"));
  CHECK(has(lvpEnv, "POLYRT_PLATFORM=" + lvp));
  const auto compileEnv = detail::baseEnvs(task, cfg, false, detail::compileTargetFor(task, cfg));
  CHECK(has(compileEnv, "POLYRT_PLATFORM=spirv_glcompute@"));
  CHECK_FALSE(hasKey(compileEnv, "VK_DRIVER_FILES"));

  const auto rusticlEnv = detail::baseEnvs(task, cfg, false, "opencl1_1@!opencl1_1@rusticl:*source*");
  CHECK(has(rusticlEnv, "OCL_ICD_VENDORS=" + dir.path + "/rusticl.icd"));
  CHECK(has(rusticlEnv, "GALLIUM_DRIVER=llvmpipe"));
  CHECK(has(rusticlEnv, "POLYINVOKE_DISABLE_SVM=1"));
  const auto poclEnv = detail::baseEnvs(task, cfg, false, "opencl1_1@!opencl1_1@*Portable*source*");
  CHECK(has(poclEnv, "OCL_ICD_VENDORS=" + dir.path + "/pocl.icd"));
  CHECK_FALSE(hasKey(poclEnv, "GALLIUM_DRIVER"));
  CHECK(has(poclEnv, "POLYINVOKE_DISABLE_SVM=1"));
}

#endif

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
                                        const std::string &selector = Selector, const std::string &inventory = SwiftShaderInventory) {
  using namespace polyregion::polytest;
  script(dir, "polyinfo", std::string("cat <<'EOF'\n") + inventory + "\nEOF");
  script(dir, "skip.sh", "cat <<'EOF' >&2\n" + record + "\nEOF\nexit 77");
  std::ofstream(dir.file("out.images")) << manifest;
  return {Mode::Offload,
          "check_x.cpp",
          "c",
          {{"arch", selector}},
          dir.path,
          dir.file("out"),
          std::nullopt,
          {{"compile", {}, {}}},
          {RunVariant{"c", selector, "", {}, {{dir.file("skip.sh"), {}, {}}}, "c"}},
          std::nullopt};
}

constexpr const char *Fp64Manifest = R"({"module":"m","images":[{"kind":"Managed","format":"SPIRV_GLCompute","features":["fp64"]}]})";
constexpr const char *PlainManifest = R"({"module":"m","images":[{"kind":"Managed","format":"SPIRV_GLCompute","features":[]}]})";

std::string record(const std::string &device, const std::string &kind, const std::string &format, const std::string &features,
                   const std::string &outcome, const std::string &detail) {
  return R"(polyrt-skip: {"device":{"platform":"Vulkan","name":")" + device + R"(","kind":"Managed","format":"SPIRV_GLCompute"},)"
         + R"("attempts":[{"module":"m","kind":")" + kind + R"(","format":")" + format + R"(","features":[)" + features + R"(],"outcome":")"
         + outcome + R"(","detail":[)" + detail + R"(]}]})";
}

constexpr const char *CudaSelector = "cuda@sm_35!cuda@*";
constexpr const char *PtxManifest = R"({"module":"m","images":[{"kind":"Managed","format":"PTX","features":[]}]})";
constexpr const char *PagingRecord =
    R"(polyrt-skip: {"device":{"platform":"CUDA","name":"Ocelot PTX Emulator","kind":"Managed","format":"PTX"},"attempts":[)"
    R"({"module":"m","kind":"Managed","format":"PTX","features":[],"outcome":"lacks","detail":["paging:system"]}]})";

std::string cudaInventory(const std::string &paging) {
  return R"({"backend":"CUDA","platform":"CUDA","kind":"Managed","devices":[{"name":"Ocelot PTX Emulator","format":"PTX",)"
         R"("features":["cuda","nvidia","sm_35","fp64","int64",")"
         + paging + R"("]}]})";
}

} // namespace

TEST_CASE("A skip whose device lacks the feature its image requires stays a skip") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task =
      skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""), Fp64Manifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  INFO(outcome.failReason);
  CHECK(outcome.status == TaskStatus::Skip);
  CHECK(outcome.failReason.find("SwiftShader Device lacks a feature its image or run requires") != std::string::npos);
}

TEST_CASE("A usm skip on a device without system paging stays a skip") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv mirror(polyregion::env::PolyrtMirror, "off");
  const auto task = skippingTask(dir, PagingRecord, PtxManifest, CudaSelector, cudaInventory("paging:None"));
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  INFO(outcome.failReason);
  CHECK(outcome.status == TaskStatus::Skip);
  CHECK(outcome.failReason.find("Ocelot PTX Emulator lacks a feature its image or run requires") != std::string::npos);
}

TEST_CASE("A usm skip on a device with system paging fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv mirror(polyregion::env::PolyrtMirror, "off");
  const auto task = skippingTask(dir, PagingRecord, PtxManifest, CudaSelector, cudaInventory("paging:System"));
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("has every feature required for image") != std::string::npos);
}

TEST_CASE("A paging skip outside a usm run fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  polyregion::env::ScopedEnv mirror(polyregion::env::PolyrtMirror, "runtime");
  const auto task = skippingTask(dir, PagingRecord, PtxManifest, CudaSelector, cudaInventory("paging:None"));
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("has every feature required for image") != std::string::npos);
}

TEST_CASE("A skip on a device that has every required feature fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "", "lacks", "\"fp64\""), PlainManifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("has every feature") != std::string::npos);
}

TEST_CASE("A skip whose selector matches no device fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""),
                                 Fp64Manifest, "spirv_glcompute@!spirv_glcompute@*NoSuchGpu*");
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("matched 0 of 1") != std::string::npos);
}

TEST_CASE("A skip with no image for the device format fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const std::string objectManifest = R"({"module":"m","images":[{"kind":"HostThreaded","format":"Object","features":["fp64"]}]})";
  const auto task =
      skippingTask(dir, record("SwiftShader Device", "HostThreaded", "Object", "\"fp64\"", "format", "\"Object\""), objectManifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("no image for Managed/SPIRV_GLCompute") != std::string::npos);
}

TEST_CASE("A skip without a runtime record fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, "polyrt: no compatible target: Selector 'x' matched none (strict)", Fp64Manifest);
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
  CHECK(outcome.status == TaskStatus::Fail);
  CHECK(outcome.failReason.find("without a skip record") != std::string::npos);
}

TEST_CASE("A skip with no manifest from the compile step fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""), "");
  const auto outcome = detail::runTask(task, testConfig(dir.path), task.variants[0]);
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
  CHECK(skip::judge(Selector, inventory, manifest, attempts, {}).justified);
  const auto onlyB = skip::parseManifest(R"({"module":"b","images":[{"kind":"HostThreaded","format":"Source","features":[]}]})");
  const auto bAttempt =
      R"(polyrt-skip: {"device":{"platform":"Vulkan","name":"SwiftShader Device","kind":"Managed","format":"SPIRV_GLCompute"},"attempts":[)"
      R"({"module":"b","kind":"HostThreaded","format":"Source","features":[],"outcome":"format","detail":["Source"]}]})";
  CHECK_FALSE(skip::judge(Selector, inventory, onlyB, bAttempt, {}).justified);
}

TEST_CASE("A compile step exiting 77 fails") {
  using namespace polyregion::polytest;
  TempDir dir;
  script(dir, "compile.sh", "exit 77");
  const Task task{
      Mode::Offload, "check_x.cpp", "c", {}, dir.file("task"), dir.file("task/out"), std::nullopt, {{dir.file("compile.sh"), {}, {}}}, {},
      std::nullopt};
  CHECK(detail::compileTask(task, testConfig(dir.path)).status == TaskStatus::Fail);
}

TEST_CASE("The repro check rebuilds the output and compares it") {
  using namespace polyregion::polytest;
  TempDir dir;
  const auto compile = [&](const std::string &body) {
    script(dir, "compile.sh",
           "n=$(cat " + dir.file("count") + " 2>/dev/null || echo 0); echo $((n + 1)) > " + dir.file("count") + "\n" + body);
    llvm::sys::fs::remove(dir.file("count"));
    const auto command = dir.file("compile.sh") + " -fstdpar -o " + dir.file("task/out") + " check_x.cpp";
    return detail::compileTask(Task{Mode::Offload,
                                    "check_x.cpp",
                                    "c",
                                    {},
                                    dir.file("task"),
                                    dir.file("task/out"),
                                    std::nullopt,
                                    {{command, {}, {}}},
                                    {},
                                    std::nullopt},
                               testConfig(dir.path));
  };
  SECTION("identical outputs pass and keep the output") {
    const auto o = compile(R"sh(echo same > "$3")sh");
    CHECK(o.status == TaskStatus::Pass);
    CHECK(polyregion::read_string(dir.file("task/out")) == "same\n");
    CHECK_FALSE(llvm::sys::fs::exists(dir.file("task/out.repro.o")));
  }
  SECTION("differing outputs fail") {
    const auto o = compile(R"sh(echo "$n" > "$3")sh");
    CHECK(o.status == TaskStatus::Fail);
    CHECK(o.failReason.find("non-reproducible build") != std::string::npos);
  }
  SECTION("a failing rebuild reports its own stderr") {
    const auto o = compile(R"sh([ "$n" -gt 0 ] && { echo 'repro diagnostic' >&2; exit 3; }
: > "$3")sh");
    CHECK(o.status == TaskStatus::Fail);
    CHECK(o.failReason.find("repro compile failed") != std::string::npos);
    CHECK(o.stderrCapture == "repro diagnostic\n");
  }
}

TEST_CASE("A later step expected to fail passes only with its diagnostic") {
  using namespace polyregion::polytest;
  TempDir dir;
  script(dir, "fails.sh", "echo 'include not found' >&2; exit 1");
  script(dir, "passes.sh", "exit 0");
  const auto task = [&](const std::string &step, const std::string &message) {
    return Task{Mode::Offload,
                "check_x.cpp",
                "c",
                {},
                dir.path,
                dir.file("out"),
                std::nullopt,
                {{"compile", {}, {}}},
                {RunVariant{"c", "", "", {}, {{dir.file(step), {}, {}, message}}, "c"}},
                std::nullopt};
  };
  const auto run = [&](const std::string &step, const std::string &message) {
    const auto t = task(step, message);
    return detail::runTask(t, testConfig(dir.path), t.variants[0]);
  };
  CHECK(run("fails.sh", "include not found").status == TaskStatus::Pass);
  const auto wrong = run("fails.sh", "other diagnostic");
  CHECK(wrong.status == TaskStatus::Fail);
  CHECK(wrong.failReason.find("without expected diagnostic") != std::string::npos);
  const auto succeeded = run("passes.sh", "include not found");
  CHECK(succeeded.status == TaskStatus::Fail);
  CHECK(succeeded.failReason.find("unexpectedly succeeded") != std::string::npos);
}

TEST_CASE("A skip is judged against the device the run selected") {
  using namespace polyregion::polytest;
  TempDir dir;
  const std::string twoDevices =
      R"json({"backend":"Vulkan","platform":"Vulkan","kind":"Managed","devices":[)json"
      R"json({"name":"llvmpipe (LLVM 21)","format":"SPIRV_GLCompute","features":["vulkan","spirv_glcompute"]},)json"
      R"json({"name":"SwiftShader Device","format":"SPIRV_GLCompute","features":["vulkan","spirv_glcompute","unknown"]}]})json";
  auto task = skippingTask(dir, record("SwiftShader Device", "Managed", "SPIRV_GLCompute", "\"fp64\"", "lacks", "\"fp64\""), Fp64Manifest,
                           "spirv_glcompute@", twoDevices);
  const auto cfg = testConfig(dir.path);

  task.variants[0].target = Selector;
  const auto onSwiftShader = detail::runTask(task, cfg, task.variants[0]);
  INFO(onSwiftShader.failReason);
  CHECK(onSwiftShader.status == TaskStatus::Skip);

  task.variants[0].target = "spirv_glcompute@!spirv_glcompute@*llvmpipe*";
  const auto onLlvmpipe = detail::runTask(task, cfg, task.variants[0]);
  CHECK(onLlvmpipe.status == TaskStatus::Fail);
  CHECK(onLlvmpipe.failReason.find("runtime used 'SwiftShader Device' but the selector resolves to 'llvmpipe (LLVM 21)'")
        != std::string::npos);

  task.variants[0].target = "spirv_glcompute@";
  CHECK(detail::runTask(task, cfg, task.variants[0]).status == TaskStatus::Fail);
}
TEST_CASE("A preamble spec drops macros, input and output, and refuses steps that are not plain stdpar compiles") {
  using namespace polyregion::polytest;
  TempDir dir;
  std::ofstream(dir.file("check_x.cpp")) << "#include <cstdio>\n#include \"test_utils.h\"\n";
  std::ofstream(dir.file("check_y.cpp")) << "#include <cstdio>\n";
  const auto expand = [](const std::string &t) {
    return t == "{flags}" ? std::string("-O3 -fstdpar") : t == "{libm}" ? std::string("-lm") : t;
  };
  const auto spec = [&](const std::string &file, const std::string &tmpl) {
    return pch::specFor("test_utils.h", dir.file(file), tmpl, expand);
  };
  const auto x = spec("check_x.cpp", "A=1 polycpp {flags} -DSIZE=2 -USTRICT -o {output} {input} {libm}");
  REQUIRE(x);
  CHECK(x->header == dir.file("test_utils.h"));
  CHECK(x->command == std::vector<std::string>{"A=1", "polycpp", "-O3", "-fstdpar"});
  CHECK(x->macros == std::vector<std::string>{"SIZE", "STRICT"});
  CHECK(spec("check_y.cpp", "polycpp {flags} -o {output} {input}")->header.empty());
  CHECK_FALSE(spec("check_x.cpp", "polycpp {flags} -x cuda -o {output} {input}"));
  CHECK_FALSE(spec("check_x.cpp", "polycpp {flags} -c -o {output}.o {input}"));
  CHECK_FALSE(spec("check_x.cpp", "polycpp {flags} -fstdpar-library-path={output}.packages -o {output} {input}"));
  CHECK_FALSE(spec("check_x.cpp", "polycpp -O3 -o {output} {input}"));
}

TEST_CASE("The preamble PCH is built once per key, rebuilt when a dependency changes, and bypassed when rejected") {
  using namespace polyregion::polytest;
  TempDir dir;
  struct Cwd {
    llvm::SmallString<256> previous;
    explicit Cwd(const std::string &to) {
      llvm::sys::fs::current_path(previous);
      llvm::sys::fs::set_current_path(to);
    }
    ~Cwd() { llvm::sys::fs::set_current_path(previous); }
  } cwd(dir.path);
  dir.touch("test_utils.h");
  script(dir, "polycpp",
         "echo \"$@\" >> " + dir.file("log") + R"sh(
precompile=0; include=
while [ $# -gt 0 ]; do
  case "$1" in -x) precompile=1;; -o) out=$2; shift;; -dependency-file) dep=$3; shift 2;; -include-pch) include=$2; shift;; esac
  shift
done
if [ $precompile = 1 ]; then echo pch > "$out"; echo "$out: $(dirname "$out")/../test_utils.h" > "$dep"; exit 0; fi
if [ -n "$include" ] && [ -e )sh"
             + dir.file("reject") + R"sh( ]; then echo "error: PCH was rejected" >&2; exit 1; fi
echo bin > "$out")sh");
  auto task = Task{Mode::Offload,
                   "check_x.cpp",
                   "c",
                   {},
                   dir.file("task"),
                   dir.file("task/out"),
                   std::nullopt,
                   {{dir.file("polycpp") + " -fstdpar -o " + dir.file("task/out") + " check_x.cpp", {}, {}}},
                   {},
                   std::nullopt};
  task.pchSpec = pch::Spec{dir.file("test_utils.h"), {dir.file("polycpp"), "-fstdpar"}, {}};
  const auto cfg = testConfig(dir.path);
  const auto invocations = [&] {
    return polyregion::read_string(dir.file("log")) ^ split('\n') ^ filter([](const auto &l) { return !l.empty(); });
  };
  const auto usedPch = [&](const std::string &line) { return line.find("-include-pch") != std::string::npos; };

  CHECK(detail::compileTask(task, cfg).status == TaskStatus::Pass);
  auto log = invocations();
  REQUIRE(log.size() == 3);
  CHECK(log[0].find("-x c++-header") != std::string::npos);
  CHECK(usedPch(log[1]));
  CHECK(log[1] == log[2]);

  log = invocations();
  CHECK(detail::compileTask(task, cfg).status == TaskStatus::Pass);
  CHECK(invocations().size() == log.size() + 2);
  CHECK(usedPch(invocations().back()));

  std::ofstream(dir.file("test_utils.h"), std::ios::app) << "// edited\n";
  log = invocations();
  CHECK(detail::compileTask(task, cfg).status == TaskStatus::Pass);
  CHECK(invocations().size() == log.size() + 3);
  CHECK(invocations()[log.size()].find("-x c++-header") != std::string::npos);

  dir.touch("reject");
  log = invocations();
  CHECK(detail::compileTask(task, cfg).status == TaskStatus::Pass);
  const auto after = invocations();
  REQUIRE(after.size() > log.size());
  CHECK_FALSE(usedPch(after.back()));
  CHECK(std::any_of(after.begin() + static_cast<std::ptrdiff_t>(log.size()), after.end(), usedPch));
}
#endif
