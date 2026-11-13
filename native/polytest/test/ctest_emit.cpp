#include "polytest/ctest_emit.hpp"

#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "polyregion/io.hpp"

TEST_CASE("CTest variants lock their prefixed shared output") {
  using namespace polyregion::polytest;

  const std::string path = "polytest-ctest-emitter-test.cmake";
  struct Remove {
    const std::string &path;
    ~Remove() { std::remove(path.c_str()); }
  } remove{path};

  const std::vector<CtestEntry> entries{{"shared", "codegen", {{"", {}}, {"runtime", "MODE=runtime"}}}};
  emitCtestFragment(path, "polycpp", "runner", {}, {}, entries);
  const auto cpp = polyregion::read_string(path);
  CHECK(cpp.find("FIXTURES_REQUIRED \"fix-polycpp-shared\" RESOURCE_LOCK \"fix-polycpp-shared\"") != std::string::npos);
  CHECK(cpp.find("FIXTURES_SETUP \"fix-polycpp-shared\"") != std::string::npos);
  CHECK(cpp.find("FIXTURES_CLEANUP \"fix-polycpp-shared\"") != std::string::npos);

  emitCtestFragment(path, "polyfc", "runner", {}, {}, entries);
  const auto fc = polyregion::read_string(path);
  CHECK(fc.find("fix-polyfc-shared") != std::string::npos);
  CHECK(fc.find("fix-polycpp-shared") == std::string::npos);
}

TEST_CASE("CTest counts exit 77 of every test that runs a case as a skip") {
  using namespace polyregion::polytest;

  const std::string path = "polytest-ctest-skip-test.cmake";
  struct Remove {
    const std::string &path;
    ~Remove() { std::remove(path.c_str()); }
  } remove{path};

  const std::vector<CtestEntry> entries{{"plain", "", {}}, {"labelled", "codegen", {}}, {"shared", "", {{"", {}}, {"usm", "MODE=off"}}}};
  emitCtestFragment(path, "p", "runner", {}, {}, entries);
  const auto fragment = polyregion::read_string(path);
  for (const auto *name : {"p-plain", "p-labelled", "compile-p-shared", "p-shared", "p-shared-usm"})
    CHECK(fragment.find(fmt::format("set_tests_properties(\"{}\" PROPERTIES SKIP_RETURN_CODE 77", name)) != std::string::npos);
  CHECK(fragment.find("set_tests_properties(\"cleanup-p-shared\" PROPERTIES SKIP_RETURN_CODE") == std::string::npos);
}

TEST_CASE("CTest entries name their source file and carry a scaled timeout") {
  using namespace polyregion::polytest;

  const std::string path = "polytest-ctest-timeout-test.cmake";
  struct Remove {
    const std::string &path;
    ~Remove() { std::remove(path.c_str()); }
  } remove{path};

  const std::vector<CtestEntry> entries{{"plain", "", {}, "check_a.cpp", DefaultTimeoutSecs},
                                        {"shared", "", {{"", {}}, {"usm", "MODE=off"}}, "check_b.cpp", 700}};
  emitCtestFragment(path, "p", "runner", {}, {}, entries);
  const auto fragment = polyregion::read_string(path);
  CHECK(fragment.find("add_test(\"p-plain\" \"runner\" --run-task \"plain\" --source \"check_a.cpp\")") != std::string::npos);
  for (const auto *flag : {"--compile-task", "--run-only-task", "--cleanup-task"})
    CHECK(fragment.find(fmt::format("{} \"shared\" --source \"check_b.cpp\")", flag)) != std::string::npos);
  CHECK(fragment.find(fmt::format("math(EXPR polytest_timeout_{0} \"{0} * ${{polytest_timeout_scale}}\")", DefaultTimeoutSecs))
        != std::string::npos);
  CHECK(fragment.find("math(EXPR polytest_timeout_700 ") != std::string::npos);
  CHECK(fragment.find("set_tests_properties(\"p-shared-usm\" PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_REQUIRED \"fix-p-shared\" "
                      "RESOURCE_LOCK \"fix-p-shared\" TIMEOUT ${polytest_timeout_700}")
        != std::string::npos);
  CHECK(fragment.find(fmt::format("\"cleanup-p-shared\" PROPERTIES FIXTURES_CLEANUP \"fix-p-shared\" TIMEOUT ${{polytest_timeout_{}}}",
                                  DefaultTimeoutSecs))
        != std::string::npos);
}

TEST_CASE("CTest runs of a shared compile are named and labelled per variant") {
  using namespace polyregion::polytest;

  const std::string path = "polytest-ctest-variant-test.cmake";
  struct Remove {
    const std::string &path;
    ~Remove() { std::remove(path.c_str()); }
  } remove{path};

  const std::vector<CtestEntry> entries{
      {"compile-id",
       "codegen;shard",
       {{"", {}, "run-a", "codegen;a;shard"}, {"usm", "MODE=off", "run-a", "codegen;a;shard"}, {"", {}, "run-b"}}}};
  emitCtestFragment(path, "p", "runner", {}, {}, entries);
  const auto fragment = polyregion::read_string(path);
  CHECK(fragment.find("add_test(\"compile-p-compile-id\" \"runner\" --compile-task \"compile-id\")") != std::string::npos);
  CHECK(fragment.find("add_test(\"p-run-a\" \"runner\" --run-only-task \"run-a\")") != std::string::npos);
  CHECK(fragment.find("add_test(\"p-run-a-usm\" \"runner\" --run-only-task \"run-a\")") != std::string::npos);
  CHECK(fragment.find("add_test(\"p-run-b\" \"runner\" --run-only-task \"run-b\")") != std::string::npos);
  CHECK(fragment.find("FIXTURES_SETUP \"fix-p-compile-id\" TIMEOUT ${polytest_timeout_150} LABELS \"codegen;shard\"") != std::string::npos);
  CHECK(fragment.find("RESOURCE_LOCK \"fix-p-compile-id\" TIMEOUT ${polytest_timeout_150} LABELS \"codegen;a;shard\"")
        != std::string::npos);
  CHECK(
      fragment.find("\"p-run-b\" PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_REQUIRED \"fix-p-compile-id\" RESOURCE_LOCK \"fix-p-compile-id\" "
                    "TIMEOUT ${polytest_timeout_150} LABELS \"codegen;shard\"")
      != std::string::npos);
}
