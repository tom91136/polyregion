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
