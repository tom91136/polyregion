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
