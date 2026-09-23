#include "polytest/profile.hpp"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Profile environment separates generic inherited-variable directives") {
  using namespace polyregion::polytest;

  const auto profile = parseProfileEnvironment({
      "STATIC=profile",
      "POLYTEST_FORWARD_ENV=RPC_PORT; RPC_CLIENT;;RPC_PORT",
      "POLYTEST_FORWARD_ENV=:LD_PRELOAD",
      "OTHER=value",
  });

  CHECK(profile.assignments == std::vector<std::string>{"STATIC=profile", "OTHER=value"});
  CHECK(profile.forwardedNames == std::vector<std::string>{"RPC_PORT", "RPC_CLIENT", "LD_PRELOAD"});

  polyregion::env::put("POLYTEST_TEST_INHERITED", "dynamic", true);
  CHECK(materialiseProfileEnvironment({"STATIC=profile", "POLYTEST_FORWARD_ENV=:POLYTEST_TEST_INHERITED"})
        == std::vector<std::string>{"STATIC=profile", "POLYTEST_TEST_INHERITED=dynamic"});
}
