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

TEST_CASE("The compile part of a target is what precedes the device selector") {
  using namespace polyregion::polytest;

  CHECK(compileTargetOf("spirv_glcompute@!spirv_glcompute@*llvmpipe*") == "spirv_glcompute@");
  CHECK(compileTargetOf("spirv64_kernel@1_1!spirv64_kernel@rusticl:*SPIR-V*") == "spirv64_kernel@1_1");
  CHECK(compileTargetOf("host@native") == "host@native");
}
