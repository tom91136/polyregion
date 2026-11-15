#include <catch2/catch_test_macros.hpp>

#include "polytest/skip_proof.hpp"

using namespace polyregion::polytest::skip;

namespace {

const Device Fp32Only{.name = "Lavapipe", .format = {}, .features = {"spirv_kernel"}};
const Device Fp64Capable{.name = "Big GPU", .format = {}, .features = {"fp64", "spirv_kernel"}};
const Need NeedsFp64{"fp64", true};
const Need WantsNoSpirv{"spirv_kernel", false, "source variant"};

} // namespace

TEST_CASE("Devices that all lack a needed feature justify a skip") {
  const auto verdict = judgeNeeds({Fp32Only}, {NeedsFp64});
  CHECK(verdict.justified);
  CHECK(verdict.reason.find("Lavapipe lacks fp64") != std::string::npos);
}

TEST_CASE("A device holding a feature the test excludes justifies a skip and names why") {
  const auto verdict = judgeNeeds({Fp64Capable}, {WantsNoSpirv});
  CHECK(verdict.justified);
  CHECK(verdict.reason.find("Big GPU has spirv_kernel (source variant)") != std::string::npos);
}

TEST_CASE("Every candidate is explained") {
  const auto verdict = judgeNeeds({Fp32Only, Fp64Capable}, {NeedsFp64, WantsNoSpirv});
  CHECK(verdict.justified);
  CHECK(verdict.reason.find("Lavapipe lacks fp64") != std::string::npos);
  CHECK(verdict.reason.find("Big GPU has spirv_kernel") != std::string::npos);
}

TEST_CASE("A device that satisfies every need fails the skip") {
  const auto verdict = judgeNeeds({Fp32Only, Fp64Capable}, {NeedsFp64});
  CHECK_FALSE(verdict.justified);
  CHECK(verdict.reason.find("Big GPU") != std::string::npos);
}

TEST_CASE("A skip with no candidate device fails") { CHECK_FALSE(judgeNeeds({}, {NeedsFp64}).justified); }

TEST_CASE("A skip that names no need fails") { CHECK_FALSE(judgeNeeds({Fp32Only}, {}).justified); }

TEST_CASE("Feature names compare ignoring case") {
  CHECK(judgeNeeds({{.name = "d", .format = {}, .features = {"FP64"}}}, {{"fp64", false}}).justified);
}
