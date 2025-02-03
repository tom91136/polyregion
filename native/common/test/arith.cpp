#include <cmath>

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "polyregion/concurrency_utils.hpp"

TEMPLATE_TEST_CASE("split-static", "[template]", uint32_t, uint64_t, int32_t, int64_t) {
  using T = TestType;
  auto lowerBound = GENERATE(range<T>(std::is_unsigned_v<T> ? 0 : -8, 8));
  auto upperBound = GENERATE(range<T>(std::is_unsigned_v<T> ? 0 : -8, 8));
  auto step = GENERATE(range<T>(std::is_unsigned_v<T> ? 0 : -8, 8));
  if (step == 0 || (step > 0 && lowerBound >= upperBound) || (step < 0 && lowerBound <= upperBound)) return;
  DYNAMIC_SECTION("for(int i = " << lowerBound << "; i < " << upperBound << "; i+=" << step) {
    INFO("LB:" << lowerBound);
    INFO("UB: " << upperBound);
    INFO("Step:" << step);
    std::vector<T> expected;
    for (T i = lowerBound; i < upperBound; i += step)
      expected.push_back(i);
    auto actualTripCount = static_cast<size_t>(polyregion::concurrency_utils::tripCountExclusive(lowerBound, upperBound, step));
    REQUIRE(expected.size() == actualTripCount);

    {
      std::vector<T> actual;
      for (size_t i = 0; i < actualTripCount; ++i) {
        actual.emplace_back(polyregion::concurrency_utils::zeroOffset<T>(i, lowerBound, step));
      }
      CHECK(expected == actual);
    }

    auto N = GENERATE(range(1, 8));
    DYNAMIC_SECTION("N=" << N) {
      std::vector<T> actual;
      using ST = std::make_signed_t<T>;
      auto [begins, ends] = polyregion::concurrency_utils::splitStaticExclusive<ST>(0, actualTripCount, N);
      const ST *begin = begins.data();
      const ST *end = ends.data();
      for (size_t n = 0; n < begins.size(); ++n) {
        for (ST i = begin[n]; i < end[n]; ++i) {
          actual.emplace_back(polyregion::concurrency_utils::zeroOffset<T>(i, lowerBound, step));
        }
      }
      CHECK(expected == actual);
    }
  }
}