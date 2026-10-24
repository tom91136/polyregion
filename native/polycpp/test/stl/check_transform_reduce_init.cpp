#pragma region case: transform-reduce-init
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 2031 1100592709632 516.0 42

#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> xs(1031, 1);
  const int ints = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 1000, std::plus<>(), [](int x) { return x; });
  const long long wide = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 1LL << 40, std::plus<>(),
                                               [](int x) { return static_cast<long long>(x) << 20; });
  const double reals =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0.5, std::plus<>(), [](int x) { return x * 0.5; });
  const int single =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.begin() + 1, 41, std::plus<>(), [](int x) { return x; });
  std::printf("%d %lld %.1f %d", ints, wide, reals, single);
  return 0;
}
