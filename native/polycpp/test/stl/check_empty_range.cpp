#pragma region case: empty-range
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7 0

#include <algorithm>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> xs;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = 1; });
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 7, std::plus<>(), [](int x) { return x; });
  std::printf("%d %zu", sum, xs.size());
  return 0;
}
