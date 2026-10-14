#pragma region case: core-fallback
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3 600

#pragma region case: core-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3 600

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> xs(100, 1);
  const int bias = 2;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [bias](int &x) { x += bias; });
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x * 2; });
  std::printf("%d %d", xs[7], sum);
  return 0;
}
