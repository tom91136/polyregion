#pragma region case: host-algorithms
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 500 499500

#pragma region case: host-algorithms-rejected
#pragma region offload-only
#pragma region compile-fails: -fstdpar-require-offload rejects
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-require-offload -DCHECK_KIND=0 -o {output} {input}

#pragma region case: offloaded-and-sequenced-accepted
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-require-offload -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 2000

#ifndef CHECK_KIND
  #error "CHECK_KIND undefined"
#endif

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> xs(1000);
  std::iota(xs.rbegin(), xs.rend(), 0);
#if CHECK_KIND == 0
  std::sort(std::execution::par_unseq, xs.begin(), xs.end());
  const auto evens = std::count_if(std::execution::par_unseq, xs.begin(), xs.end(), [](int x) { return x % 2 == 0; });
  const auto sum = std::reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0);
  std::printf("%d %ld %d", xs.front(), static_cast<long>(evens), sum);
#elif CHECK_KIND == 1
  std::sort(std::execution::seq, xs.begin(), xs.end());
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = 2; });
  const auto sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x; });
  std::printf("%d %d", xs.front() - 2, sum);
#endif
  return 0;
}
