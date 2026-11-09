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

#pragma region case: non-random-access-rejected
#pragma region offload-only
#pragma region compile-fails: -fstdpar-require-offload rejects
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-require-offload -DCHECK_KIND=2 -o {output} {input}

#pragma region case: mixed-iterator-copy-rejected
#pragma region offload-only
#pragma region compile-fails: -fstdpar-require-offload rejects
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-require-offload -DCHECK_KIND=3 -o {output} {input}

#pragma region case: non-random-access-results
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11 23 208 3164 4886649

#ifndef CHECK_KIND
  #error "CHECK_KIND undefined"
#endif

#include <algorithm>
#include <cstdio>
#include <execution>
#include <functional>
#include <list>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> xs(1000);
  std::iota(xs.rbegin(), xs.rend(), 0);
#if CHECK_KIND == 0
  std::nth_element(std::execution::par_unseq, xs.begin(), xs.begin() + 500, xs.end());
  const auto median = xs[500];
  std::partial_sort(std::execution::par_unseq, xs.begin(), xs.begin() + 1, xs.end());
  const auto sum = std::accumulate(xs.begin(), xs.end(), 0);
  std::printf("%d %d %d", xs.front(), median, sum);
#elif CHECK_KIND == 1
  std::sort(std::execution::seq, xs.begin(), xs.end());
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = 2; });
  const auto sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x; });
  std::printf("%d %d", xs.front() - 2, sum);
#elif CHECK_KIND == 2
  std::list<int> ys(xs.begin(), xs.end());
  std::for_each(std::execution::par_unseq, ys.begin(), ys.end(), [](int &x) { x = 2; });
  std::printf("%d", ys.front());
#elif CHECK_KIND == 3
  std::list<int> ys(xs.size());
  std::copy(std::execution::par_unseq, xs.begin(), xs.end(), ys.begin());
  std::printf("%d", ys.front());
#elif CHECK_KIND == 4
  const std::vector<int> src{1, 2, 3, 4};
  std::list<int> ys(src.size());
  std::copy(std::execution::par_unseq, src.begin(), src.end(), ys.begin());
  std::for_each(std::execution::par_unseq, ys.begin(), ys.end(), [](int &x) { x *= 10; });
  std::for_each_n(std::execution::par_unseq, ys.begin(), 2, [](int &x) { x += 1; });
  std::list<int> zs(ys.size());
  std::transform(std::execution::par_unseq, ys.begin(), ys.end(), zs.begin(), [](int x) { return x + 1; });
  std::list<int> ws(ys.size());
  std::transform(std::execution::par_unseq, ys.begin(), ys.end(), zs.begin(), ws.begin(), std::plus<>());
  const auto sum = std::transform_reduce(std::execution::par_unseq, ws.begin(), ws.end(), 0, std::plus<>(), [](int x) { return x; });
  const auto dot = std::transform_reduce(std::execution::par_unseq, ys.begin(), ys.end(), zs.begin(), 0);
  const auto product =
      std::transform_reduce(std::execution::par_unseq, ys.begin(), ys.end(), zs.begin(), 1, std::multiplies<>(), std::plus<>());
  std::printf("%d %d %d %d %d", ys.front(), ws.front(), sum, dot, product);
#endif
  return 0;
}
