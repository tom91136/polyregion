#pragma region case: edge_sizes
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 1 .core_transform_reduce_w8)
#pragma region stderr-count: 1 .core_transform_reduce_w16)
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0:0,-1/-1 1:1,0/0 1031:1062961,0/1030 1048576:1099511627776,0/1048575

#pragma region case: edge_sizes-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0:0,-1/-1 1:1,0/0 1031:1062961,0/1030 1048576:1099511627776,0/1048575

#include <algorithm>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

struct Range {
  long long lo, hi;
};

void run(int n) {
  std::vector<int> xs(n);
  std::iota(xs.begin(), xs.end(), 0);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = 2 * x + 1; });
  const long long sum =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0LL, std::plus<>(), [](int x) { return (long long)x; });
  const auto range = std::transform_reduce(
      std::execution::par_unseq, xs.begin(), xs.end(), Range{-1, -1},
      [](Range l, Range r) {
        return Range{l.lo < 0 ? r.lo : r.lo < 0 ? l.lo : std::min(l.lo, r.lo), std::max(l.hi, r.hi)};
      },
      [](int x) { return Range{x / 2, x / 2}; });
  std::printf("%s%d:%lld,%lld/%lld", n == 0 ? "" : " ", n, sum, range.lo, range.hi);
}

int main() {
  run(0);
  run(1);
  run(1031);
  run(1 << 20);
  return 0;
}
