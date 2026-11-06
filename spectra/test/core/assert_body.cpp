#pragma region case: assert_body
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 1 .core_transform_reduce_w4)
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0 1 7 -1 9 101 1 5050

#pragma region case: assert_body-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0 1 7 -1 9 101 1 5050

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

#include "polystl/polystl.h"

void bump(std::vector<int> &xs) {
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) {
    assert(x >= 0);
    x += 1;
  });
}

int total(const std::vector<int> &xs) {
  return std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) {
    assert(x >= 0);
    return x;
  });
}

int main() {
  using polyregion::polystl::details::lastAssert;
  std::vector<int> xs(100);
  std::iota(xs.begin(), xs.end(), 1);
  bump(xs);
  const bool clean = lastAssert().raised;
  std::iota(xs.begin(), xs.end(), 1);
  xs[6] = -1;
  bump(xs);
  const bool raised = lastAssert().raised;
  std::printf("%d %d %d %d %d %d", clean ? 1 : 0, raised ? 1 : 0, xs[5], xs[6], xs[7], xs[99]);
  xs[6] = -5;
  total(xs);
  const bool reduced = lastAssert().raised;
  std::iota(xs.begin(), xs.end(), 1);
  std::printf(" %d %d", reduced ? 1 : 0, total(xs));
  return 0;
}
