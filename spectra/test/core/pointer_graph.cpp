#pragma region case: pointer_graph
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 3 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 4 .core_transform_reduce_w4)
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 531996 1045 2124891 2062 3 4124

#pragma region case: pointer_graph-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 531996 1045 2124891 2062 3 4124

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

struct Grid {
  std::vector<int> a, b;
};

int main() {
  const int n = 1031;
  std::vector<int> ids(n), xs(n, 1), out(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { out[i] = xs[i] + i; });
  const int references =
      std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return out[i] * xs[i]; });

  Grid g{std::vector<int>(n, 0), std::vector<int>(n)};
  std::iota(g.b.begin(), g.b.end(), 0);
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { g.a[i] = g.b[i] * 2 + static_cast<int>(g.b.size()); });
  const int nested =
      std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return g.a[i]; });

  std::vector<std::vector<int>> rows(4, std::vector<int>(n, 2));
  const int deep =
      std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return rows[i % 4][i]; });

  std::vector<int> ys(n, 1), ws(n, 1);
  int *p = ws.data();
  std::for_each(std::execution::par_unseq, ys.begin(), ys.end(), [p](int &y) { y += p[0] + 1; });
  const int pointer =
      std::transform_reduce(std::execution::par_unseq, ys.begin(), ys.end(), 0, std::plus<>(), [p](int y) { return y + p[1]; });

  std::printf("%d %d %d %d %d %d %d", out[7], references, g.a[7], nested, deep, ys[7], pointer);
  return 0;
}
