#pragma region case: for_each
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 8 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 10 4130 2.0 384.0 2136232.0 4135

#pragma region case: for_each-env
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH={spectra_packages} POLYFRONT_LIBRARY_CAPABILITIES={spectra_capability} polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 8 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 10 4130 2.0 384.0 2136232.0 4135

#pragma region case: for_each-multi-target
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -fstdpar-arch={spectra_arch};host@native -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 8 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 10 4130 2.0 384.0 2136232.0 4135

#pragma region case: for_each-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 10 4130 2.0 384.0 2136232.0 4135

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

template <class T> void scale(std::vector<T> &xs, T factor) {
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [factor](T &x) { x *= factor; });
}

int main() {
  const int n = 1031;
  std::vector<int> xs(n);
  std::iota(xs.begin(), xs.end(), 0);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x += 1; });
  const int bias = 3;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [bias](int &x) { x = x * 2 + bias; });

  std::vector<double> ys(n, 0.5);
  scale(xs, 2);
  scale(ys, 4.0);

  std::array<float, 256> zs{};
  std::for_each(std::execution::par_unseq, zs.begin(), zs.end(), [](float &z) { z = 1.5f; });

  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<double> out(n);
  double *o = out.data();
  const int *x = xs.data();
  const double *y = ys.data();
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [o, x, y](int i) { o[i] = x[i] + y[i]; });

  std::vector<int> pairs(n);
  int *p = pairs.data();
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [p, x, n](int i) { p[i] = x[i] + x[n - 1 - i]; });

  const int shift = 5;
  std::for_each(std::execution::par_unseq, pairs.begin(), pairs.end(), [&](int &v) { v -= shift; });

  float zsum = 0;
  for (const float z : zs)
    zsum += z;
  double osum = 0;
  for (const double v : out)
    osum += v;
  std::printf("%d %d %.1f %.1f %.1f %d", xs[0], xs[n - 1], ys[0], zsum, osum, pairs[5]);
  return 0;
}
