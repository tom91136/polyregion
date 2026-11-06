#pragma region case: transform_reduce
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 9 polyregion.core.transform_reduce bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 3 .core_transform_reduce_w4)
#pragma region stderr-count: 2 .core_transform_reduce_w8)
#pragma region stderr-count: 1 .core_transform_reduce_w12)
#pragma region stderr-count: 1 .core_transform_reduce_w32)
#pragma region stderr-count: 1 .core_transform_reduce_w40)
#pragma region stderr-count: 1 .core_transform_reduce_w64)
#pragma region stderr-count: 1 polyregion.core.transform_reduce uses the inline body: no implementation fits
#pragma region stderr-count: 1 of 72 bytes
#pragma region stderr-count: 1 uses the inline body
#pragma region do: {output}
#pragma region requires: 3094 3094.0 1547.0 1102755921920 2 | 3094.0 1031.0 9.0 | 3094.0 1031.0 13450.0 442.0 | 3094.0 1031.0 13450.0 0 9 | 3094.0 1031.0 | 3094.0 1031.0

#pragma region case: transform_reduce-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 3094 3094.0 1547.0 1102755921920 2 | 3094.0 1031.0 9.0 | 3094.0 1031.0 13450.0 442.0 | 3094.0 1031.0 13450.0 0 9 | 3094.0 1031.0 | 3094.0 1031.0

#include <algorithm>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

struct V12 {
  float sum, count, top;
};

struct V32 {
  double a, b, c, d;
  V32 operator+(const V32 &r) const { return {a + r.a, b + r.b, c + r.c, d + r.d}; }
};

struct V40 {
  double sum, count, square;
  int lo, hi;
  double last;
};

struct V64 {
  double a, b, c, d, e, f, g, h;
  V64 operator+(const V64 &r) const { return {a + r.a, b + r.b, c + r.c, d + r.d, e + r.e, f + r.f, g + r.g, h + r.h}; }
};

struct V72 {
  double a, b, c, d, e, f, g, h, i;
  V72 operator+(const V72 &r) const { return {a + r.a, b + r.b, c + r.c, d + r.d, e + r.e, f + r.f, g + r.g, h + r.h, i + r.i}; }
};

template <class V> V spread(const std::vector<int> &xs) {
  return std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), V{}, std::plus<>(), [](int x) {
    V o{};
    o.a = x;
    o.h = 1;
    return o;
  });
}

int main() {
  const int n = 1031;
  std::vector<int> xs(n);
  for (int i = 0; i < n; ++i)
    xs[i] = i % 7;
  xs[500] = 9;

  const int is = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x; });
  const float fs =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0.f, std::plus<>(), [](int x) { return float(x); });
  const double ds =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0.0, std::plus<>(), [](int x) { return x * 0.5; });
  const long long ls = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 1LL << 40, std::plus<>(),
                                             [](int x) { return static_cast<long long>(x) << 20; });
  const int mn = std::transform_reduce(
      std::execution::par_unseq, xs.begin(), xs.end(), 1 << 30, [](int a, int b) { return std::min(a, b); }, [](int x) { return x + 2; });
  const auto v12 = std::transform_reduce(
      std::execution::par_unseq, xs.begin(), xs.end(), V12{0, 0, 0},
      [](V12 l, V12 r) { return V12{l.sum + r.sum, l.count + r.count, std::max(l.top, r.top)}; },
      [](int x) { return V12{float(x), 1, float(x)}; });
  const auto v32 = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), V32{}, std::plus<>(),
                                         [](int x) { return V32{double(x), 1, double(x * x), x > 3 ? 1.0 : 0.0}; });
  const auto v40 = std::transform_reduce(
      std::execution::par_unseq, xs.begin(), xs.end(), V40{0, 0, 0, 1 << 30, -(1 << 30), 0},
      [](V40 l, V40 r) {
        return V40{l.sum + r.sum, l.count + r.count, l.square + r.square, std::min(l.lo, r.lo), std::max(l.hi, r.hi), 0};
      },
      [](int x) { return V40{double(x), 1, double(x) * x, x, x, 0}; });
  const auto v64 = spread<V64>(xs);
  const auto v72 = spread<V72>(xs);

  std::printf("%d %.1f %.1f %lld %d | %.1f %.1f %.1f | %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %d %d | %.1f %.1f | %.1f %.1f", is, fs, ds, ls,
              mn, v12.sum, v12.count, v12.top, v32.a, v32.b, v32.c, v32.d, v40.sum, v40.count, v40.square, v40.lo, v40.hi, v64.a, v64.h,
              v72.a, v72.h);
  return 0;
}
