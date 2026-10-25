#pragma region case: core-fallback
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 3 600

#pragma region case: core-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 3 600

#pragma region case: core-package-without-device
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region do: POLYRT_PLATFORM=none POLYRT_HOST_FALLBACK=1 {output}
#pragma region requires: 3 600

#pragma region case: core-package-std-captures
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSTD_CAPTURES -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to polyregion.core
#pragma region do: {output}
#pragma region requires: 13 1300

#pragma region case: core-package-assert
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DASSERT -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to polyregion.core
#pragma region do: {output}
#pragma region requires: 1 7 -1 9 101

#pragma region case: core-package-specialisations
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSPECIALISATIONS -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 60 6100 400.0

#pragma region case: core-device-package-reference-vectors
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DREFERENCE_VECTORS -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 5050

#pragma region case: core-device-package-nested-vectors
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DNESTED_VECTORS -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 114 19900

#pragma region case: core-device-package-pointer
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPOINTER_CAPTURE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 3 400

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

#if defined(REFERENCE_VECTORS)
int main() {
  const int n = 100;
  std::vector<int> ids(n), xs(n, 1), out(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { out[i] = xs[i] + i; });
  const int sum =
      std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return out[i] * xs[i]; });
  std::printf("%d %d", out[7], sum);
  return 0;
}
#elif defined(NESTED_VECTORS)
struct Grid {
  std::vector<int> a, b;
};

int main() {
  const int n = 100;
  Grid g{std::vector<int>(n, 0), std::vector<int>(n)};
  std::iota(g.b.begin(), g.b.end(), 0);
  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { g.a[i] = g.b[i] * 2 + static_cast<int>(g.b.size()); });
  const int sum = std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return g.a[i]; });
  std::printf("%d %d", g.a[7], sum);
  return 0;
}
#elif defined(POINTER_CAPTURE)
int main() {
  const int n = 100;
  std::vector<int> xs(n, 1), ws(n, 1);
  int *p = ws.data();
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [p](int &x) { x += p[0] + 1; });
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [p](int x) { return x + p[1]; });
  std::printf("%d %d", xs[7], sum);
  return 0;
}
#elif defined(STD_CAPTURES)
int main() {
  const std::string word = "polyregion";
  const std::array<int, 4> weights{1, 2, 3, 4};
  std::vector<int> xs(100, 0);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(),
                [word, weights](int &x) { x = static_cast<int>(word.size()) + weights[2]; });
  int sum = 0;
  for (const int x : xs)
    sum += x;
  std::printf("%d %d", xs[7], sum);
  return 0;
}
#elif defined(ASSERT)
  #include "polyregion/enums.h"
  #include "polystl/polystl.h"

int checked(int x) {
  if (x < 0) __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "core");
  return x + 1;
}

int main() {
  std::vector<int> xs(100);
  std::iota(xs.begin(), xs.end(), 1);
  xs[6] = -1;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = checked(x); });
  const bool raised = polyregion::polystl::details::lastAssert().raised;
  std::printf("%d %d %d %d %d", raised ? 1 : 0, xs[5], xs[6], xs[7], xs[99]);
  return 0;
}
#elif defined(SPECIALISATIONS)
template <class T> void scale(std::vector<T> &xs, T factor) {
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [factor](T &x) { x *= factor; });
}

int main() {
  std::vector<int> xs(100, 1);
  std::vector<double> ys(100, 0.5);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x += 1; });
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x *= 3; });
  scale(xs, 2);
  scale(xs, 5);
  scale(ys, 4.0);
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x + 1; });
  const double total =
      std::transform_reduce(std::execution::par_unseq, ys.begin(), ys.end(), 0.0, std::plus<>(), [](double y) { return y * 2; });
  std::printf("%d %d %.1f", xs[7], sum, total);
  return 0;
}
#else
int main() {
  std::vector<int> xs(100, 1);
  const int bias = 2;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [bias](int &x) { x += bias; });
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x * 2; });
  std::printf("%d %d", xs[7], sum);
  return 0;
}
#endif
