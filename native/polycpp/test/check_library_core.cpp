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

#pragma region case: core-device-package-recursion
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DRECURSION -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 55 94

#pragma region case: core-device-package-pointer-reference
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPOINTER_REFERENCE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 9 5149

#pragma region case: core-device-package-iterator-copy
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DITERATOR_COPY -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 5050

#pragma region case: core-device-package-iterator-copy-base
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DITERATOR_COPY_BASE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 5050

#pragma region case: core-device-package-iterator-copy-multi-target
#pragma region offload-only
#pragma region do: {package_fixture} --device-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-arch={polycpp_arch};x86_64@native -DITERATOR_COPY -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 2 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 8 5050

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
#elif defined(RECURSION)
static int fib(int n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2);
}

int main() {
  std::vector<int> xs(16);
  std::iota(xs.begin(), xs.end(), 0);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x = fib(x); });
  const int sum =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return fib(x % 10); });
  std::printf("%d %d", xs[10], sum);
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
#elif defined(POINTER_REFERENCE)
int main() {
  const int n = 100;
  const std::vector<std::string> words{"a", "bb", "ccc"};
  std::vector<int> ids(n), out(n, 0);
  std::iota(ids.begin(), ids.end(), 0);
  int *p = out.data();
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { p[i] = i + static_cast<int>(words[i % 3].size()); });
  const int sum = std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return p[i]; });
  std::printf("%d %d", out[7], sum);
  return 0;
}
#elif defined(ITERATOR_COPY) || defined(ITERATOR_COPY_BASE)
struct Empty {};

  #if defined(ITERATOR_COPY_BASE)
struct ConstIter : Empty {
  int *_Ptr;
};

struct Iter : ConstIter {
  static Iter of(int *p) { return Iter{{{}, p}}; }
  #else
struct Iter {
  int *_Ptr;
  static Iter of(int *p) { return Iter{p}; }
  #endif
  using iterator_category = std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = long;
  using pointer = int *;
  using reference = int &;
  int &operator*() const { return *_Ptr; }
  Iter &operator+=(long n) {
    _Ptr += n;
    return *this;
  }
  Iter operator+(long n) const {
    Iter t = *this;
    t += n;
    return t;
  }
  long operator-(const Iter &o) const { return _Ptr - o._Ptr; }
};

struct Span {
  int *first, *last;
  int &operator[](long n) const { return first[n]; }
  Iter begin() const { return Iter::of(first); }
  Iter end() const { return Iter::of(last); }
};

int main() {
  const int n = 100;
  std::vector<int> idsData(n), xsData(n, 1), outData(n, 0);
  std::iota(idsData.begin(), idsData.end(), 0);
  const Span ids{idsData.data(), idsData.data() + n}, xs{xsData.data(), xsData.data() + n}, out{outData.data(), outData.data() + n};
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [&](int i) { out[i] = xs[i] + i; });
  const int sum =
      std::transform_reduce(std::execution::par_unseq, ids.begin(), ids.end(), 0, std::plus<>(), [&](int i) { return out[i] * xs[i]; });
  std::printf("%d %d", out[7], sum);
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
