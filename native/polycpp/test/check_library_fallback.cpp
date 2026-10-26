#pragma region case: fallback-without-package
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 141

#pragma region case: fallback-with-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to foo (bar.implementation.increment)
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: fallback-remote-without-device
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DREMOTE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to foo (bar.implementation.remote_increment)
#pragma region do: {output}
#pragma region requires: 42
#pragma region do: POLYRT_PLATFORM=none {output}
#pragma region requires: 141

#pragma region case: fallback-without-exceptions
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fno-exceptions -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 uses the inline body: -fno-exceptions
#pragma region do: {output}
#pragma region requires: 141

#pragma region case: fallback-without-capability
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: POLYFRONT_LIBRARY_CAPABILITIES= polycpp {polycpp_defaults} {polycpp_stdpar} -DCAPABLE -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 uses the inline body: capability not satisfied (capabilities: none)
#pragma region do: {output}
#pragma region requires: 141

#pragma region case: fallback-with-capability
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCAPABLE -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo -o {output} {input}
#pragma region stderr-count: 1 bound to foo (bar.implementation.capable_increment)
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: fallback-core-width
#pragma region offload-only
#pragma region do: {package_fixture} --sized-core {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCORE_WIDTH -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region stderr-count: 1 bound to polyregion.core (polyregion.core.host.transform_reduce)
#pragma region stderr-count: 2 uses the inline body: no implementation fits
#pragma region stderr-count: 1 uses the inline body: no implementation fits `F64` of 8 bytes (sizes: 4)
#pragma region do: {output}
#pragma region requires: 5050 2525.0 15 30 45

#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

namespace foo {
template <class T>
[[clang::annotate("polyregion_interface:foo:bar.increment")]] [[clang::annotate("polyregion_interface_fallback")]] inline T increment(T x) {
  return x + 100;
}

template <class T>
[[clang::annotate("polyregion_interface:foo:bar.capable_increment")]] [[clang::annotate("polyregion_interface_fallback")]] inline T
capable_increment(T x) {
  return x + 100;
}

template <class T>
[[clang::annotate("polyregion_interface:foo:bar.remote_increment")]] [[clang::annotate("polyregion_interface_fallback")]] inline T
remote_increment(T x) {
  return x + 100;
}
} // namespace foo

#if defined(CORE_WIDTH)
struct Triple {
  int a, b, c;
};

int main() {
  std::vector<int> xs(100);
  std::iota(xs.begin(), xs.end(), 1);
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x; });
  const double half =
      std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0.0, std::plus<>(), [](int x) { return x * 0.5; });
  const auto triple = std::transform_reduce(
      std::execution::par_unseq, xs.begin(), xs.begin() + 5, Triple{0, 0, 0},
      [](Triple l, Triple r) { return Triple{l.a + r.a, l.b + r.b, l.c + r.c}; }, [](int x) { return Triple{x, 2 * x, 3 * x}; });
  std::printf("%d %.1f %d %d %d", sum, half, triple.a, triple.b, triple.c);
}
#elif defined(REMOTE)
int main() { std::printf("%d", foo::remote_increment(41)); }
#elif defined(CAPABLE)
int main() { std::printf("%d", foo::capable_increment(41)); }
#else
int main() { std::printf("%d", foo::increment(41)); }
#endif
