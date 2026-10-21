#pragma region case: package-cub-warp-scan-closure
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -fstdpar-emit-library={output}.program.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --write-binary-marker-interface {output}.interface.polyast
#pragma region do: {polypackage_emit} package link {output}.interface.polyast {output}.packages {output}.program.polyast
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#ifdef PRODUCER

  #define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]
  #define POLYREGION_IMPLEMENTS(name) [[clang::annotate("polyregion_implements:" name)]]
  #define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]
  #define POLYREGION_CALLABLE_VARIABLE(name) [[clang::annotate("polyregion_callable_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("Element") Element {
  int storage;
};

template <class Signature> struct Callable;

template <class R, class... Args> struct POLYREGION_CALLABLE_VARIABLE("Callable0") Callable<R(Args...)> {
  Callable() = default;
  Callable(const Callable &) = default;
  Callable &operator=(const Callable &) = default;
  R operator()(Args...) const { return R{}; }
};

namespace cub {
template <class T, int Width> struct WarpScanShfl {
  unsigned member_mask;
  unsigned lane_id;
  template <class Op> T InclusiveScanStep(T input, Op op, int firstLane, int offset) const {
    T output = op(input, input);
    if (static_cast<int>(lane_id) < firstLane + offset) output = input;
    return output;
  }
};
} // namespace cub

template <class F> Element scan(Element value, F op) {
  const cub::WarpScanShfl<Element, 32> warp{~0u, 5};
  return warp.InclusiveScanStep(value, op, 0, 1);
}

POLYREGION_EXPORT_AS("bar.implementation.apply")
POLYREGION_IMPLEMENTS("bar.apply") Element apply(Element value, Callable<Element(Element, Element)> op) { return scan(value, op); }

#else

namespace foo {
template <class T, class Op> [[clang::annotate("polyregion_interface:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
} // namespace foo

int main(int argc, char **) {
  const int k = argc + 1;
  std::printf("%d", foo::apply(20, [k](int x, int y) { return x + y + k; }));
}

#endif
