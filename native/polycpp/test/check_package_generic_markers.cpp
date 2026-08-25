#pragma region case: generic-package-stub-markers
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -DUSE_CUB -fstdpar-emit-library={output}.program.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --write-marker-interface {output}.interface.polyast
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

namespace vendor {

struct POLYREGION_TYPE_VARIABLE("Element") Element {
  int storage;
};

template <class Signature> struct Callable;

template <class R, class... Args> struct POLYREGION_CALLABLE_VARIABLE("Callable0") Callable<R(Args...)> {
  R operator()(Args...) const { return R{}; }
};

Element load(Element *ptr) { return *ptr; }

} // namespace vendor

namespace cub {
template <class T> T ThreadLoad(T *ptr) { return *ptr; }
} // namespace cub

namespace vendor {

POLYREGION_EXPORT_AS("bar.implementation.apply")
POLYREGION_IMPLEMENTS("bar.apply") Element apply(Element x, Callable<Element(Element)> op) {
  #ifdef USE_CUB
  return op(cub::ThreadLoad(&x));
  #else
  return op(load(&x));
  #endif
}

} // namespace vendor

#else

namespace foo {
template <class T, class Op> [[clang::annotate("polyregion_interface:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
} // namespace foo

int main() {
  std::printf("%d", foo::apply(40, [](int x) { return x + 2; }));
}

#endif
