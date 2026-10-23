#pragma region case: package-template-export-closure-identity
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -fstdpar-emit-library={output}.program.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --write-marker-interface {output}.interface.polyast
#pragma region do: {polypackage_emit} package link {output}.interface.polyast {output}.packages {output}.program.polyast
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42

#pragma region case: package-template-export-closure-pack-identity
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -DPACK -fstdpar-emit-library={output}.program.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --write-marker-interface {output}.interface.polyast
#pragma region do: {polypackage_emit} package link {output}.interface.polyast {output}.packages {output}.program.polyast
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42 42

#include <cstdio>

#ifdef PRODUCER

  #include <utility>

  #define POLYREGION_EXPORT_TEMPLATE(name) [[clang::annotate("polyregion_export_template:" name)]]
  #define POLYREGION_IMPLEMENTS(name) [[clang::annotate("polyregion_implements:" name)]]
  #define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]
  #define POLYREGION_CALLABLE_VARIABLE(name) [[clang::annotate("polyregion_callable_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("E4:size=4") alignas(4) E4 {
  char storage[4];
};
struct POLYREGION_TYPE_VARIABLE("E8:size=8") alignas(8) E8 {
  char storage[8];
};

template <class Signature> struct Callable;

template <class R, class... Args> struct POLYREGION_CALLABLE_VARIABLE("Callable0") Callable<R(Args...)> {
  R operator()(Args...) const { return R{}; }
};

  #ifdef PACK
template <class... F> struct Holder;
template <class F> struct Holder<F> {
  F f;
};
  #else
template <class F> struct Holder {
  F f;
  Holder(F g) : f(std::move(g)) {}
};
  #endif

template <class E>
POLYREGION_EXPORT_TEMPLATE("bar.implementation.apply")
POLYREGION_IMPLEMENTS("bar.apply") E apply(E x, Callable<E(E)> op) {
  auto value = [x]() { return x; };
  Holder<decltype(value)> holder{value};
  return op(holder.f());
}

template E4 apply(E4, Callable<E4(E4)>);
template E8 apply(E8, Callable<E8(E8)>);

#else

namespace foo {
template <class T, class Op> [[clang::annotate("polyregion_interface:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
} // namespace foo

int main() {
  std::printf("%d %lld", foo::apply(40, [](int x) { return x + 2; }), foo::apply(40LL, [](long long x) { return x + 2; }));
}

#endif
