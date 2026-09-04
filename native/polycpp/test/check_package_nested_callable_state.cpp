#pragma region case: package-nested-callable-nontrivial-assignment
#pragma region offload-only
#pragma region compile-fails: non-trivial callable-variable assignment
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -DCHECK_NONTRIVIAL_CALLABLE_ASSIGNMENT -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: package-nested-callable-state
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DPRODUCER -fstdpar-emit-library={output}.program.polyast -c -o {output}.o {input}
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

struct POLYREGION_TYPE_VARIABLE("Element") Element {
  int storage;
};

template <class Signature> struct Callable;

template <class R, class... Args> struct POLYREGION_CALLABLE_VARIABLE("Callable0") Callable<R(Args...)> {
  Callable() = default;
  Callable(const Callable &) = default;
  #ifdef CHECK_NONTRIVIAL_CALLABLE_ASSIGNMENT
  Callable &operator=(const Callable &) { return *this; }
  #else
  Callable &operator=(const Callable &) = default;
  #endif
  R operator()(Args...) const { return R{}; }
};

template <class F> struct Agent {
  F op;

  Agent() = default;
  Agent(const Agent &) = default;

  Element invoke(Element value) const { return op(value); }
};

POLYREGION_EXPORT_AS("bar.implementation.apply")
POLYREGION_IMPLEMENTS("bar.apply") Element apply(Element value, Callable<Element(Element)> op) {
  Agent<Callable<Element(Element)>> first{};
  first.op = op;
  Agent<Callable<Element(Element)>> second(first);
  return second.invoke(value);
}

#else

namespace foo {
template <class T, class Op> [[clang::annotate("polyregion_interface:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
} // namespace foo

int main() {
  std::printf("%d", foo::apply(40, [](int x) { return x + 2; }));
}

#endif
