#pragma region case: stateful-callable-package-diagnostic
#pragma region offload-only
#pragma region compile-fails: stateful library callables are not supported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSTATEFUL_CALLABLE -fstdpar-library-path={output}.packages -c -o {output}.o {input}

#pragma region case: conflicting-callable-package-diagnostic
#pragma region offload-only
#pragma region compile-fails: one interface specialization cannot bind conflicting callable identities
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCONFLICTING_CALLABLES -fstdpar-library-path={output}.packages -c -o {output}.o {input}

#pragma region case: reused-callable-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 86

#include <cstdio>

namespace foo {
template <class T, class Left, class Right>
[[clang::annotate("polyregion_interface:foo:bar.combine")]] inline T combine(T x, Left left, Right right) {
  __builtin_trap();
}

int plusOne(int x) { return x + 1; }
int plusTwo(int x) { return x + 2; }
} // namespace foo

#ifdef STATEFUL_CALLABLE
struct Add {
  int bias;
  int operator()(int value) const { return value + bias; }
};
int main() { return foo::combine(38, Add{2}, Add{2}); }
#elif defined(CONFLICTING_CALLABLES)
int main() {
  const auto first = foo::combine(38, &foo::plusOne, &foo::plusOne);
  return first + foo::combine(38, &foo::plusTwo, &foo::plusTwo);
}
#else
int main() {
  const auto first = foo::combine(38, &foo::plusTwo, &foo::plusTwo);
  const auto second = foo::combine(40, &foo::plusTwo, &foo::plusTwo);
  std::printf("%d", first + second);
}
#endif
