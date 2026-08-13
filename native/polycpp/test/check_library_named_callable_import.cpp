#pragma region case: named-callable-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

namespace foo {
template <class T, class Op> [[clang::annotate("polyregion_import:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
struct PlusTwo {
  int operator()(int x) const { return x + 2; }
};
} // namespace foo

int main() { std::printf("%d", foo::apply(40, foo::PlusTwo{})); }
