#pragma region case: reused-callable-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

namespace foo {
template <class T, class Left, class Right>
[[clang::annotate("polyregion_interface:foo:bar.combine")]] inline T combine(T x, Left left, Right right) {
  __builtin_trap();
}

int plusTwo(int x) { return x + 2; }
} // namespace foo

int main() { std::printf("%d", foo::combine(38, &foo::plusTwo, &foo::plusTwo)); }
