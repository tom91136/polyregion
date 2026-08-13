#pragma region case: two-callable-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 84

#include <cstdio>

namespace foo {
template <class T, class Left, class Right>
[[clang::annotate("polyregion_import:foo:bar.combine")]] inline T combine(T x, Left left, Right right) {
  __builtin_trap();
}
} // namespace foo

int main() {
  std::printf("%d", foo::combine(40, [](int x) { return x + 2; }, [](int x) { return x * 2; }));
}
