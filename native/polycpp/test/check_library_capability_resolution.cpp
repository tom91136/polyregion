#pragma region case: capability-resolution-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.capable_increment")]] inline T increment(T x) { __builtin_trap(); }
} // namespace foo

int main() { std::printf("%d", foo::increment(41)); }
