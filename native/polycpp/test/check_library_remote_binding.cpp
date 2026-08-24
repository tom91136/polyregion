#pragma region case: remote-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.remote_increment")]] inline T remote_increment(T x) { __builtin_trap(); }
} // namespace foo

int main() { std::printf("%d", foo::remote_increment(41)); }
