#pragma region case: fallback-without-package
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 141

#pragma region case: fallback-with-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: fallback-without-exceptions
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fno-exceptions -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 141

#include <cstdio>

namespace foo {
template <class T>
[[clang::annotate("polyregion_interface:foo:bar.increment")]] [[clang::annotate("polyregion_interface_fallback")]] inline T increment(T x) {
  return x + 100;
}
} // namespace foo

int main() { std::printf("%d", foo::increment(41)); }
