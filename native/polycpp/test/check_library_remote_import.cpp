#pragma region case: remote-package-import
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#pragma region case: remote-package-import-assert
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DASSERT -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 42

#include <cstdio>

#ifdef ASSERT
  #include "polystl/polystl.h"
#endif

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.remote_increment")]] inline T remote_increment(T x) { __builtin_trap(); }
template <class T> [[clang::annotate("polyregion_interface:foo:bar.remote_assert")]] inline T remote_assert(T x) { __builtin_trap(); }
} // namespace foo

#ifdef ASSERT
int main() {
  const int y = foo::remote_assert(41);
  std::printf("%d %d", polyregion::polystl::details::lastAssert().raised ? 1 : 0, y);
}
#else
int main() { std::printf("%d", foo::remote_increment(41)); }
#endif
