#pragma region case: callable-element-assert-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 6 8 100

#include <cstdio>
#include <vector>

#include "polyregion/enums.h"
#include "polystl/intrinsics.h"
#include "polystl/polystl.h"

namespace foo {
template <class Op> [[clang::annotate("polyregion_interface:foo:bar.for_each_index")]] inline void for_each_index(int n, int *out, Op op) {
  __builtin_trap();
}
} // namespace foo

int checked(int x) {
  if (x == 6) __polyregion_builtin_assert(static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert), "callable");
  return x + 1;
}

int main() {
  std::vector<int> data(100, -1);
  foo::for_each_index(100, data.data(), [](int i) { return checked(i); });
  const bool raised = polyregion::polystl::details::lastAssert().raised;
  std::printf("%d %d %d %d", raised ? 1 : 0, data[5], data[7], data[99]);
  return 0;
}
