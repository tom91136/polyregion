#pragma region case: missing-package
#pragma region offload-only
#pragma region compile-fails: no package is available for library `foo`
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -c -o {output}.o {input}

#include <cstdint>

#if !defined(CHECK_SCALAR) && !defined(CHECK_POINTER)
namespace foo {

template <class T>
[[clang::annotate("polyregion_interface:foo:bar.increment")]] inline void increment(const T *in, T *out, std::int32_t n) {
  __builtin_trap();
}

} // namespace foo

void consume(const int *in, int *out) { foo::increment(in, out, 4); }
#endif

#pragma region case: pointer-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -DCHECK_POINTER -o {output} {input}
#pragma region do: {output}

#ifdef CHECK_POINTER
  #include <cstdint>

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.copy")]] inline void copy(const T *in, T *out, std::int32_t n) {
  __builtin_trap();
}
} // namespace foo

int main() {
  const int in[4]{1, 2, 3, 4};
  int out[4]{};
  foo::copy(in, out, 4);
  return 0;
}
#endif

#pragma region case: scalar-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -DCHECK_SCALAR -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#ifdef CHECK_SCALAR
  #include <cstdio>

namespace foo {
template <class T> [[clang::annotate("polyregion_interface:foo:bar.increment")]] inline T increment(T x) { __builtin_trap(); }
} // namespace foo

int main() { std::printf("%d", foo::increment(41)); }
#endif
