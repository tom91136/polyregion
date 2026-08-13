#pragma region case: multi-tu-import-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -DTU_A -c -o {output}.a.o {input}
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -DTU_B -c -o {output}.b.o {input}
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DMAIN -c -o {output}.main.o {input}
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {output}.a.o {output}.b.o {output}.main.o
#pragma region do: {output}
#pragma region requires: 84

#include <cstdio>

#if defined(TU_A)
namespace multi_tu_a {
template <class T> [[clang::annotate("polyregion_import:foo:bar.increment")]] inline T increment(T x) { __builtin_trap(); }
} // namespace multi_tu_a
extern "C" int a() { return multi_tu_a::increment(41); }
#elif defined(TU_B)
namespace multi_tu_b {
template <class T, class Op> [[clang::annotate("polyregion_import:foo:bar.apply")]] inline T apply(T x, Op op) { __builtin_trap(); }
} // namespace multi_tu_b
extern "C" int b() {
  return multi_tu_b::apply(40, [](int x) { return x + 2; });
}
#elif defined(MAIN)
extern "C" int a();
extern "C" int b();
int main() { std::printf("%d", a() + b()); }
#endif
