#pragma region case: deref-leaf
#pragma region using: depth=1,2,3,20
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_DEPTH={depth} -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#pragma region case: swap-deepest
#pragma region using: depth=2,3,20
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_DEPTH={depth} -DCHECK_SWAP -o {output} {input}
#pragma region do: {output}
#pragma region requires: ok

#ifndef CHECK_DEPTH
  #error "CHECK_DEPTH undefined"
#endif

#include <cstdio>
#include <type_traits>

#include "test_utils.h"

template <int N> struct Tower {
  using Below = typename Tower<N - 1>::Ptr;
  using Ptr = Below *;
  static Ptr build() { return new Below(Tower<N - 1>::build()); }
};
template <> struct Tower<0> {
  using Ptr = int;
  static Ptr build() { return 7; }
};

template <typename P> static int leafOf(P p) {
  if constexpr (std::is_same_v<P, int *>) return *p;
  else return leafOf(*p);
}
template <typename P> static void setLeaf(P p, int v) {
  if constexpr (std::is_same_v<P, int *>) *p = v;
  else setLeaf(*p, v);
}
template <typename P> static void swapDeepest(P p, int *alt) {
  if constexpr (std::is_same_v<P, int **>) *p = alt;
  else swapDeepest(*p, alt);
}

int main() {
  auto top = Tower<CHECK_DEPTH>::build();
#ifdef CHECK_SWAP
  int *alt = new int(99);
  __polyregion_offload_f1__([top, alt]() {
    swapDeepest(top, alt);
    return 0;
  });
  printf("%s", leafOf(top) == 99 ? "ok" : "bad");
#else
  __polyregion_offload_f1__([top]() {
    setLeaf(top, 42);
    return 0;
  });
  printf("%s", leafOf(top) == 42 ? "ok" : "bad");
#endif
  return 0;
}
