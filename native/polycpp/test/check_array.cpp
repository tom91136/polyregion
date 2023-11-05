// #CASE: =array
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE== -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: -1 -1

// #CASE: &array
// #MATRIX: size=1,10,100
// #RUN: polycpp -fstdpar -DCHECK_SIZE_DEF={size} -DCHECK_CAPTURE=& -DCHECK_MUT -o {output} {input}
// #RUN: POLY_PLATFORM=host {output}
//   #EXPECT: 42 42

#include <cstddef>
#include <cstdio>
#include <cstring>

#ifndef CHECK_SIZE_DEF
  #error "CHECK_SIZE_DEF undefined"
#endif

#ifndef CHECK_CAPTURE
  #error "CHECK_CAPTURE undefined"
#endif



int main() {

  int xs[CHECK_SIZE_DEF] = {};
  std::memset(xs, -1, sizeof(xs));
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
#ifdef CHECK_MUT
    xs[CHECK_SIZE_DEF - 1] = 42;
#endif
//    int aaa = xs[4];
    return xs[CHECK_SIZE_DEF - 1];
  });
  printf("%d %d", result, xs[CHECK_SIZE_DEF - 1]);
  return 0;
}
