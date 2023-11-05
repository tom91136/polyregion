// #CASE: a<b
// #RUN: polycpp -fstdpar -DCHECK_EXPR_DEF=a<b -o {output} {input}
// #RUN: POLY_PLATFORM=host A=1 B=2 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=2 B=1 {output}
//   #EXPECT: -1
// #RUN: POLY_PLATFORM=host A=1 B=1 {output}
//   #EXPECT: -1

// #CASE: a>b
// #RUN: polycpp -fstdpar -DCHECK_EXPR_DEF=a>b -o {output} {input}
// #RUN: POLY_PLATFORM=host A=1 B=2 {output}
//   #EXPECT: -1
// #RUN: POLY_PLATFORM=host A=2 B=1 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=1 B=1 {output}
//   #EXPECT: -1

// #CASE: a==b
// #RUN: polycpp -fstdpar -DCHECK_EXPR_DEF=a==b -o {output} {input}
// #RUN: POLY_PLATFORM=host A=1 B=2 {output}
//   #EXPECT: -1
// #RUN: POLY_PLATFORM=host A=2 B=1 {output}
//   #EXPECT: -1
// #RUN: POLY_PLATFORM=host A=1 B=1 {output}
//   #EXPECT: 42

// #CASE: a!=b
// #RUN: polycpp -fstdpar -DCHECK_EXPR_DEF=a!=b -o {output} {input}
// #RUN: POLY_PLATFORM=host A=1 B=2 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=2 B=1 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=1 B=1 {output}
//   #EXPECT: -1

// #CASE: a==b?false:true
// #RUN: polycpp -fstdpar -DCHECK_EXPR_DEF=a==b?false:true -o {output} {input}
// #RUN: POLY_PLATFORM=host A=1 B=2 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=2 B=1 {output}
//   #EXPECT: 42
// #RUN: POLY_PLATFORM=host A=1 B=1 {output}
//   #EXPECT: -1



#include <cstddef>
#include <cstdio>
#include <string>

#ifndef CHECK_EXPR_DEF
  #error "CHECK_EXPR_DEF undefined"
#endif

int main() {


  auto a = std::stoi(std::getenv("A"));
  auto b = std::stoi(std::getenv("B"));

  int result = __polyregion_offload_f1__([=]() {
    if (CHECK_EXPR_DEF) {
      return 42;
    } else {
      return -1;
    }
  });
  printf("%d", result);
  return 0;
}
