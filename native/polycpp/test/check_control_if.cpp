#pragma region case: a<b
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXPR_DEF=a<b -o {output} {input}
#pragma region do: A=1 B=2 {output}
#pragma region requires: 42
#pragma region do: A=2 B=1 {output}
#pragma region requires: -1
#pragma region do: A=1 B=1 {output}
#pragma region requires: -1

#pragma region case: a>b
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXPR_DEF=a>b -o {output} {input}
#pragma region do: A=1 B=2 {output}
#pragma region requires: -1
#pragma region do: A=2 B=1 {output}
#pragma region requires: 42
#pragma region do: A=1 B=1 {output}
#pragma region requires: -1

#pragma region case: a==b
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXPR_DEF=a==b -o {output} {input}
#pragma region do: A=1 B=2 {output}
#pragma region requires: -1
#pragma region do: A=2 B=1 {output}
#pragma region requires: -1
#pragma region do: A=1 B=1 {output}
#pragma region requires: 42

#pragma region case: a!=b
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXPR_DEF=a!=b -o {output} {input}
#pragma region do: A=1 B=2 {output}
#pragma region requires: 42
#pragma region do: A=2 B=1 {output}
#pragma region requires: 42
#pragma region do: A=1 B=1 {output}
#pragma region requires: -1

#pragma region case: a==b?false:true
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_EXPR_DEF=a==b?false:true -o {output} {input}
#pragma region do: A=1 B=2 {output}
#pragma region requires: 42
#pragma region do: A=2 B=1 {output}
#pragma region requires: 42
#pragma region do: A=1 B=1 {output}
#pragma region requires: -1

#ifndef CHECK_EXPR_DEF
  #error "CHECK_EXPR_DEF undefined"
#endif

#include <cstddef>
#include <cstdio>
#include <string>

#include "test_utils.h"

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
