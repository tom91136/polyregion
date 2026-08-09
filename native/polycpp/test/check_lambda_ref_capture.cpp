#pragma region case: ref-captured-pointer-and-reference
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 84

#include <cstdio>

#include "test_utils.h"

int main() {
  int result = __polyregion_offload_f1__([=]() mutable {
    int pointerValue = 40;
    int *pointer = &pointerValue;
    auto bumpPointer = [&pointer]() { *pointer += 2; };
    bumpPointer();

    int referenceValue = 40;
    int &reference = referenceValue;
    auto bumpReference = [&reference]() { reference += 2; };
    bumpReference();

    return pointerValue + referenceValue;
  });
  std::printf("%d", result);
  return 0;
}
