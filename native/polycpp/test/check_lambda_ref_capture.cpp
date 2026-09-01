#pragma region case: ref-captured-pointer-and-reference
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 168

#include <cstdio>

#include "test_utils.h"

int main() {
  int capturedValue = 40;
  int result = __polyregion_offload_f1__([=]() mutable {
    int pointerValue = 40;
    int *pointer = &pointerValue;
    auto bumpPointer = [&pointer]() { *pointer += 2; };
    bumpPointer();

    int referenceValue = 40;
    int &reference = referenceValue;
    auto bumpReference = [&reference]() { reference += 2; };
    bumpReference();

    int scalarValue = 40;
    auto bumpScalar = [&scalarValue]() { scalarValue += 2; };
    bumpScalar();

    int *capturedPointer = &capturedValue;
    auto bumpCapturedPointer = [&capturedPointer]() { *capturedPointer += 2; };
    bumpCapturedPointer();

    return pointerValue + referenceValue + scalarValue + capturedValue;
  });
  std::printf("%d", result);
  return 0;
}
