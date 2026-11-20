#pragma region case: polystl-throw-dynamic
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires@0: caught: polyrt_context_acquire: operator(): unknown or released context handle

#pragma region case: polystl-throw-static
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-rt=static -o {output} {input}
#pragma region do: {output}
#pragma region requires@0: caught: polyrt_context_acquire: operator(): unknown or released context handle

#include <cstdio>
#include <exception>

#include "test_utils.h"

int main() {
  try {
    polyrt_context_acquire_or_throw(reinterpret_cast<void *>(0x1234));
  } catch (const std::exception &e) {
    std::printf("caught: %s\n", e.what());
    return 0;
  }
  std::printf("no throw\n");
  return 1;
}
