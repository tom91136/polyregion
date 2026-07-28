#pragma region case: goto
#pragma region offload-only
#pragma region compile-fails: Unhandled stmt GotoStmt at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=1 -o {output} {input}

#pragma region case: try-catch
#pragma region offload-only
#pragma region compile-fails: Unsupported try/catch at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=2 -o {output} {input}

#pragma region case: inline-asm
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=3 -o {output} {input}

#ifndef CHECK_UNSUPPORTED
  #error "CHECK_UNSUPPORTED undefined"
#endif

#include "test_utils.h"

int main() {
  return __polyregion_offload_f1__([=]() {
#if CHECK_UNSUPPORTED == 1
    int result = 1;
    goto done;
    result = 2;
  done:;
    return result;
#elif CHECK_UNSUPPORTED == 2
    int result = 1;
    try {
      result = 2;
    } catch (...) {
      result = 3;
    }
    return result;
#elif CHECK_UNSUPPORTED == 3
    asm volatile("nop");
    return 0;
#endif
  });
}
