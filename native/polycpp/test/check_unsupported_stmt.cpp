#pragma region case: goto
#pragma region offload-only
#pragma region compile-fails: Unhandled stmt GotoStmt at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=1 -o {output} {input}

#pragma region case: inline-asm
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=3 -o {output} {input}

#pragma region case: abrupt-non-trivial-catch-by-value
#pragma region offload-only
#pragma region compile-fails: an abrupt handler exit cannot preserve destruction order
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=4 -o {output} {input}

#pragma region case: partially-initialised-non-trivial-array
#pragma region offload-only
#pragma region compile-fails: only fully initialised arrays of aggregate elements are supported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_UNSUPPORTED=5 -o {output} {input}

#ifndef CHECK_UNSUPPORTED
  #error "CHECK_UNSUPPORTED undefined"
#endif

#include "test_utils.h"

struct CatchValue {
  int value;
  ~CatchValue() {}
};
struct ArrayValue {
  int *sink;
  ~ArrayValue() { ++*sink; }
};

int main() {
  return __polyregion_offload_f1__([=]() {
#if CHECK_UNSUPPORTED == 1
    int result = 1;
    goto done;
    result = 2;
  done:;
    return result;
#elif CHECK_UNSUPPORTED == 3
    asm volatile("nop");
    return 0;
#elif CHECK_UNSUPPORTED == 4
    try {
      throw CatchValue{1};
    } catch (CatchValue value) {
      return value.value;
    }
    return 0;
#elif CHECK_UNSUPPORTED == 5
    int sink = 0;
    ArrayValue values[2] = {{&sink}};
    return sink;
#endif
  });
}
