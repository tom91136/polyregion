#pragma region case: indirect-call
#pragma region offload-only
#pragma region compile-fails: Call with no resolvable callee
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CALL=1 -o {output} {input}

#pragma region case: pseudo-destructor
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CALL=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

int increment(int value) { return value + 1; }

int main() {
  int result = __polyregion_offload_f1__([=]() {
#if CHECK_CALL == 1
    auto fn = &increment;
    return fn(41);
#elif CHECK_CALL == 2
    using Scalar = int;
    int value = 42;
    (&value)->~Scalar();
    return value;
#else
  #error "CHECK_CALL undefined"
#endif
  });
  std::printf("%d", result);
  return 0;
}
