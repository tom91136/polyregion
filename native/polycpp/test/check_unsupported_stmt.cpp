#pragma region case: goto
#pragma region offload-only
#pragma region compile-fails: Unhandled stmt GotoStmt at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}

#include "test_utils.h"

int main() {
  return __polyregion_offload_f1__([=]() {
    int result = 1;
    goto done;
    result = 2;
  done:;
    return result;
  });
}
