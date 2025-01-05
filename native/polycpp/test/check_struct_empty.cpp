#pragma region case: capture
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct foo{};
  foo value{};
  __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });

  return 0;
}
