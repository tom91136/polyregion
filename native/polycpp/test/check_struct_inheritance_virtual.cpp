#pragma region case: single
#pragma region offload-only
#pragma region compile-fails: Unsupported virtual base in Derived
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_VIRTUAL=1 -o {output} {input}

#pragma region case: diamond
#pragma region offload-only
#pragma region compile-fails: Unsupported virtual base in Diamond
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_VIRTUAL=2 -o {output} {input}

#include "test_utils.h"

int main() {

  struct Base {
    int x;
  };

#if CHECK_VIRTUAL == 1
  struct Derived : virtual Base {
    int y;
  };

  Derived value{};
  return __polyregion_offload_f1__([=]() { return value.x + value.y; });
#elif CHECK_VIRTUAL == 2
  struct Left : virtual Base {
    int l;
  };

  struct Right : virtual Base {
    int r;
  };

  struct Diamond : Left, Right {
    int d;
  };

  Diamond value{};
  return __polyregion_offload_f1__([=]() { return value.x + value.l + value.r + value.d; });
#else
  #error "CHECK_VIRTUAL undefined"
#endif
}
