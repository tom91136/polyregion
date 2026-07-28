#pragma region case: dispatch
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=1 -o {output} {input}
#pragma region do: X=1 {output}
#pragma region requires: 10
#pragma region do: X=2 {output}
#pragma region requires: 20
#pragma region do: X=7 {output}
#pragma region requires: -1

#pragma region case: fallthrough
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=2 -o {output} {input}
#pragma region do: X=1 {output}
#pragma region requires: 7
#pragma region do: X=2 {output}
#pragma region requires: 6
#pragma region do: X=3 {output}
#pragma region requires: 4
#pragma region do: X=4 {output}
#pragma region requires: 8
#pragma region do: X=5 {output}
#pragma region requires: 0

#pragma region case: default-not-last
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=3 -o {output} {input}
#pragma region do: X=1 {output}
#pragma region requires: 1
#pragma region do: X=2 {output}
#pragma region requires: 2
#pragma region do: X=3 {output}
#pragma region requires: 3
#pragma region do: X=9 {output}
#pragma region requires: 102

#pragma region case: break-in-loop
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=4 -o {output} {input}
#pragma region do: X=0 {output}
#pragma region requires: 5122

#pragma region case: no-compound-body
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=5 -o {output} {input}
#pragma region do: X=1 {output}
#pragma region requires: 42
#pragma region do: X=2 {output}
#pragma region requires: 0

#pragma region case: default-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=6 -o {output} {input}
#pragma region do: X=1 {output}
#pragma region requires: 7
#pragma region do: X=2 {output}
#pragma region requires: 7

#pragma region case: continue-in-switch
#pragma region offload-only
#pragma region compile-fails: Unsupported continue targeting an enclosing loop from a switch at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SWITCH=7 -o {output} {input}

#pragma region case: case-range
#pragma region offload-only
#pragma region compile-fails: Unsupported case range at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -Wno-gnu-case-range -DCHECK_SWITCH=8 -o {output} {input}

#ifndef CHECK_SWITCH
  #error "CHECK_SWITCH undefined"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>

#include "test_utils.h"

int main() {
  auto x = std::stoi(std::getenv("X"));

  int result = __polyregion_offload_f1__([=]() {
#if CHECK_SWITCH == 1
    int r = 0;
    switch (x) {
      case 1: r = 10; break;
      case 2: r = 20; break;
      default: r = -1; break;
    }
    return r;
#elif CHECK_SWITCH == 2
    int r = 0;
    switch (x) {
      case 1: r += 1;
      case 2: r += 2;
      case 3: r += 4; break;
      case 4: r += 8;
    }
    return r;
#elif CHECK_SWITCH == 3
    int r = 0;
    switch (x) {
      case 1: r = 1; break;
      default: r = 100;
      case 2: r += 2; break;
      case 3: r = 3; break;
    }
    return r;
#elif CHECK_SWITCH == 4
    int r = 0;
    for (int i = x; i < x + 5; ++i) {
      switch (i % 3) {
        case 0: r += 1; break;
        case 1: r += 10; break;
        default: r += 100; break;
      }
      r += 1000;
    }
    return r;
#elif CHECK_SWITCH == 5
    int r = 0;
    {
      switch (x)
      case 1: r = 42;
    }
    return r;
#elif CHECK_SWITCH == 6
    int r = 0;
    switch (x) {
      default: r = 7;
    }
    return r;
#elif CHECK_SWITCH == 7
    int r = 0;
    for (int i = x; i < x + 3; ++i) {
      switch (i) {
        case 0: continue;
        default: r += 1;
      }
      r += 10;
    }
    return r;
#elif CHECK_SWITCH == 8
    int r = 0;
    switch (x) {
      case 1 ... 3: r = 5; break;
      default: r = 0; break;
    }
    return r;
#endif
  });
  std::printf("%d", result);
  return 0;
}
