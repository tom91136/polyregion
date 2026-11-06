#pragma region case: for-each-small
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -DCHECK_SIZE=100 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 97 3 [ASRT]

#pragma region case: for-each-large
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -DCHECK_SIZE=1000003 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 972976 27027 [ASRT]

#pragma region case: transform-reduce-small
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -DCHECK_SIZE=100 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 97 3 [ASRT]

#pragma region case: transform-reduce-large
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -DCHECK_SIZE=1000003 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 972976 27027 [ASRT]

#pragma region case: raise-small
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -DCHECK_SIZE=100 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 97 3 [EXCP]

#pragma region case: raise-large
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -DCHECK_SIZE=1000003 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 972976 27027 [EXCP]

#pragma region case: never-fired
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -DCHECK_SIZE=1000003 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 1000003 0 []

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif
#ifndef CHECK_SIZE
  #define CHECK_SIZE 100
#endif

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

#include "polyregion/enums.h"

#include "test_utils.h"

constexpr int Stride = 37;
constexpr int Phase = 9;

struct Cell {
  int value;
  int done;
};

int main() {
  using polyregion::polystl::details::lastAssert;
  const int n = CHECK_SIZE;
  std::vector<Cell> cells(n, Cell{1, 0});
  for (int i = 0; i < n; ++i)
    if (i % Stride == Phase && CHECK_KIND != 3) cells[i].value = -1;
#if CHECK_KIND == 0 || CHECK_KIND == 3
  std::for_each(std::execution::par_unseq, cells.begin(), cells.end(), [](Cell &c) {
    assert(c.value >= 0);
    c.done = 1;
  });
#elif CHECK_KIND == 1
  std::transform_reduce(std::execution::par_unseq, cells.begin(), cells.end(), 0, std::plus<>(), [](Cell &c) {
    assert(c.value >= 0);
    c.done = 1;
    return c.value;
  });
#elif CHECK_KIND == 2
  std::for_each(std::execution::par_unseq, cells.begin(), cells.end(), [](Cell &c) {
    if (c.value < 0) throw c.value;
    c.done = 1;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  int completed = 0;
  for (const Cell &c : cells)
    completed += c.done;
  const auto a = lastAssert();
  const char cc[5] = {char(a.code), char(a.code >> 8), char(a.code >> 16), char(a.code >> 24), 0};
  std::printf("%d %d %d [%s]", a.raised ? 1 : 0, completed, n - completed, a.raised ? cc : "");
  return 0;
}
