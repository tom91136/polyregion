#pragma region case: assert_element
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 1 .core_transform_reduce_w4)
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 1 972976 1 972976

#pragma region case: assert_element-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 1 972976 1 972976

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

#include "polystl/polystl.h"

struct Cell {
  int value;
  int done;
};

int completed(const std::vector<Cell> &cells) {
  int total = 0;
  for (const Cell &c : cells)
    total += c.done;
  return total;
}

int main() {
  using polyregion::polystl::details::lastAssert;
  constexpr int n = 1000003;
  std::vector<Cell> cells(n, Cell{1, 0});
  for (int i = 9; i < n; i += 37)
    cells[i].value = -1;
  std::for_each(std::execution::par_unseq, cells.begin(), cells.end(), [](Cell &c) {
    assert(c.value >= 0);
    c.done = 1;
  });
  const bool forEachRaised = lastAssert().raised;
  const int forEachDone = completed(cells);
  for (Cell &c : cells)
    c.done = 0;
  std::transform_reduce(std::execution::par_unseq, cells.begin(), cells.end(), 0, std::plus<>(), [](Cell &c) {
    assert(c.value >= 0);
    c.done = 1;
    return 1;
  });
  const bool reduceRaised = lastAssert().raised;
  std::printf("%d %d %d %d", forEachRaised ? 1 : 0, forEachDone, reduceRaised ? 1 : 0, completed(cells));
  return 0;
}
