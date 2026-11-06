#pragma region case: launch_geometry
#pragma region using: opt=O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
#pragma region do: POLYRT_DEBUG=2 {output}
#pragma region stderr-count: 1 grid=(
#pragma region stderr-count: 1 block=(256,1,1)
#pragma region requires: 100000

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

int main() {
  const int n = 100000;
  std::vector<int> xs(n, 1);
  int *p = xs.data();
  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [p](int i) { p[i] = i; });
  long sum = 0;
  for (int i = 0; i < n; ++i)
    sum += p[i] == i;
  std::printf("%ld", sum);
  return 0;
}
