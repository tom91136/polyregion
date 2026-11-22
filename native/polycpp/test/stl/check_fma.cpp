#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: fma OK double=1 float=1

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  std::vector<double> d{0.1, 10.0, -1.0, 0.0};
  std::vector<float> f{0.1f, 10.0f, -1.0f, 0.0f};
  std::vector<int> idx{0};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = d.data(), f = f.data()](int) {
    d[3] = std::fma(d[0], d[1], d[2]);
    f[3] = std::fma(f[0], f[1], f[2]);
  });
  const bool dOk = d[3] != 0.0 && d[3] == std::fma(0.1, 10.0, -1.0);
  const bool fOk = f[3] != 0.0f && f[3] == std::fma(0.1f, 10.0f, -1.0f);
  printf("fma OK double=%d float=%d", dOk, fOk);
  fflush(stdout);
  return 0;
}
