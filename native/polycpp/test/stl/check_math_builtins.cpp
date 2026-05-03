#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: sum=44.000000 abs=16.000000 pow=8.000000

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <execution>

int main() {
  std::array<float, 8> v{1, 4, 9, 16, 25, 36, 49, 64};
  std::array<int, 8> idx{0, 1, 2, 3, 4, 5, 6, 7};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [data = v.data()](int i) {
    float x = data[i];
    float s = std::sin(x);
    float c = std::cos(x);
    data[i] = std::sqrt(std::abs(x)) + s * s + c * c;
  });
  float sum = 0;
  for (auto x : v) sum += x;

  std::array<float, 1> negv{-16.0f};
  std::array<int, 1> negi{0};
  std::for_each(std::execution::par_unseq, negi.begin(), negi.end(),
                [d = negv.data()](int) { d[0] = std::abs(d[0]); });

  std::array<float, 1> powv{0};
  std::array<int, 1> powi{0};
  std::for_each(std::execution::par_unseq, powi.begin(), powi.end(),
                [d = powv.data()](int) { d[0] = std::pow(2.0f, 3.0f); });

  printf("sum=%f abs=%f pow=%f", sum, negv[0], powv[0]);
  fflush(stdout);
  return 0;
}
