#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: sum OK abs=16.000000 pow=8.000000

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  std::vector<float> v{1, 4, 9, 16, 25, 36, 49, 64};
  std::vector<int> idx{0, 1, 2, 3, 4, 5, 6, 7};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [data = v.data()](int i) {
    float x = data[i];
    float s = std::sin(x);
    float c = std::cos(x);
    data[i] = std::sqrt(std::abs(x)) + s * s + c * c;
  });
  float sum = 0;
  for (auto x : v)
    sum += x;

  std::vector<float> negv{-16.0f};
  std::vector<int> negi{0};
  std::for_each(std::execution::par_unseq, negi.begin(), negi.end(), //
                [d = negv.data()](int) { d[0] = std::abs(d[0]); });

  std::vector<float> powv{0};
  std::vector<int> powi{0};
  std::for_each(std::execution::par_unseq, powi.begin(), powi.end(), //
                [d = powv.data()](int) { d[0] = std::pow(2.0f, 3.0f); });

  if (std::abs(sum - 44.0f) < 2.0f) printf("sum OK abs=%f pow=%f", negv[0], powv[0]);
  else printf("sum=%f abs=%f pow=%f", sum, negv[0], powv[0]);
  fflush(stdout);
  return 0;
}
