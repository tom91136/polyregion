#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: int=42 float=3.250000 bigfloat=99.000000

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>
#include <limits>

namespace {
constexpr int kAnswer = 42;
constexpr float kQuarter = 0.25f;
const float kBig = std::numeric_limits<float>::max();
} // namespace

int main() {
  std::array<int, 1> idx{0};

  std::array<int, 1> outI{0};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [d = outI.data()](int) { d[0] = kAnswer; });

  std::array<float, 1> outF{0};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [d = outF.data()](int) { d[0] = 3.0f + kQuarter; });

  std::array<float, 1> outBig{0};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [d = outBig.data()](int) { d[0] = (kBig > 0.0f) ? 99.0f : -1.0f; });

  printf("int=%d float=%f bigfloat=%f", outI[0], outF[0], outBig[0]);
  fflush(stdout);
  return 0;
}
