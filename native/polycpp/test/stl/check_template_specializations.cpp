#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: n1=10 n2=20 n4=40 n8=80

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>

template <int Multiplier> int runKernel() {
  std::array<int, 1> idx{0};
  std::array<int, 1> out{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [d = out.data()](int) { d[0] = Multiplier * 10; });
  return out[0];
}

int main() {
  printf("n1=%d n2=%d n4=%d n8=%d", runKernel<1>(), runKernel<2>(), runKernel<4>(), runKernel<8>());
  fflush(stdout);
  return 0;
}
