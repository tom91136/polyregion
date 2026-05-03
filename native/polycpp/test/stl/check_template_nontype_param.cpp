#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: n1=1 n4=10 n8=36

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>

template <size_t N> int triangleSum() {
  std::array<int, 1> outArr{-1};
  std::array<int, 1> idx{0};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outArr.data()](int) {
    int sum = 0;
    for (size_t l = 0; l < N; l++) sum += static_cast<int>(l) + 1;
    d[0] = sum;
  });
  return outArr[0];
}

int main() {
  printf("n1=%d n4=%d n8=%d", triangleSum<1>(), triangleSum<4>(), triangleSum<8>());
  fflush(stdout);
  return 0;
}
