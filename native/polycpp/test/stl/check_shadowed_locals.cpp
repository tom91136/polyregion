#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: sibling=15 nested=18

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>

int main() {
  std::array<int, 1> idx{0};

  std::array<int, 1> outSibling{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outSibling.data()](int) {
    int sum = 0;
    for (int l = 0; l < 5; l++) sum += 1;
    for (int l = 0; l < 5; l++) sum += l;
    d[0] = sum;
  });

  std::array<int, 1> outNested{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outNested.data()](int) {
    int sum = 0;
    for (int l = 0; l < 4; l++) {
      for (int l = 0; l < 3; l++) sum += 1;
      sum += l;
    }
    d[0] = sum;
  });

  printf("sibling=%d nested=%d", outSibling[0], outNested[0]);
  fflush(stdout);
  return 0;
}
