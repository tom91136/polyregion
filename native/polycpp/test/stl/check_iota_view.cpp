#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -std=c++20 -o {output} {input}
#pragma region do: SIZE=1024 {output}
#pragma region requires: 1047552.000000

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

int main() {
  int size = 1024;
  if (auto sizeEnv = std::getenv("SIZE"); sizeEnv) size = std::stoi(sizeEnv);
  std::vector<int> a(size, -1);
  int *p = a.data();
  auto beg = std::views::iota(0).begin();
  std::for_each_n(std::execution::par_unseq, beg, size, [p](int i) { p[i] = i * 2; });
  double sum = std::reduce(a.begin(), a.end(), 0.0, std::plus<>());
  printf("%f", sum);
  fflush(stdout);
  return EXIT_SUCCESS;
}
