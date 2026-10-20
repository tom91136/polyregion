#pragma region case: static_runtime
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-rt=static -o {output} {input}
#pragma region do: {output}
#pragma region requires: 90

#include <algorithm>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  std::vector<int> xs(10);
  for (int i = 0; i < 10; ++i)
    xs[i] = i;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x *= 2; });
  int sum = 0;
  for (const int x : xs)
    sum += x;
  std::printf("%d", sum);
  return 0;
}
