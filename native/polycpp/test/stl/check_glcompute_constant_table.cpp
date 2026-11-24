#pragma region case: glcompute-constant-table
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: table ok

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <vector>

struct Triple {
  double a, b, c;
};

int main() {
  const Triple table[8] = {
      {1.0, 2.0, 3.0},    {4.0, 5.0, 6.0},    {7.0, 8.0, 9.0},    {10.0, 11.0, 12.0},
      {13.0, 14.0, 15.0}, {16.0, 17.0, 18.0}, {19.0, 20.0, 21.0}, {22.0, 23.0, 24.0},
  };
  const std::vector<int> idx{5, 0, 7, 3, 1, 6, 2, 4};
  std::vector<double> out(idx.size(), 0.0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&, o = out.data()](int i) {
    const Triple t = table[i];
    o[i] = t.a * 100.0 + t.b * 10.0 + t.c;
  });
  const double expected[] = {123.0, 456.0, 789.0, 1122.0, 1455.0, 1788.0, 2121.0, 2454.0};
  int failures = 0;
  for (std::size_t i = 0; i < idx.size(); ++i)
    if (out[i] != expected[i]) {
      printf("table[%d]: got %.17g expected %.17g\n", idx[i], out[i], expected[i]);
      ++failures;
    }
  printf(failures == 0 ? "table ok" : "table failures=%d", failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
}
