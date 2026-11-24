#pragma region case: glcompute-type-pun
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: type pun ok

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  const double table[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<int> idx{5, 0, 7, 3, 1, 6, 2, 4};
  std::vector<std::uint64_t> out(idx.size(), 0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&, o = out.data()](int i) {
    double value = table[i];
    std::uint64_t bits = *reinterpret_cast<std::uint64_t *>(&value);
    o[i] = bits;
  });
  const std::uint64_t expected[8] = {0x3ff0000000000000ULL, 0x4000000000000000ULL, 0x4008000000000000ULL, 0x4010000000000000ULL,
                                     0x4014000000000000ULL, 0x4018000000000000ULL, 0x401c000000000000ULL, 0x4020000000000000ULL};
  int failures = 0;
  for (std::size_t i = 0; i < idx.size(); ++i)
    if (out[i] != expected[i]) {
      printf("table[%zu]: got %016llx expected %016llx\n", i, (unsigned long long)out[i], (unsigned long long)expected[i]);
      ++failures;
    }
  printf(failures == 0 ? "type pun ok" : "type pun failures=%d", failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
}
