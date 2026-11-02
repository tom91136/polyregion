#pragma region case: string-temp-record
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3072 6
#pragma region stderr-count: 0 failed to release

#include <cstddef>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

int main() {
  constexpr std::size_t n = 1024;
  std::vector<std::size_t> out(n);
  std::size_t total = 0;
  for (int round = 0; round < 3; ++round) {
    total += std::string(static_cast<std::size_t>(round) + 1, 'a').size();
    std::for_each(std::execution::par_unseq, out.begin(), out.end(), [&](std::size_t &x) { x = static_cast<std::size_t>(round) + 1; });
  }
  std::size_t sum = 0;
  for (auto v : out)
    sum += v;
  std::printf("%zu %zu", sum, total);
  return 0;
}
