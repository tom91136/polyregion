#pragma region case: predicates
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 9 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: predicates-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

namespace {
std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}
}

int main() {
  const auto par = std::execution::par_unseq;
  const auto positive = [](const int x) { return x > 0; };
  const auto negative = [](const int x) { return x < 0; };
  const auto same = [](const int x, const int y) { return x == y; };
  const auto descending = [](const int x, const int y) { return x > y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    std::vector<int> b = a;
    if (n > 0) b[n / 2] += 1;
    std::vector<int> sorted = a;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> sub;
    for (std::size_t i = 0; i < n; i += 2) sub.push_back(sorted[i]);
    std::vector<int> partitioned = a;
    std::partition(partitioned.begin(), partitioned.end(), negative);

    expect("all_of", std::all_of(par, a.begin(), a.end(), positive) == std::all_of(a.begin(), a.end(), positive));
    expect("any_of", std::any_of(par, a.begin(), a.end(), positive) == std::any_of(a.begin(), a.end(), positive));
    expect("none_of", std::none_of(par, a.begin(), a.end(), positive) == std::none_of(a.begin(), a.end(), positive));
    expect("equal", std::equal(par, a.begin(), a.end(), b.begin(), same) == std::equal(a.begin(), a.end(), b.begin(), same));
    expect("equal_default", std::equal(par, a.begin(), a.end(), b.begin()) == std::equal(a.begin(), a.end(), b.begin()));
    expect("includes", std::includes(par, sorted.begin(), sorted.end(), sub.begin(), sub.end()) ==
                           std::includes(sorted.begin(), sorted.end(), sub.begin(), sub.end()));
    expect("is_partitioned", std::is_partitioned(par, partitioned.begin(), partitioned.end(), negative) ==
                                 std::is_partitioned(partitioned.begin(), partitioned.end(), negative));
    expect("is_sorted", std::is_sorted(par, sorted.begin(), sorted.end()) == std::is_sorted(sorted.begin(), sorted.end()));
    expect("is_sorted_op", std::is_sorted(par, a.begin(), a.end(), descending) == std::is_sorted(a.begin(), a.end(), descending));
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
