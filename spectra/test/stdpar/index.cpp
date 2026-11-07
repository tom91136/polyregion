#pragma region case: index
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 8 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: index-baseline
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
  const auto same = [](const int x, const int y) { return x == y; };
  const auto descending = [](const int x, const int y) { return x > y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    std::vector<int> b = a;
    if (n > 0) b[n / 2] += 1;
    std::vector<int> sorted = a;
    std::sort(sorted.begin(), sorted.end());

    expect("count", std::count(par, a.begin(), a.end(), 7) == std::count(a.begin(), a.end(), 7));
    expect("count_if", std::count_if(par, a.begin(), a.end(), positive) == std::count_if(a.begin(), a.end(), positive));
    expect("mismatch", std::mismatch(par, a.begin(), a.end(), b.begin()) == std::mismatch(a.begin(), a.end(), b.begin()));
    expect("mismatch_pred",
           std::mismatch(par, a.begin(), a.end(), b.begin(), same) == std::mismatch(a.begin(), a.end(), b.begin(), same));
    expect("mismatch_last",
           std::mismatch(par, a.begin(), a.end(), b.begin(), b.end()) == std::mismatch(a.begin(), a.end(), b.begin(), b.end()));
    expect("mismatch_last_pred", std::mismatch(par, a.begin(), a.end(), b.begin(), b.end(), same) ==
                                     std::mismatch(a.begin(), a.end(), b.begin(), b.end(), same));
    expect("is_sorted_until", std::is_sorted_until(par, a.begin(), a.end()) == std::is_sorted_until(a.begin(), a.end()));
    expect("is_sorted_until_sorted",
           std::is_sorted_until(par, sorted.begin(), sorted.end()) == std::is_sorted_until(sorted.begin(), sorted.end()));
    expect("is_sorted_until_op", std::is_sorted_until(par, a.begin(), a.end(), descending) ==
                                     std::is_sorted_until(a.begin(), a.end(), descending));
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
