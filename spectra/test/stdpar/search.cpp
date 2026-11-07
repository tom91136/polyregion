#pragma region case: search
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 4 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: search-baseline
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
  const auto same = [](const int x, const int y) { return x == y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i / 4) % 11);
    for (const std::size_t m : {0u, 1u, 2u, 5u, 33u}) {
      std::vector<int> present = m <= n ? std::vector<int>(a.begin() + (n - m) / 2, a.begin() + (n - m) / 2 + m) : std::vector<int>(m, 42);
      std::vector<int> absent(m, 42);
      for (std::vector<int> needle : {present, absent}) {
        expect("search", std::search(par, a.begin(), a.end(), needle.begin(), needle.end()) ==
                             std::search(a.begin(), a.end(), needle.begin(), needle.end()));
        expect("search_pred", std::search(par, a.begin(), a.end(), needle.begin(), needle.end(), same) ==
                                  std::search(a.begin(), a.end(), needle.begin(), needle.end(), same));
      }
    }
    for (const std::size_t c : {0u, 1u, 4u, 5u, 33u}) {
      expect("search_n", std::search_n(par, a.begin(), a.end(), c, 3) == std::search_n(a.begin(), a.end(), c, 3));
      expect("search_n_pred", std::search_n(par, a.begin(), a.end(), c, 10, same) == std::search_n(a.begin(), a.end(), c, 10, same));
    }
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
