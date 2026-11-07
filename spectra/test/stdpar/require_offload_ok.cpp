#pragma region case: require-offload-accepted
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-require-offload -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 2 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

int main() {
  const auto par = std::execution::par_unseq;
  std::string failures;
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1000u}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    std::vector<int> p = a, s = a;
    std::reverse(par, p.begin(), p.end());
    std::reverse(s.begin(), s.end());
    if (p != s) failures += failures.empty() ? "reverse" : " reverse";
    p = a;
    s = a;
    std::sort(par, p.begin(), p.end());
    std::sort(s.begin(), s.end());
    if (p != s) failures += failures.empty() ? "sort" : " sort";
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
