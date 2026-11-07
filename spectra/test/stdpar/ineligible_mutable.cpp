#pragma region case: mutable-callable
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 0 bound to spectra
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
    std::vector<int> v(n);
    std::generate(par, v.begin(), v.end(), [c = 0]() mutable { return c++; });
    for (std::size_t i = 0; i < n; ++i) {
      if (v[i] != static_cast<int>(i)) {
        failures += failures.empty() ? "generate" : " generate";
        break;
      }
    }
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
