#pragma region case: find
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 3 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: find-baseline
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
  const auto is_seven = [](const int x) { return x == 7; };
  const auto negative = [](const int x) { return x < 0; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    for (const int needle : {-500, 0, 7, 508, 9999})
      expect("find", std::find(par, a.begin(), a.end(), needle) == std::find(a.begin(), a.end(), needle));
    expect("find_if", std::find_if(par, a.begin(), a.end(), is_seven) == std::find_if(a.begin(), a.end(), is_seven));
    expect("find_if_not", std::find_if_not(par, a.begin(), a.end(), negative) == std::find_if_not(a.begin(), a.end(), negative));
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
