#pragma region case: partial_subgroup
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 4 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <polyregion/spectra.hpp>

int main() {
  const auto ascending = [](const int a, const int b) {
    const int ka = (a * 7) % 5, kb = (b * 7) % 5;
    return ka < kb || (ka == kb && a < b);
  };
  const auto descending = [](const int a, const int b) {
    const int ka = (a * 7) % 5, kb = (b * 7) % 5;
    return kb < ka || (ka == kb && a < b);
  };
  const auto add = [](const int a, const int b) { return a + b; };
  std::string failures;
  for (int n = 1; n <= 40; ++n) {
    std::vector<int> asc(n), desc(n);
    for (int i = 0; i < n; ++i) {
      asc[i] = n - 1 - i;
      desc[i] = i;
    }
    std::vector<int> expectedAsc = asc, expectedDesc = desc;
    std::sort(expectedAsc.begin(), expectedAsc.end(), ascending);
    std::sort(expectedDesc.begin(), expectedDesc.end(), descending);

    spectra::sort(asc.data(), n, ascending);
    spectra::sort(desc.data(), n, descending);

    if (asc != expectedAsc) failures += " asc:" + std::to_string(n);
    if (desc != expectedDesc) failures += " desc:" + std::to_string(n);

    std::vector<int> in(n), out(n), expected(n);
    for (int i = 0; i < n; ++i) in[i] = (i * 7) % 5 - 2;

    std::inclusive_scan(in.begin(), in.end(), expected.begin(), add);
    spectra::inclusive_scan(in.data(), out.data(), n, add);
    if (out != expected) failures += " incl:" + std::to_string(n);

    std::exclusive_scan(in.begin(), in.end(), expected.begin(), 0, add);
    spectra::exclusive_scan(in.data(), out.data(), n, 0, add);
    if (out != expected) failures += " excl:" + std::to_string(n);
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
  return 0;
}
