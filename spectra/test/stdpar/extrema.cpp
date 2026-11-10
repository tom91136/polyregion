#pragma region case: extrema
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 6 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: extrema-baseline
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
#include <utility>
#include <vector>

namespace {
std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}

template <class It> std::pair<std::ptrdiff_t, std::ptrdiff_t> offsets(const std::pair<It, It> &r, It base) {
  return {r.first - base, r.second - base};
}
} // namespace

int main() {
  const auto par = std::execution::par_unseq;
  const auto descending = [](const int x, const int y) { return x > y; };
  for (const std::size_t n : {0u, 1u, 2u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> ties(n);
    for (std::size_t i = 0; i < n; ++i)
      ties[i] = static_cast<int>((i * 7919 + 3) % 11);
    std::vector<int> flat(n, 5);
    std::vector<int> ramp(n);
    for (std::size_t i = 0; i < n; ++i)
      ramp[i] = static_cast<int>((i * 7919) % 1009) - 500;

    for (const auto *v : {&ties, &flat, &ramp}) {
      const auto b = v->begin(), e = v->end();
      expect("min_element", std::min_element(par, b, e) - b == std::min_element(b, e) - b);
      expect("max_element", std::max_element(par, b, e) - b == std::max_element(b, e) - b);
      expect("min_element_op", std::min_element(par, b, e, descending) - b == std::min_element(b, e, descending) - b);
      expect("max_element_op", std::max_element(par, b, e, descending) - b == std::max_element(b, e, descending) - b);
      expect("minmax_element", offsets(std::minmax_element(par, b, e), b) == offsets(std::minmax_element(b, e), b));
      expect("minmax_element_op",
             offsets(std::minmax_element(par, b, e, descending), b) == offsets(std::minmax_element(b, e, descending), b));
    }

    if (n > 1) {
      const auto b = flat.cbegin(), e = flat.cend();
      expect("min_first", std::min_element(par, b, e) == b);
      expect("max_first", std::max_element(par, b, e) == b);
      const auto r = std::minmax_element(par, b, e);
      expect("minmax_first_last", r.first == b && r.second == e - 1);
    }
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
