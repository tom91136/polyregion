#pragma region case: deque-index
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 4 bound to spectra
#pragma region do: {output}
#pragma region requires: pass
#pragma region do: POLYRT_DEBUG=2 {output} multi
#pragma region stderr-count: 0 grid=(
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <execution>
#include <string>
#include <vector>

namespace {
std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}
}

int main(int argc, char **argv) {
  const auto par = std::execution::par_unseq;
  const auto sizes = argc > 1 ? std::vector<std::size_t>{10000} : std::vector<std::size_t>{0, 1, 31, 32, 33};
  for (const std::size_t n : sizes) {
    std::deque<int> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<int>((i / 4) % 11);
    std::deque<int> p = d;
    if (n > 0) p[n / 2] += 1;

    expect("find", std::find(par, d.begin(), d.end(), 3) == std::find(d.begin(), d.end(), 3));
    expect("count", std::count(par, d.begin(), d.end(), 3) == std::count(d.begin(), d.end(), 3));
    const auto r = std::mismatch(par, d.begin(), d.end(), p.begin());
    const auto s = std::mismatch(d.begin(), d.end(), p.begin());
    expect("mismatch", r.first - d.begin() == s.first - d.begin() && r.second - p.begin() == s.second - p.begin());
    expect("is_sorted_until", std::is_sorted_until(par, d.begin(), d.end()) == std::is_sorted_until(d.begin(), d.end()));
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
