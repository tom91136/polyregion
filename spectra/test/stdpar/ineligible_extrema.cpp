#pragma region case: deque-extrema
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
#include <iterator>
#include <list>
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
  const auto negative = [](const int x) { return x < 0; };
  const auto sizes = argc > 1 ? std::vector<std::size_t>{10000} : std::vector<std::size_t>{0, 1, 31, 32, 33};
  for (const std::size_t n : sizes) {
    std::deque<int> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<int>((i * 7919 + 3) % 11) - 5;

    expect("min_element", std::min_element(par, d.begin(), d.end()) - d.begin() == std::min_element(d.begin(), d.end()) - d.begin());
    expect("max_element", std::max_element(par, d.begin(), d.end()) - d.begin() == std::max_element(d.begin(), d.end()) - d.begin());
    const auto r = std::minmax_element(par, d.begin(), d.end());
    const auto s = std::minmax_element(d.begin(), d.end());
    expect("minmax_element", r.first - d.begin() == s.first - d.begin() && r.second - d.begin() == s.second - d.begin());

    std::deque<int> p = d, q = d;
    expect("stable_partition", std::stable_partition(par, p.begin(), p.end(), negative) - p.begin() ==
                                   std::stable_partition(q.begin(), q.end(), negative) - q.begin());
    expect("stable_partition_order", p == q);
  }

  std::list<int> l{5, -3, 9, -3, 1, 9, -3};
  expect("list_min", std::distance(l.begin(), std::min_element(par, l.begin(), l.end())) == 1);
  expect("list_max", std::distance(l.begin(), std::max_element(par, l.begin(), l.end())) == 2);
  const auto m = std::minmax_element(par, l.begin(), l.end());
  expect("list_minmax", std::distance(l.begin(), m.first) == 1 && std::distance(l.begin(), m.second) == 5);
  const auto pivot = std::partition(par, l.begin(), l.end(), negative);
  expect("list_partition", std::distance(l.begin(), pivot) == 3 && std::all_of(l.begin(), pivot, negative));
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
