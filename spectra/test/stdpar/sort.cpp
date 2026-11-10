#pragma region case: sort
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 6 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: sort-baseline
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
} // namespace

int main() {
  const auto par = std::execution::par_unseq;
  const auto descending = [](const int x, const int y) { return x > y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i)
      a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    std::vector<int> asc = a, desc = a;
    std::sort(asc.begin(), asc.end());
    std::sort(desc.begin(), desc.end(), descending);

    std::vector<int> p = a;
    std::sort(par, p.begin(), p.end());
    expect("sort", p == asc);

    p = a;
    std::sort(par, p.begin(), p.end(), descending);
    expect("sort_op", p == desc);

    p = a;
    std::stable_sort(par, p.begin(), p.end());
    expect("stable_sort", p == asc);

    p = a;
    std::stable_sort(par, p.begin(), p.end(), descending);
    expect("stable_sort_op", p == desc);

    std::vector<int> hx(a.begin(), a.begin() + n / 2), hy(a.begin() + n / 2, a.end());
    std::sort(hx.begin(), hx.end());
    std::sort(hy.begin(), hy.end());
    std::vector<int> m(n);
    std::merge(par, hx.begin(), hx.end(), hy.begin(), hy.end(), m.begin());
    expect("merge", m == asc);

    std::vector<int> dx(hx.rbegin(), hx.rend()), dy(hy.rbegin(), hy.rend());
    std::merge(par, dx.begin(), dx.end(), dy.begin(), dy.end(), m.begin(), descending);
    expect("merge_op", m == desc);
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
