#pragma region case: setops
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 4 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: setops-baseline
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
#include <functional>
#include <string>
#include <vector>

namespace {
std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}

template <class T, class Par, class Seq> void check_out(const char *name, const std::size_t capacity, Par par, Seq seq) {
  std::vector<T> ref(capacity);
  const auto count = static_cast<std::size_t>(seq(ref.begin()) - ref.begin());
  ref.resize(count);

  std::vector<T> exact(count);
  expect(name, par(exact.begin()) == exact.end() && exact == ref);

  const T sentinel = static_cast<T>(12345);
  std::vector<T> big(capacity + 5, sentinel);
  bool ok = par(big.begin()) == big.begin() + static_cast<std::ptrdiff_t>(count);
  ok = ok && std::equal(ref.begin(), ref.end(), big.begin());
  for (std::size_t i = count; i < big.size(); ++i)
    ok = ok && big[i] == sentinel;
  expect(name, ok);
}
}

int main() {
  const auto par = std::execution::par_unseq;
  const auto greater = [](const int x, const int y) { return x > y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1000u}) {
    const std::size_t m = n * 2 / 3 + 1;
    std::vector<int> a(n), b(m);
    for (std::size_t i = 0; i < n; ++i)
      a[i] = static_cast<int>((i * 7919) % 211) - 100;
    for (std::size_t i = 0; i < m; ++i)
      b[i] = static_cast<int>((i * 104729) % 223) - 100;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<int> da(a.rbegin(), a.rend()), db(b.rbegin(), b.rend());

    check_out<int>(
        "set_difference", n, [&](auto out) { return std::set_difference(par, a.begin(), a.end(), b.begin(), b.end(), out); },
        [&](auto out) { return std::set_difference(a.begin(), a.end(), b.begin(), b.end(), out); });
    check_out<int>(
        "set_intersection", std::min(n, m),
        [&](auto out) { return std::set_intersection(par, a.begin(), a.end(), b.begin(), b.end(), out); },
        [&](auto out) { return std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), out); });
    check_out<int>(
        "set_union", n + m, [&](auto out) { return std::set_union(par, a.begin(), a.end(), b.begin(), b.end(), out); },
        [&](auto out) { return std::set_union(a.begin(), a.end(), b.begin(), b.end(), out); });
    check_out<int>(
        "set_union_op", n + m, [&](auto out) { return std::set_union(par, da.begin(), da.end(), db.begin(), db.end(), out, greater); },
        [&](auto out) { return std::set_union(da.begin(), da.end(), db.begin(), db.end(), out, greater); });
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
