#pragma region case: compact
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 12 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: compact-baseline
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

template <class T, class Par, class Seq> void check_out(const char *name, const std::size_t n, Par par, Seq seq) {
  std::vector<T> ref(n);
  const auto count = static_cast<std::size_t>(seq(ref.begin()) - ref.begin());
  ref.resize(count);

  std::vector<T> exact(count);
  expect(name, par(exact.begin()) == exact.end() && exact == ref);

  const T sentinel = static_cast<T>(12345);
  std::vector<T> big(n + 5, sentinel);
  bool ok = par(big.begin()) == big.begin() + static_cast<std::ptrdiff_t>(count);
  ok = ok && std::equal(ref.begin(), ref.end(), big.begin());
  for (std::size_t i = count; i < big.size(); ++i)
    ok = ok && big[i] == sentinel;
  expect(name, ok);
}

template <class T, class Par, class Seq> void check_inplace(const char *name, const std::vector<T> &a, Par par, Seq seq) {
  std::vector<T> ref = a, p = a;
  const auto count = seq(ref) - ref.begin();
  const auto got = par(p) - p.begin();
  expect(name, got == count && std::equal(ref.begin(), ref.begin() + count, p.begin()));
}
}

int main() {
  const auto par = std::execution::par_unseq;
  const auto positive = [](const int x) { return x > 0; };
  const auto never = [](const int) { return false; };
  const auto always = [](const int) { return true; };
  const auto same = [](const int x, const int y) { return x == y; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 16}) {
    std::vector<int> a(n);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = static_cast<int>(((i / 3) * 7919) % 101) - 50;
      d[i] = a[i] * 0.5;
    }
    const int victim = n ? a[n / 2] : 0;
    const auto dpositive = [](const double x) { return x > 0; };

    check_out<int>(
        "copy_if", n, [&](auto out) { return std::copy_if(par, a.begin(), a.end(), out, positive); },
        [&](auto out) { return std::copy_if(a.begin(), a.end(), out, positive); });
    check_out<int>(
        "copy_if_none", n, [&](auto out) { return std::copy_if(par, a.begin(), a.end(), out, never); },
        [&](auto out) { return std::copy_if(a.begin(), a.end(), out, never); });
    check_out<double>(
        "copy_if_double", n, [&](auto out) { return std::copy_if(par, d.begin(), d.end(), out, dpositive); },
        [&](auto out) { return std::copy_if(d.begin(), d.end(), out, dpositive); });
    check_out<int>(
        "remove_copy", n, [&](auto out) { return std::remove_copy(par, a.begin(), a.end(), out, victim); },
        [&](auto out) { return std::remove_copy(a.begin(), a.end(), out, victim); });
    check_out<int>(
        "remove_copy_if", n, [&](auto out) { return std::remove_copy_if(par, a.begin(), a.end(), out, positive); },
        [&](auto out) { return std::remove_copy_if(a.begin(), a.end(), out, positive); });
    check_out<int>(
        "remove_copy_if_all", n, [&](auto out) { return std::remove_copy_if(par, a.begin(), a.end(), out, always); },
        [&](auto out) { return std::remove_copy_if(a.begin(), a.end(), out, always); });
    check_out<int>(
        "unique_copy", n, [&](auto out) { return std::unique_copy(par, a.begin(), a.end(), out); },
        [&](auto out) { return std::unique_copy(a.begin(), a.end(), out); });
    check_out<int>(
        "unique_copy_op", n, [&](auto out) { return std::unique_copy(par, a.begin(), a.end(), out, same); },
        [&](auto out) { return std::unique_copy(a.begin(), a.end(), out, same); });

    check_inplace(
        "remove", a, [&](auto &v) { return std::remove(par, v.begin(), v.end(), victim); },
        [&](auto &v) { return std::remove(v.begin(), v.end(), victim); });
    check_inplace(
        "remove_if", a, [&](auto &v) { return std::remove_if(par, v.begin(), v.end(), positive); },
        [&](auto &v) { return std::remove_if(v.begin(), v.end(), positive); });
    check_inplace(
        "unique", a, [&](auto &v) { return std::unique(par, v.begin(), v.end()); },
        [&](auto &v) { return std::unique(v.begin(), v.end()); });
    check_inplace(
        "unique_op", a, [&](auto &v) { return std::unique(par, v.begin(), v.end(), same); },
        [&](auto &v) { return std::unique(v.begin(), v.end(), same); });
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
