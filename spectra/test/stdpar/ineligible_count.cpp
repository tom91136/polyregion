#pragma region case: deque-list-count
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 3 bound to spectra
#pragma region do: {output}
#pragma region requires: pass
#pragma region do: POLYRT_DEBUG=2 {output} multi
#pragma region stderr-count: 0 grid=(
#pragma region requires: pass
#pragma region do: POLYRT_DEBUG=2 {output} list
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

template <class C, class Par, class Seq> void check_out(const char *name, const std::size_t capacity, Par par, Seq seq) {
  C ref(capacity), got(capacity);
  const auto expected = std::distance(ref.begin(), seq(ref.begin()));
  const auto actual = std::distance(got.begin(), par(got.begin()));
  expect(name, actual == expected && std::equal(ref.begin(), std::next(ref.begin(), expected), got.begin()));
}

template <class C, class Par, class Seq> void check_inplace(const char *name, const C &a, Par par, Seq seq) {
  C ref = a, got = a;
  const auto expected = std::distance(ref.begin(), seq(ref));
  const auto actual = std::distance(got.begin(), par(got));
  expect(name, actual == expected && std::equal(ref.begin(), std::next(ref.begin(), expected), got.begin()));
}

template <class C> void run(const std::size_t n) {
  const auto par = std::execution::par_unseq;
  const auto positive = [](const int x) { return x > 0; };
  const std::size_t m = n / 2 + 1;
  std::vector<int> va(n), vb(m);
  for (std::size_t i = 0; i < n; ++i)
    va[i] = static_cast<int>(((i / 3) * 7919) % 101) - 50;
  for (std::size_t i = 0; i < m; ++i)
    vb[i] = static_cast<int>((i * 104729) % 223) - 100;
  const int victim = n ? va[n / 2] : 0;
  std::sort(va.begin(), va.end());
  std::sort(vb.begin(), vb.end());
  const C a(va.begin(), va.end()), b(vb.begin(), vb.end());

  check_out<C>(
      "copy_if", n, [&](auto out) { return std::copy_if(par, a.begin(), a.end(), out, positive); },
      [&](auto out) { return std::copy_if(a.begin(), a.end(), out, positive); });
  check_out<C>(
      "remove_copy", n, [&](auto out) { return std::remove_copy(par, a.begin(), a.end(), out, victim); },
      [&](auto out) { return std::remove_copy(a.begin(), a.end(), out, victim); });
  check_out<C>(
      "remove_copy_if", n, [&](auto out) { return std::remove_copy_if(par, a.begin(), a.end(), out, positive); },
      [&](auto out) { return std::remove_copy_if(a.begin(), a.end(), out, positive); });
  check_out<C>(
      "unique_copy", n, [&](auto out) { return std::unique_copy(par, a.begin(), a.end(), out); },
      [&](auto out) { return std::unique_copy(a.begin(), a.end(), out); });
  check_out<C>(
      "set_difference", n, [&](auto out) { return std::set_difference(par, a.begin(), a.end(), b.begin(), b.end(), out); },
      [&](auto out) { return std::set_difference(a.begin(), a.end(), b.begin(), b.end(), out); });
  check_out<C>(
      "set_intersection", std::min(n, m), [&](auto out) { return std::set_intersection(par, a.begin(), a.end(), b.begin(), b.end(), out); },
      [&](auto out) { return std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), out); });
  check_out<C>(
      "set_union", n + m, [&](auto out) { return std::set_union(par, a.begin(), a.end(), b.begin(), b.end(), out); },
      [&](auto out) { return std::set_union(a.begin(), a.end(), b.begin(), b.end(), out); });
  check_inplace(
      "remove", a, [&](auto &c) { return std::remove(par, c.begin(), c.end(), victim); },
      [&](auto &c) { return std::remove(c.begin(), c.end(), victim); });
  check_inplace(
      "remove_if", a, [&](auto &c) { return std::remove_if(par, c.begin(), c.end(), positive); },
      [&](auto &c) { return std::remove_if(c.begin(), c.end(), positive); });
  check_inplace(
      "unique", a, [&](auto &c) { return std::unique(par, c.begin(), c.end()); }, [&](auto &c) { return std::unique(c.begin(), c.end()); });

  std::vector<int> pushed, expected;
  std::copy_if(par, a.begin(), a.end(), std::back_inserter(pushed), positive);
  std::copy_if(a.begin(), a.end(), std::back_inserter(expected), positive);
  expect("back_inserter", pushed == expected);
}
} // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "list") run<std::list<int>>(300);
  else if (mode == "multi") run<std::deque<int>>(10000);
  else
    for (const std::size_t n : {0u, 1u, 31u, 32u, 33u})
      run<std::deque<int>>(n);
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
