#pragma region case: merge
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `merge`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  const int na = n / 2, nb = n / 2;
  spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref(na + nb);
  for (int i = 0; i < na; ++i)
    a[i] = T(2 * i);
  for (int i = 0; i < nb; ++i)
    b[i] = T(i);
  std::merge(a.begin(), a.end(), b.begin(), b.end(), ref.begin());
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  spectra::merge<T>(a.data(), na, b.data(), nb, out.data(), op);
  const bool ok = sameSeq(out, ref, na + nb);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "merge", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  const int na = n / 2, nb = n / 2;
  spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref(na + nb);
  for (int i = 0; i < na; ++i)
    a[i] = T(2 * i);
  for (int i = 0; i < nb; ++i)
    b[i] = T(i);
  std::merge(a.begin(), a.end(), b.begin(), b.end(), ref.begin());
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a < b) : bool(a < b); };
  spectra::merge<T>(a.data(), na, b.data(), nb, out.data(), op);
  const bool ok = sameSeq(out, ref, na + nb);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "merge", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n200", 200);
  ok &= run<spectra_test_value8>("w8/n200", 200);
  ok &= run<spectra_test_value12>("w12/n200", 200);
  ok &= run_captured<spectra_test_value4>("w4/n200/captured", 200);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
