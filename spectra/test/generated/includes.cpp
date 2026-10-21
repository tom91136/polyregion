#pragma region case: includes
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `includes`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  const int na = n, nb = n / 2;
  spectra_test_vector<T> a(na), b(nb);
  for (int i = 0; i < na; ++i)
    a[i] = T(i);
  for (int j = 0; j < nb; ++j)
    b[j] = T(2 * j);
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  const bool __r = spectra::includes<T>(a.data(), na, b.data(), nb, op);
  const bool ok = (__r == true);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "includes", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n300", 300);
  ok &= run<spectra_test_value8>("w8/n300", 300);
  ok &= run<spectra_test_value12>("w12/n300", 300);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
