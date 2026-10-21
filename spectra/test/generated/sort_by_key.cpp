#pragma region case: sort_by_key
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `sort_by_key`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class K, class V> static bool run(const char *scenario, int n) {
  spectra_test_vector<K> keys(n), kref(n);
  spectra_test_vector<V> values(n), vref(n);
  for (int i = 0; i < n; ++i) {
    const int x = (i * 37 + 11) % n;
    keys[i] = K(x);
    values[i] = V(x * 2);
    kref[i] = K(i);
    vref[i] = V(i * 2);
  }
  const auto op = [] SPECTRA_TEST_FN(K a, K b) { return bool(a < b); };
  spectra::sort_by_key<K, V>(keys.data(), values.data(), n, op);
  const bool ok = sameSeq(keys, kref, n) && sameSeq(values, vref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "sort_by_key", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n200", 200);
  ok &= run<spectra_test_value8, spectra_test_value8>("w8-w8/n200", 200);
  ok &= run<spectra_test_value12, spectra_test_value12>("w12-w12/n200", 200);
  ok &= run<spectra_test_value4, spectra_test_value8>("w4-w8/n200", 200);
  ok &= run<spectra_test_value8, spectra_test_value12>("w8-w12/n200", 200);
  ok &= run<spectra_test_value12, spectra_test_value4>("w12-w4/n200", 200);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
