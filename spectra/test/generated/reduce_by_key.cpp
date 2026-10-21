#pragma region case: reduce_by_key
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `reduce_by_key`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class K, class V> static bool run(const char *scenario, int n) {
  const int nseg = n / 32;
  spectra_test_vector<K> keys(n), kout(n, K(-1)), kref(nseg);
  spectra_test_vector<V> vals(n), vout(n, V(-1)), vref(nseg);
  for (int i = 0; i < n; ++i) {
    keys[i] = K(i / 32);
    vals[i] = V(i % 5 + 1);
  }
  for (int s = 0; s < nseg; ++s) {
    kref[s] = K(s);
    V acc = V(0);
    for (int i = 0; i < 32; ++i)
      acc = V(acc + vals[s * 32 + i]);
    vref[s] = acc;
  }
  const auto eq = [] SPECTRA_TEST_FN(K a, K b) { return bool(a == b); };
  const auto op = [] SPECTRA_TEST_FN(V a, V b) { return V(a + b); };
  const int __r = spectra::reduce_by_key<K, V>(keys.data(), vals.data(), kout.data(), vout.data(), n, eq, op);
  const bool ok = vecNear(kout, kref, nseg) && vecNear(vout, vref, nseg);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "reduce_by_key", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class K, class V> static bool run_captured(const char *scenario, int n) {
  const int nseg = n / 32;
  spectra_test_vector<K> keys(n), kout(n, K(-1)), kref(nseg);
  spectra_test_vector<V> vals(n), vout(n, V(-1)), vref(nseg);
  for (int i = 0; i < n; ++i) {
    keys[i] = K(i / 32);
    vals[i] = V(i % 5 + 1);
  }
  for (int s = 0; s < nseg; ++s) {
    kref[s] = K(s);
    V acc = V(0);
    for (int i = 0; i < 32; ++i)
      acc = V(acc + vals[s * 32 + i]);
    vref[s] = acc;
  }
  const bool live = n >= 0;
  const auto eq = [live] SPECTRA_TEST_FN(K a, K b) { return live ? bool(a == b) : bool(a == b); };
  const auto op = [live] SPECTRA_TEST_FN(V a, V b) { return live ? V(a + b) : V(a + b); };
  const int __r = spectra::reduce_by_key<K, V>(keys.data(), vals.data(), kout.data(), vout.data(), n, eq, op);
  const bool ok = vecNear(kout, kref, nseg) && vecNear(vout, vref, nseg);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "reduce_by_key", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value8>("w8-w8/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value12>("w12-w12/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value8>("w4-w8/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value12>("w8-w12/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value4>("w12-w4/n256", 256);
  ok &= run_captured<spectra_test_value4, spectra_test_value4>("w4-w4/n256/captured", 256);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
