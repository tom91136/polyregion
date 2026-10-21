#pragma region case: inclusive_scan_by_key
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `inclusive_scan_by_key`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

#if defined(SPECTRA_NATIVE_ONEDPL)
template <class K> struct spectra_test_key_equal {
  SPECTRA_TEST_FN bool operator()(K a, K b) const { return bool(a == b); }
};
#endif
template <class K, class V> static bool run(const char *scenario, int n) {
  spectra_test_vector<K> keys(n);
  spectra_test_vector<V> vals(n), out(n, V(0)), ref(n);
  for (int i = 0; i < n; ++i) {
    keys[i] = K(i / 32);
    vals[i] = V(i % 5 + 1);
  }
  {
    V acc = vals[0];
    ref[0] = acc;
    for (int i = 1; i < n; ++i) {
      acc = (keys[i] == keys[i - 1]) ? V(acc + vals[i]) : vals[i];
      ref[i] = acc;
    }
  }
#if defined(SPECTRA_NATIVE_ONEDPL)
  const spectra_test_key_equal<K> eq{};
#else
  const auto eq = [] SPECTRA_TEST_FN(K a, K b) { return bool(a == b); };
#endif
  const auto op = [] SPECTRA_TEST_FN(V a, V b) { return V(a + b); };
  spectra::inclusive_scan_by_key<K, V>(keys.data(), vals.data(), out.data(), n, eq, op);
  const bool ok = vecNear(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inclusive_scan_by_key", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class K, class V> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<K> keys(n);
  spectra_test_vector<V> vals(n), out(n, V(0)), ref(n);
  for (int i = 0; i < n; ++i) {
    keys[i] = K(i / 32);
    vals[i] = V(i % 5 + 1);
  }
  {
    V acc = vals[0];
    ref[0] = acc;
    for (int i = 1; i < n; ++i) {
      acc = (keys[i] == keys[i - 1]) ? V(acc + vals[i]) : vals[i];
      ref[i] = acc;
    }
  }
  const bool live = n >= 0;
#if defined(SPECTRA_NATIVE_ONEDPL)
  const spectra_test_key_equal<K> eq{};
#else
  const auto eq = [live] SPECTRA_TEST_FN(K a, K b) { return live ? bool(a == b) : bool(a == b); };
#endif
  const auto op = [live] SPECTRA_TEST_FN(V a, V b) { return live ? V(a + b) : V(a + b); };
  spectra::inclusive_scan_by_key<K, V>(keys.data(), vals.data(), out.data(), n, eq, op);
  const bool ok = vecNear(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inclusive_scan_by_key", scenario, ok ? "OK" : "MISMATCH");
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
