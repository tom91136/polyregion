#pragma region case: inner_product
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `inner_product`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T, class U, class V> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> a(n);
  spectra_test_vector<U> b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = T(i % 7);
    b[i] = U((i * 3) % 5);
  }
  const V init = V(0);
  V ref = init;
  for (int i = 0; i < n; ++i)
    ref = V(ref + V(a[i]) * V(b[i]));
  const auto op_reduce = [] SPECTRA_TEST_FN(V a, V b) { return V(a + b); };
  const auto op_product = [] SPECTRA_TEST_FN(T a, U b) { return V(a * b); };
  const V __r = spectra::inner_product<T, U, V>(a.data(), b.data(), n, init, op_reduce, op_product);
  const bool ok = near_(__r, ref);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inner_product", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T, class U, class V> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> a(n);
  spectra_test_vector<U> b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = T(i % 7);
    b[i] = U((i * 3) % 5);
  }
  const V init = V(0);
  V ref = init;
  for (int i = 0; i < n; ++i)
    ref = V(ref + V(a[i]) * V(b[i]));
  const bool live = n >= 0;
  const auto op_reduce = [live] SPECTRA_TEST_FN(V a, V b) { return live ? V(a + b) : V(a + b); };
  const auto op_product = [live] SPECTRA_TEST_FN(T a, U b) { return live ? V(a * b) : V(a * b); };
  const V __r = spectra::inner_product<T, U, V>(a.data(), b.data(), n, init, op_reduce, op_product);
  const bool ok = near_(__r, ref);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inner_product", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value8, spectra_test_value8>("w4-w8-w8/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value12, spectra_test_value12>("w4-w12-w12/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value4, spectra_test_value8>("w8-w4-w8/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value8, spectra_test_value12>("w8-w8-w12/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value12, spectra_test_value4>("w8-w12-w4/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value4, spectra_test_value12>("w12-w4-w12/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value8, spectra_test_value4>("w12-w8-w4/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value12, spectra_test_value8>("w12-w12-w8/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n31", 31);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n32", 32);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n33", 33);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n255", 255);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n257", 257);
  ok &= run<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n777", 777);
  ok &= run_captured<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n256/captured", 256);
  ok &= run_captured<spectra_test_value4, spectra_test_value4, spectra_test_value4>("w4-w4-w4/n777/captured", 777);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
