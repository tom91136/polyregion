#pragma region case: inclusive_scan
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `inclusive_scan`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 7);
  {
    T acc = in[0];
    ref[0] = acc;
    for (int i = 1; i < n; ++i) {
      acc = T(acc + in[i]);
      ref[i] = acc;
    }
  }
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return T(a + b); };
  spectra::inclusive_scan<T>(in.data(), out.data(), n, op);
  const bool ok = vecNear(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inclusive_scan", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 7);
  {
    T acc = in[0];
    ref[0] = acc;
    for (int i = 1; i < n; ++i) {
      acc = T(acc + in[i]);
      ref[i] = acc;
    }
  }
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T a, T b) { return live ? T(a + b) : T(a + b); };
  spectra::inclusive_scan<T>(in.data(), out.data(), n, op);
  const bool ok = vecNear(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "inclusive_scan", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n256", 256);
  ok &= run<spectra_test_value8>("w8/n256", 256);
  ok &= run<spectra_test_value12>("w12/n256", 256);
  ok &= run<spectra_test_value4>("w4/n31", 31);
  ok &= run<spectra_test_value4>("w4/n32", 32);
  ok &= run<spectra_test_value4>("w4/n33", 33);
  ok &= run<spectra_test_value4>("w4/n255", 255);
  ok &= run<spectra_test_value4>("w4/n257", 257);
  ok &= run<spectra_test_value4>("w4/n777", 777);
  ok &= run<spectra_test_value4>("w4/n262144", 262144);
  ok &= run_captured<spectra_test_value4>("w4/n256/captured", 256);
  ok &= run_captured<spectra_test_value4>("w4/n262144/captured", 262144);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
