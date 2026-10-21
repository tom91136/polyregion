#pragma region case: sequence
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `sequence`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> out(n, T(-1)), ref(n);
  const T init = T(0), step = T(1);
  for (int i = 0; i < n; ++i)
    ref[i] = T(init + T(i) * step);
  const auto value_at = [] SPECTRA_TEST_FN(T a, T b, int i) { return T(a + T(i) * b); };
  spectra::sequence<T>(out.data(), n, init, step, value_at);
  const bool ok = sameSeq(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "sequence", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> out(n, T(-1)), ref(n);
  const T init = T(0), step = T(1);
  for (int i = 0; i < n; ++i)
    ref[i] = T(init + T(i) * step);
  const bool live = n >= 0;
  const auto value_at = [live] SPECTRA_TEST_FN(T a, T b, int i) { return live ? T(a + T(i) * b) : T(a + T(i) * b); };
  spectra::sequence<T>(out.data(), n, init, step, value_at);
  const bool ok = sameSeq(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "sequence", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n256", 256);
  ok &= run<spectra_test_value8>("w8/n256", 256);
  ok &= run<spectra_test_value12>("w12/n256", 256);
  ok &= run_captured<spectra_test_value4>("w4/n256/captured", 256);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
