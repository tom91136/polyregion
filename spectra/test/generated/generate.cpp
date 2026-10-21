#pragma region case: generate
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `generate`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> out(n, T(-1));
  const auto op = [] SPECTRA_TEST_FN() { return T(7); };
  spectra::generate<T>(out.data(), n, op);
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (out[i] != T(7)) ok = false;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "generate", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> out(n, T(-1));
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN() { return live ? T(7) : T(7); };
  spectra::generate<T>(out.data(), n, op);
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (out[i] != T(7)) ok = false;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "generate", scenario, ok ? "OK" : "MISMATCH");
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
