#pragma region case: is_sorted_until
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `is_sorted_until`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> data(n);
  for (int i = 0; i < n; ++i)
    data[i] = T(i);
  int ref = n;
  for (int i = 1; i < n; ++i)
    if (data[i] < data[i - 1]) {
      ref = i;
      break;
    }
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  const int __r = spectra::is_sorted_until<T>(data.data(), n, op);
  const bool ok = __r == ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "is_sorted_until", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> data(n);
  for (int i = 0; i < n; ++i)
    data[i] = T(i);
  int ref = n;
  for (int i = 1; i < n; ++i)
    if (data[i] < data[i - 1]) {
      ref = i;
      break;
    }
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a < b) : bool(a < b); };
  const int __r = spectra::is_sorted_until<T>(data.data(), n, op);
  const bool ok = __r == ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "is_sorted_until", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n300", 300);
  ok &= run<spectra_test_value8>("w8/n300", 300);
  ok &= run<spectra_test_value12>("w12/n300", 300);
  ok &= run_captured<spectra_test_value4>("w4/n300/captured", 300);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
