#pragma region case: unique
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `unique`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref;
  for (int i = 0; i < n; ++i)
    data[i] = T(i / 4);
  for (int i = 0; i < n; ++i)
    if (ref.empty() || ref.back() != data[i]) ref.push_back(data[i]);
  const auto eq = [] SPECTRA_TEST_FN(T a, T b) { return bool(a == b); };
  const int __r = spectra::unique<T>(data.data(), n, eq);
  const bool ok = __r == (int)ref.size() && sameSeq(data, ref, __r);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "unique", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref;
  for (int i = 0; i < n; ++i)
    data[i] = T(i / 4);
  for (int i = 0; i < n; ++i)
    if (ref.empty() || ref.back() != data[i]) ref.push_back(data[i]);
  const bool live = n >= 0;
  const auto eq = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a == b) : bool(a == b); };
  const int __r = spectra::unique<T>(data.data(), n, eq);
  const bool ok = __r == (int)ref.size() && sameSeq(data, ref, __r);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "unique", scenario, ok ? "OK" : "MISMATCH");
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
