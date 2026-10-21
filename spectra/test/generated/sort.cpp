#pragma region case: sort
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `sort`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref;
  for (int i = 0; i < n; ++i)
    data[i] = T((i * 37 + 11) % n);
  ref = data;
  std::sort(ref.begin(), ref.end());
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  spectra::sort<T>(data.data(), n, op);
  const bool ok = sameSeq(data, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "sort", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref;
  for (int i = 0; i < n; ++i)
    data[i] = T((i * 37 + 11) % n);
  ref = data;
  std::sort(ref.begin(), ref.end());
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a < b) : bool(a < b); };
  spectra::sort<T>(data.data(), n, op);
  const bool ok = sameSeq(data, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "sort", scenario, ok ? "OK" : "MISMATCH");
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
