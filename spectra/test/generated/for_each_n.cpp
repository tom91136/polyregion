#pragma region case: for_each_n
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `for_each_n`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref(n);
  for (int i = 0; i < n; ++i) {
    data[i] = T((i * 37 + 11) % n);
    ref[i] = T(data[i] + 1);
  }
  const auto op = [] SPECTRA_TEST_FN(T x) { return T(x = T(x + 1)); };
  spectra::for_each_n<T>(data.data(), n, op);
  const bool ok = sameSeq(data, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "for_each_n", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref(n);
  for (int i = 0; i < n; ++i) {
    data[i] = T((i * 37 + 11) % n);
    ref[i] = T(data[i] + 1);
  }
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T x) { return live ? T(x = T(x + 1)) : T(x = T(x + 1)); };
  spectra::for_each_n<T>(data.data(), n, op);
  const bool ok = sameSeq(data, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "for_each_n", scenario, ok ? "OK" : "MISMATCH");
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
