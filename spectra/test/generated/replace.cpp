#pragma region case: replace
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `replace`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> io(n), ref(n);
  for (int i = 0; i < n; ++i)
    io[i] = T((i * 17 + 3) % 97);
  const T oldv = T(5), newv = T(88);
  for (int i = 0; i < n; ++i)
    ref[i] = (io[i] == oldv) ? newv : io[i];
  const auto eq = [] SPECTRA_TEST_FN(T a, T b) { return bool(a == b); };
  spectra::replace<T>(io.data(), n, oldv, newv, eq);
  const bool ok = sameSeq(io, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "replace", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> io(n), ref(n);
  for (int i = 0; i < n; ++i)
    io[i] = T((i * 17 + 3) % 97);
  const T oldv = T(5), newv = T(88);
  for (int i = 0; i < n; ++i)
    ref[i] = (io[i] == oldv) ? newv : io[i];
  const bool live = n >= 0;
  const auto eq = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a == b) : bool(a == b); };
  spectra::replace<T>(io.data(), n, oldv, newv, eq);
  const bool ok = sameSeq(io, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "replace", scenario, ok ? "OK" : "MISMATCH");
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
