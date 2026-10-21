#pragma region case: count
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `count`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  for (int i = 0; i < n; ++i)
    in[i] = T((i * 17 + 3) % 97);
  const T value = T(3);
  int ref = 0;
  for (int i = 0; i < n; ++i)
    if (in[i] == value) ++ref;
  const auto eq = [] SPECTRA_TEST_FN(T a, T b) { return bool(a == b); };
  const int __r = spectra::count<T>(in.data(), n, value, eq);
  const bool ok = __r == ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "count", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n256", 256);
  ok &= run<spectra_test_value8>("w8/n256", 256);
  ok &= run<spectra_test_value12>("w12/n256", 256);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
