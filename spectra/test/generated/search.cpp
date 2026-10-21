#pragma region case: search
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `search`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 11);
  const int m = 5;
  spectra_test_vector<T> sub(in.begin() + 10, in.begin() + 10 + m);
  int ref = n - m + 1;
  for (int i = 0; i <= n - m; ++i) {
    bool f = true;
    for (int k = 0; k < m; ++k)
      if (!(in[i + k] == sub[k])) {
        f = false;
        break;
      }
    if (f) {
      ref = i;
      break;
    }
  }
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a == b); };
  const int __r = spectra::search<T>(in.data(), n, sub.data(), m, op);
  const bool ok = __r == ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "search", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n300", 300);
  ok &= run<spectra_test_value8>("w8/n300", 300);
  ok &= run<spectra_test_value12>("w12/n300", 300);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
