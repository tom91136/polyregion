#pragma region case: search_n
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `search_n`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 5);
  in[40] = in[41] = in[42] = T(9);
  const int count = 3;
  const T value = T(9);
  int ref = n - count + 1;
  for (int i = 0; i <= n - count; ++i) {
    bool f = true;
    for (int k = 0; k < count; ++k)
      if (!(in[i + k] == value)) {
        f = false;
        break;
      }
    if (f) {
      ref = i;
      break;
    }
  }
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a == b); };
  const int __r = spectra::search_n<T>(in.data(), n, count, value, op);
  const bool ok = __r == ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "search_n", scenario, ok ? "OK" : "MISMATCH");
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
