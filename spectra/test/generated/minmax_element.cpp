#pragma region case: minmax_element
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `minmax_element`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n), min_out(1, T(-1)), max_out(1, T(-1));
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 11);
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  spectra::minmax_element<T>(in.data(), n, min_out.data(), max_out.data(), op);
  const bool ok = near_(min_out[0], n == 0 ? T{} : T(0)) && near_(max_out[0], n == 0 ? T{} : T(10));
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "minmax_element", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n300", 300);
  ok &= run<spectra_test_value8>("w8/n300", 300);
  ok &= run<spectra_test_value12>("w12/n300", 300);
  ok &= run<spectra_test_value4>("w4/n0", 0);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
