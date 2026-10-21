#pragma region case: transform
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `transform`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T, class U> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  spectra_test_vector<U> out(n, U(0)), ref(n);
  for (int i = 0; i < n; ++i) {
    in[i] = T(i % 7);
    ref[i] = U(in[i] * T(2) + T(1));
  }
  const auto op = [] SPECTRA_TEST_FN(T x) { return U(x * 2 + 1); };
  spectra::transform<T, U>(in.data(), out.data(), n, op);
  const bool ok = sameSeq(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "transform", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value8>("w8-w8/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value12>("w12-w12/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value8>("w4-w8/n256", 256);
  ok &= run<spectra_test_value8, spectra_test_value12>("w8-w12/n256", 256);
  ok &= run<spectra_test_value12, spectra_test_value4>("w12-w4/n256", 256);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n31", 31);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n32", 32);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n33", 33);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n255", 255);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n257", 257);
  ok &= run<spectra_test_value4, spectra_test_value4>("w4-w4/n777", 777);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
