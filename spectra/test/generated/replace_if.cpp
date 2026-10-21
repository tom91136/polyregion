#pragma region case: replace_if
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `replace_if`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> data(n), ref(n);
  for (int i = 0; i < n; ++i)
    data[i] = T((i * 37 + 11) % n);
  const T new_value = T(-1);
  for (int i = 0; i < n; ++i)
    ref[i] = (data[i] >= T(100)) ? new_value : data[i];
  const auto op = [] SPECTRA_TEST_FN(T x) { return bool(x >= T(100)); };
  spectra::replace_if<T>(data.data(), n, new_value, op);
  const bool ok = sameSeq(data, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "replace_if", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n200", 200);
  ok &= run<spectra_test_value8>("w8/n200", 200);
  ok &= run<spectra_test_value12>("w12/n200", 200);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
