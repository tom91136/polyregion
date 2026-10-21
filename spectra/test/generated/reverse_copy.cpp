#pragma region case: reverse_copy
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `reverse_copy`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n), out(n, T(-1)), ref(n);
  for (int i = 0; i < n; ++i)
    in[i] = T((i * 17 + 3) % 97);
  for (int i = 0; i < n; ++i)
    ref[i] = in[n - 1 - i];

  spectra::reverse_copy<T>(in.data(), n, out.data());
  const bool ok = sameSeq(out, ref, n);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "reverse_copy", scenario, ok ? "OK" : "MISMATCH");
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
