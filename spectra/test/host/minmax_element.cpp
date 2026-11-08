#pragma region case: minmax_element
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `minmax_element`: deterministic input -> spectra::host call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra/host.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  spectra_test_vector<std::int32_t> min_out(1, -1), max_out(1, -1);
  for (int i = 0; i < n; ++i)
    in[i] = T((i + 3) % 11);
  const auto extrema = std::minmax_element(in.begin(), in.end());
  const int min_ref = int(extrema.first - in.begin());
  const int max_ref = int(extrema.second - in.begin());
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  spectra::host::minmax_element<T>(in.data(), n, min_out.data(), max_out.data(), op);
  const bool ok = min_out[0] == min_ref && max_out[0] == max_ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "minmax_element", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> in(n);
  spectra_test_vector<std::int32_t> min_out(1, -1), max_out(1, -1);
  for (int i = 0; i < n; ++i)
    in[i] = T((i + 3) % 11);
  const auto extrema = std::minmax_element(in.begin(), in.end());
  const int min_ref = int(extrema.first - in.begin());
  const int max_ref = int(extrema.second - in.begin());
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T a, T b) { return live ? bool(a < b) : bool(a < b); };
  spectra::host::minmax_element<T>(in.data(), n, min_out.data(), max_out.data(), op);
  const bool ok = min_out[0] == min_ref && max_out[0] == max_ref;
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "minmax_element", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>("w4/n300", 300);
  ok &= run<spectra_test_value8>("w8/n300", 300);
  ok &= run<spectra_test_value12>("w12/n300", 300);
  ok &= run<spectra_test_value4>("w4/n0", 0);
  ok &= run_captured<spectra_test_value4>("w4/n300/captured", 300);
  ok &= run_captured<spectra_test_value4>("w4/n0/captured", 0);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
