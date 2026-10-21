#pragma region case: copy_if
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `copy_if`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  spectra_test_vector<T> in(n), out(n, T(-1)), ref;
  for (int i = 0; i < n; ++i)
    in[i] = T((i * 37 + 11) % n);
  for (int i = 0; i < n; ++i)
    if ((long long)(in[i]) % 2 == 0) ref.push_back(in[i]);
  const auto op = [] SPECTRA_TEST_FN(T x) { return bool((long long)(x) % 2 == 0); };
  const int __r = spectra::copy_if<T>(in.data(), n, out.data(), op);
  const bool ok = __r == (int)ref.size() && __r >= 0 && __r <= n && sameSeq(out, ref, __r);
  if (!ok) {
    std::printf(" result=%d expected=%d out:", __r, (int)ref.size());
    for (int i = 0; i < 8 && i < n; ++i)
      std::printf(" %lld", (long long)out[i]);
    std::printf(" ref:");
    for (int i = 0; i < 8 && i < (int)ref.size(); ++i)
      std::printf(" %lld", (long long)ref[i]);
    std::printf("\n");
  }
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "copy_if", scenario, ok ? "OK" : "MISMATCH");
  return ok;
}

template <class T> static bool run_captured(const char *scenario, int n) {
  spectra_test_vector<T> in(n), out(n, T(-1)), ref;
  for (int i = 0; i < n; ++i)
    in[i] = T((i * 37 + 11) % n);
  for (int i = 0; i < n; ++i)
    if ((long long)(in[i]) % 2 == 0) ref.push_back(in[i]);
  const bool live = n >= 0;
  const auto op = [live] SPECTRA_TEST_FN(T x) { return live ? bool((long long)(x) % 2 == 0) : bool((long long)(x) % 2 == 0); };
  const int __r = spectra::copy_if<T>(in.data(), n, out.data(), op);
  const bool ok = __r == (int)ref.size() && __r >= 0 && __r <= n && sameSeq(out, ref, __r);
  if (!ok) {
    std::printf(" result=%d expected=%d out:", __r, (int)ref.size());
    for (int i = 0; i < 8 && i < n; ++i)
      std::printf(" %lld", (long long)out[i]);
    std::printf(" ref:");
    for (int i = 0; i < 8 && i < (int)ref.size(); ++i)
      std::printf(" %lld", (long long)ref[i]);
    std::printf("\n");
  }
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "copy_if", scenario, ok ? "OK" : "MISMATCH");
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
