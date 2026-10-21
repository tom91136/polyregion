#pragma region case: set_intersection
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Self-checking Spectra API test for `set_intersection`: deterministic input -> device call -> host-oracle comparison.
#include <algorithm>
#include <cstdio>
#include <iterator>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(const char *scenario, int n) {
  const int na = n / 2, nb = n / 2;
  spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref;
  for (int i = 0; i < na; ++i)
    a[i] = T(2 * i);
  for (int i = 0; i < nb; ++i)
    b[i] = T(i);
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ref));
  const auto op = [] SPECTRA_TEST_FN(T a, T b) { return bool(a < b); };
  const int __r = spectra::set_intersection<T>(a.data(), na, b.data(), nb, out.data(), op);
  const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);
  std::fprintf(stderr, "  %-24s<%-18s> %s\n", "set_intersection", scenario, ok ? "OK" : "MISMATCH");
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
