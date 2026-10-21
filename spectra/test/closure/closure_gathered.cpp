#pragma region case: closure_gathered
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(int n, int bias) {
  spectra_test_vector<T> in(n), table(8), out(n, T(0));
  for (int i = 0; i < 8; ++i)
    table[i] = T(i * 3);
  for (int i = 0; i < n; ++i)
    in[i] = T(i % 8);
  const T *lookup = table.data();
  const auto op = [lookup, bias] SPECTRA_TEST_FN(T x) { return T(lookup[int(x)] + T(bias)); };
  spectra::transform<T, T>(in.data(), out.data(), n, op);
  bool ok = true;
  for (int i = 0; i < n; ++i)
    ok &= out[i] == T(table[int(in[i])] + T(bias));
  std::fprintf(stderr, "  closure_gathered<n=%d,bias=%d> %s\n", n, bias, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>(300, 2);
  ok &= run<spectra_test_value8>(777, 5);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
