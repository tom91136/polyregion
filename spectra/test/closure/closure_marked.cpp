#pragma region case: closure_marked
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdio>

#include <polyregion/spectra.hpp>

#include "support.hpp"

template <class T> static bool run(int n) {
  spectra_test_vector<T> in(n), out(n, T(0)), marks(n, T(0));
  for (int i = 0; i < n; ++i)
    in[i] = T((i * 7) % n);
  T *marked = marks.data();
  const auto op = [marked] SPECTRA_TEST_FN(T x) {
    marked[int(x)] = T(1);
    return T(x * 2);
  };
  spectra::transform<T, T>(in.data(), out.data(), n, op);
  bool ok = true;
  for (int i = 0; i < n; ++i)
    ok &= out[i] == T(in[i] * 2) && marks[int(in[i])] == T(1);
  std::fprintf(stderr, "  closure_marked<n=%d> %s\n", n, ok ? "OK" : "MISMATCH");
  return ok;
}

int main() {
  bool ok = true;
  ok &= run<spectra_test_value4>(256);
  ok &= run<spectra_test_value8>(1031);
  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
