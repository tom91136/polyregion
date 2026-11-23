#pragma region case: package_residency
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <polyregion/spectra.hpp>

#include "polyrt/rt.h"
#include "support.hpp"

namespace {
PolyrtStats stats() {
  PolyrtStats value{};
  if (!polyrt_stats(polyrt_context_current(), &value)) {
    std::fprintf(stderr, "no stats for the current context\n");
    std::exit(3);
  }
  return value;
}

const auto add = [] SPECTRA_TEST_FN(int a, int b) { return a + b; };
const auto twice = [] SPECTRA_TEST_FN(int x) { return x * 2; };

std::int32_t expectedSum(const spectra_test_vector<int> &in) {
  std::int32_t sum = 0;
  for (const auto x : in)
    sum += x;
  return sum;
}
} // namespace

int main() {
  bool ok = true;

  {
    constexpr int n = 1 << 24;
    spectra_test_vector<int> in(n);
    for (int i = 0; i < n; ++i)
      in[i] = i % 97;
    const auto expected = expectedSum(in);
    const auto base = stats();
    const int first = spectra::reduce<int>(in.data(), n, 0, add);
    const auto afterFirst = stats();
    const int second = spectra::reduce<int>(in.data(), n, 0, add);
    const auto afterSecond = stats();
    const auto firstBytes = afterFirst.h2dBytes - base.h2dBytes;
    const auto secondBytes = afterSecond.h2dBytes - afterFirst.h2dBytes;
    const bool values = first == expected && second == expected;
    const bool resident = secondBytes < (1u << 20);
    std::fprintf(stderr, "  residency h2d first=%llu second=%llu values=%s second=%s\n", static_cast<unsigned long long>(firstBytes),
                 static_cast<unsigned long long>(secondBytes), values ? "OK" : "MISMATCH", resident ? "OK" : "COPIED");
    ok &= values && resident;
  }

  {
    constexpr int n = 1 << 22;
    spectra_test_vector<int> in(n), out(n, 0);
    for (int i = 0; i < n; ++i)
      in[i] = i % 97;
    spectra::transform<int, int>(in.data(), out.data(), n, twice);
    bool deviceWrite = true;
    for (int i = 0; i < n; ++i)
      deviceWrite &= out[i] == in[i] * 2;
    std::fprintf(stderr, "  device write visible on host %s\n", deviceWrite ? "OK" : "MISMATCH");
    ok &= deviceWrite;

    for (int i = 0; i < n; i += 1024)
      in[i] += 1000;
    for (int i = 0; i < n; i += 4096)
      out[i] = -1;
    const auto expected = expectedSum(in);
    const int sum = spectra::reduce<int>(in.data(), n, 0, add);
    spectra::transform<int, int>(in.data(), out.data(), n, twice);
    bool hostWrite = sum == expected;
    for (int i = 0; i < n; ++i)
      hostWrite &= out[i] == in[i] * 2;
    std::fprintf(stderr, "  host write reaches device %s\n", hostWrite ? "OK" : "MISMATCH");
    ok &= hostWrite;
  }

  std::printf("%s", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}
