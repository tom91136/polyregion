#pragma region case: signum-builtins
#pragma region offload-only
#pragma region targets: host cuda opencl1_1
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: signum ok

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <execution>
#include <limits>
#include <vector>

#include "polystl/intrinsics.h"

template <typename T> static bool same(T a, T b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

template <typename T> static T expected(T x) { return std::isnan(x) || x == T(0) ? x : std::copysign(T(1), x); }

int main() {
  int failures = 0;
  const float finf = std::numeric_limits<float>::infinity();
  const std::vector<float> fxs{
      0.f, -0.f, -1.f, 1.f, -finf, finf, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min()};
  std::vector<float> fout(fxs.size(), 0.f);
  std::vector<int> fidx(fxs.size());
  for (size_t i = 0; i < fidx.size(); ++i)
    fidx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, fidx.begin(), fidx.end(),
                [in = fxs.data(), out = fout.data()](int i) { out[i] = __polyregion_gpu_signum(in[i]); });
  for (size_t i = 0; i < fxs.size(); ++i)
    if (!same(fout[i], expected(fxs[i]))) {
      printf("f32 signum(%.9g): device %.9g host %.9g\n", fxs[i], fout[i], expected(fxs[i]));
      ++failures;
    }

  const double dinf = std::numeric_limits<double>::infinity();
  const std::vector<double> dxs{
      0.0, -0.0, -1.0, 1.0, -dinf, dinf, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::denorm_min()};
  std::vector<double> dout(dxs.size(), 0.0);
  std::vector<int> didx(dxs.size());
  for (size_t i = 0; i < didx.size(); ++i)
    didx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, didx.begin(), didx.end(),
                [in = dxs.data(), out = dout.data()](int i) { out[i] = __polyregion_gpu_signum(in[i]); });
  for (size_t i = 0; i < dxs.size(); ++i)
    if (!same(dout[i], expected(dxs[i]))) {
      printf("f64 signum(%.17g): device %.17g host %.17g\n", dxs[i], dout[i], expected(dxs[i]));
      ++failures;
    }

  printf(failures == 0 ? "signum ok" : "signum failures=%d", failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
}
