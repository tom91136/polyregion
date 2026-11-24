#pragma region case: glcompute-math-f64-unary
#pragma region using: fn=sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,log,cbrt,expm1,log1p
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region compile-fails: Vulkan has no 64-bit float form of
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFN={fn} -o {output} {input}

#pragma region case: glcompute-math-f64-binary
#pragma region using: fn=pow,atan2,hypot
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region compile-fails: Vulkan has no 64-bit float form of
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DBINARY -DFN={fn} -o {output} {input}

#pragma region case: glcompute-math-f64-exp
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DEXP -o {output} {input}
#pragma region do: {output}
#pragma region requires: f64 exp ok

#pragma region case: glcompute-math-f64-supported
#pragma region using: fn=sqrt,floor,ceil,fabs
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFN={fn} -o {output} {input}
#pragma region do: {output}
#pragma region requires: f64 ok

#pragma region case: glcompute-math-f64-signum
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSIGNUM -o {output} {input}
#pragma region do: {output}
#pragma region requires: f64 ok

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <limits>
#include <vector>

#include "polystl/intrinsics.h"

static bool same(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

int main() {
#ifdef EXP
  const auto ulpDistance = [](double a, double b) {
    std::uint64_t ua = 0, ub = 0;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    const auto key = [](std::uint64_t u) { return (u >> 63) ? ~u : (u | (std::uint64_t(1) << 63)); };
    const auto ka = key(ua), kb = key(ub);
    return ka > kb ? static_cast<long long>(ka - kb) : -static_cast<long long>(kb - ka);
  };
  std::vector<double> in{0.0, -0.0, INFINITY, -INFINITY, NAN, 709.78, 710.0, -744.0, -745.0, 1.0, -1.0, 0.5, 2.0, 10.0, -10.0};
  constexpr std::size_t sweep = 8192;
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (std::size_t i = 0; i < sweep; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
    in.push_back(-745.0 + u * 1455.0);
  }
  std::vector<double> ref(in.size()), out(in.size(), 0.0);
  for (std::size_t i = 0; i < in.size(); ++i)
    ref[i] = static_cast<double>(std::expl(in[i]));
  std::vector<int> idx(in.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [a = in.data(), o = out.data()](int i) { o[i] = std::exp(a[i]); });
  int failures = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const bool special = i < 15;
    bool ok = false;
    if (std::isnan(in[i])) ok = std::isnan(out[i]);
    else if (special || std::isinf(ref[i])) ok = same(out[i], ref[i]);
    else ok = ulpDistance(out[i], ref[i]) <= 1 && ulpDistance(out[i], ref[i]) >= -1;
    if (!ok) {
      printf("exp(input[%zu]=%.17g): got %.17g ref %.17g ulp %lld\n", i, in[i], out[i], ref[i],
             std::isnan(out[i]) || std::isnan(ref[i]) ? 0 : ulpDistance(out[i], ref[i]));
      if (++failures >= 20) break;
    }
  }
  printf(failures == 0 ? "f64 exp ok" : "f64 exp failures=%d", failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
#elif defined(SIGNUM)
  const double inf = std::numeric_limits<double>::infinity();
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> in{0.0, -0.0, -1.0, 1.0, -2.5, 9.0, inf, -inf, qnan, std::numeric_limits<double>::min()};
  std::vector<double> out(in.size(), 0.0);
  std::vector<int> idx(in.size());
  for (size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [a = in.data(), o = out.data()](int i) { o[i] = __polyregion_gpu_signum(a[i]); });
  int failures = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    const double expected = std::isnan(in[i]) || in[i] == 0.0 ? in[i] : std::copysign(1.0, in[i]);
    if (!same(out[i], expected)) {
      printf("signum(%g): device %.17g host %.17g\n", in[i], out[i], expected);
      ++failures;
    }
  }
  printf(failures == 0 ? "f64 ok" : "f64 failures=%d", failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
#else
  std::vector<double> in{0.25, 1.5, 2.0, 9.0}, out(in.size(), 0.0);
  std::vector<int> idx{0, 1, 2, 3};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [a = in.data(), b = in.data(), o = out.data()](int i) {
  #ifdef SIGNUM
    o[i] = __polyregion_gpu_signum(a[i]);
  #elif defined(BINARY)
    o[i] = std::FN(a[i], b[i]);
  #else
    o[i] = std::FN(a[i]);
  #endif
  });
  double sum = 0;
  for (auto x : out)
    sum += x;
  printf(sum > 0 ? "f64 ok" : "f64 zero");
  fflush(stdout);
  return 0;
#endif
}
