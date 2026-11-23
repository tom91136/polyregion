#pragma region case: glcompute-math-f64-unary
#pragma region using: fn=sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,exp,log,cbrt,expm1,log1p
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
#ifdef SIGNUM
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
