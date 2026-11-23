#pragma region case: glcompute-math-f32
#pragma region using: fn=cbrt,expm1,log1p,hypot,atan2,transcendental,signum
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFN={fn} -o {output} {input}
#pragma region do: {output}
#pragma region requires: math f32 ok

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <limits>
#include <vector>

#include "polystl/intrinsics.h"

static int64_t ordered(float x) {
  int32_t i;
  std::memcpy(&i, &x, sizeof(i));
  return i < 0 ? static_cast<int64_t>(INT32_MIN) - i : i;
}

static bool within(float device, float host, double ulps, double absTol) {
  if (std::isnan(host) || std::isnan(device)) return std::isnan(host) && std::isnan(device);
  if (std::isinf(host) || std::isinf(device)) return host == device;
  if (std::fabs(static_cast<double>(device) - host) <= absTol) return true;
  return std::llabs(ordered(device) - ordered(host)) <= ulps;
}

static const float inf = std::numeric_limits<float>::infinity();
static const float qnan = std::numeric_limits<float>::quiet_NaN();

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

static bool selected(const char *name) { return std::strcmp(STRINGIFY(FN), name) == 0; }

struct Check {
  int failures = 0;

  template <typename D, typename H>
  void unary(const char *name, const std::vector<float> &xs, double ulps, double absTol, D device, H host) {
    std::vector<float> out(xs.size(), 0.f);
    std::vector<int> idx(xs.size());
    for (size_t i = 0; i < idx.size(); ++i)
      idx[i] = static_cast<int>(i);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [in = xs.data(), o = out.data(), device](int i) { o[i] = device(in[i]); });
    for (size_t i = 0; i < xs.size(); ++i) {
      const float h = host(xs[i]);
      if (!within(out[i], h, ulps, absTol)) {
        printf("%s(%.9g): device %.9g host %.9g (%lld ulp)\n", name, xs[i], out[i], h,
               static_cast<long long>(std::llabs(ordered(out[i]) - ordered(h))));
        ++failures;
      }
    }
  }

  template <typename D, typename H>
  void binary(const char *name, const std::vector<float> &xs, const std::vector<float> &ys, double ulps, double absTol, D device, H host) {
    std::vector<float> out(xs.size(), 0.f);
    std::vector<int> idx(xs.size());
    for (size_t i = 0; i < idx.size(); ++i)
      idx[i] = static_cast<int>(i);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [a = xs.data(), b = ys.data(), o = out.data(), device](int i) { o[i] = device(a[i], b[i]); });
    for (size_t i = 0; i < xs.size(); ++i) {
      const float h = host(xs[i], ys[i]);
      if (!within(out[i], h, ulps, absTol)) {
        printf("%s(%.9g, %.9g): device %.9g host %.9g (%lld ulp)\n", name, xs[i], ys[i], out[i], h,
               static_cast<long long>(std::llabs(ordered(out[i]) - ordered(h))));
        ++failures;
      }
    }
  }
};

int main() {
  Check c;

  if (selected("cbrt")) {
    const std::vector<float> wide{-1e30f, -1000.f, -27.f, -8.f, -2.5f, -1.f, -1e-3f, -1e-20f, 0.f,   1e-20f, 1e-3f, 0.125f,
                                  1.f,    2.f,     3.5f,  8.f,  27.f,  1e3f, 1e10f,  1e30f,   3e38f, inf,    -inf,  qnan};
    c.unary("cbrt", wide, 4, 0, [](float x) { return std::cbrt(x); }, [](float x) { return std::cbrt(x); });
  }
  if (selected("expm1")) {
    const std::vector<float> expArgs{-inf,  -100.f, -20.f, -5.f, -1.5f, -1.f, -0.5f, -1e-3f, -1e-7f, 0.f, 1e-7f, 1e-3f,
                                     0.25f, 0.5f,   0.99f, 1.f,  1.5f,  5.f,  20.f,  88.f,   100.f,  inf, qnan};
    c.unary("expm1", expArgs, 8, 0, [](float x) { return std::expm1(x); }, [](float x) { return std::expm1(x); });
  }
  if (selected("log1p")) {
    const std::vector<float> logArgs{-2.f,  -1.f,  -0.999f, -0.5f, -0.29f, -1e-3f, -1e-7f, 0.f,   1e-7f, 1e-3f,
                                     0.25f, 0.41f, 0.5f,    1.f,   2.f,    10.f,   1e3f,   1e30f, inf,   qnan};
    c.unary("log1p", logArgs, 4, 0, [](float x) { return std::log1p(x); }, [](float x) { return std::log1p(x); });
  }
  if (selected("hypot")) {
    const std::vector<float> hx{3.f, 0.f, 0.f, -3.f, 1e30f, 2e38f, 1e-30f, 1e-20f, 5.f, inf, qnan, inf, 1.f, 123456.f, -7.f};
    const std::vector<float> hy{4.f, 0.f, 2.f, 4.f, 1e30f, 2e38f, 1e-30f, 1.f, 12.f, qnan, inf, 3.f, 1e-4f, 654321.f, 24.f};
    c.binary("hypot", hx, hy, 4, 0, [](float x, float y) { return std::hypot(x, y); }, [](float x, float y) { return std::hypot(x, y); });
  }
  if (selected("atan2")) {
    const std::vector<float> ay{0.f, 1.f, -1.f, 1.f, -1.f, 0.f, 0.f, 5.f, -5.f, inf, -inf, inf, 1.f, 1e-20f, -3.f, qnan};
    const std::vector<float> ax{0.f, 1.f, 1.f, -1.f, -1.f, 2.f, -2.f, 0.f, 0.f, 1.f, 1.f, inf, inf, 1e20f, -4.f, 1.f};
    c.binary(
        "atan2", ay, ax, 4096, 1e-6, [](float y, float x) { return std::atan2(y, x); }, [](float y, float x) { return std::atan2(y, x); });
  }
  if (selected("transcendental")) {
    const std::vector<float> trig{-3.f, -1.5f, -1.f, -0.5f, -1e-3f, 0.f, 1e-3f, 0.5f, 1.f, 1.5f, 3.f};
    const std::vector<float> unit{-1.f, -0.9f, -0.5f, -1e-3f, 0.f, 1e-3f, 0.5f, 0.9f, 1.f};
    const std::vector<float> pos{1e-3f, 0.1f, 0.5f, 1.f, 2.f, 10.f, 1e3f, 1e10f};
    const std::vector<float> mid{-5.f, -1.f, -0.1f, 0.f, 0.1f, 1.f, 5.f};
    c.unary("sin", trig, 4096, 1e-3, [](float x) { return std::sin(x); }, [](float x) { return std::sin(x); });
    c.unary("cos", trig, 4096, 1e-3, [](float x) { return std::cos(x); }, [](float x) { return std::cos(x); });
    c.unary("tan", trig, 4096, 1e-3, [](float x) { return std::tan(x); }, [](float x) { return std::tan(x); });
    c.unary("asin", unit, 4096, 1e-3, [](float x) { return std::asin(x); }, [](float x) { return std::asin(x); });
    c.unary("acos", unit, 4096, 1e-3, [](float x) { return std::acos(x); }, [](float x) { return std::acos(x); });
    c.unary("atan", trig, 4096, 1e-3, [](float x) { return std::atan(x); }, [](float x) { return std::atan(x); });
    c.unary("sinh", mid, 4096, 1e-3, [](float x) { return std::sinh(x); }, [](float x) { return std::sinh(x); });
    c.unary("cosh", mid, 4096, 1e-3, [](float x) { return std::cosh(x); }, [](float x) { return std::cosh(x); });
    c.unary("tanh", mid, 4096, 1e-3, [](float x) { return std::tanh(x); }, [](float x) { return std::tanh(x); });
    c.unary("exp", mid, 64, 0, [](float x) { return std::exp(x); }, [](float x) { return std::exp(x); });
    c.unary("log", pos, 64, 1e-5, [](float x) { return std::log(x); }, [](float x) { return std::log(x); });
    c.unary("log10", pos, 64, 1e-5, [](float x) { return std::log10(x); }, [](float x) { return std::log10(x); });
    c.binary(
        "pow", pos, std::vector<float>(pos.size(), 1.5f), 64, 0, [](float x, float y) { return std::pow(x, y); },
        [](float x, float y) { return std::pow(x, y); });
  }

  if (selected("signum")) {
    const std::vector<float> xs{-inf, -1e30f, -1.f, -1e-20f, -0.f, 0.f, 1e-20f, 1.f, 1e30f, inf, qnan};
    const auto host = [](float x) { return std::isnan(x) || x == 0.f ? x : std::copysign(1.f, x); };
    c.unary("signum", xs, 0, 0, [](float x) { return __polyregion_gpu_signum(x); }, host);
  }

  if (c.failures == 0) printf("math f32 ok");
  else printf("math f32 failures=%d", c.failures);
  fflush(stdout);
  return c.failures == 0 ? 0 : 1;
}
