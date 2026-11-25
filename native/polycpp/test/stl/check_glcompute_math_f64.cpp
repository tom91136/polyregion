#pragma region case: glcompute-math-f64-unary
#pragma region using: fn=sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,expm1,log1p,exp
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFN={fn} -o {output} {input}
#pragma region do: {output}
#pragma region requires: f64 ok

#pragma region case: glcompute-math-f64-binary
#pragma region using: fn=hypot
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DBINARY -DFN={fn} -o {output} {input}
#pragma region do: {output}
#pragma region requires: f64 ok

#pragma region case: glcompute-math-f64-unary-refused
#pragma region using: fn=log,log10,cbrt
#pragma region offload-only
#pragma region targets: spirv_glcompute
#pragma region compile-fails: Vulkan has no 64-bit float form of
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DFN={fn} -o {output} {input}

#pragma region case: glcompute-math-f64-binary-refused
#pragma region using: fn=atan2,pow
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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <limits>
#include <utility>
#include <vector>

#include "polystl/intrinsics.h"

#define CAT_IMPL(a, b) a##b
#define CAT(a, b) CAT_IMPL(a, b)
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)
#define REFA(x) static_cast<double>(std::CAT(FN, l)(static_cast<long double>(x)))
#define REFB(x, y) static_cast<double>(std::CAT(FN, l)(static_cast<long double>(x), static_cast<long double>(y)))

static bool same(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

static long long ulpDistance(double a, double b) {
  std::uint64_t ua = 0, ub = 0;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  const auto key = [](std::uint64_t u) { return (u >> 63) ? ~u : (u | (std::uint64_t(1) << 63)); };
  const auto ka = key(ua), kb = key(ub);
  return ka > kb ? static_cast<long long>(ka - kb) : -static_cast<long long>(kb - ka);
}

static bool close(double got, double ref, int bound) {
  if (std::isnan(ref)) return std::isnan(got);
  if (std::isinf(ref) || ref == 0.0) return same(got, ref);
  const long long d = ulpDistance(got, ref);
  return d <= bound && d >= -bound;
}

static double nextU(std::uint64_t &state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(state >> 11) / 9007199254740992.0;
}

static const double inf = std::numeric_limits<double>::infinity();
static const double qnan = std::numeric_limits<double>::quiet_NaN();
static const double tiny = std::numeric_limits<double>::denorm_min();
static const double dmin = std::numeric_limits<double>::min();
static const double dmax = std::numeric_limits<double>::max();

struct UnaryConfig {
  const char *name;
  double lo, hi;
  bool logUniform;
  double edgeLo, edgeHi;
  int ulp;
  std::vector<double> specials;
};

struct BinaryConfig {
  const char *name;
  double lo, hi, lo2, hi2;
  int ulp;
  std::vector<std::pair<double, double>> specials;
};

static const std::vector<UnaryConfig> unaryConfigs = {
    {"sin",
     -1e6,
     1e6,
     false,
     0.0,
     0.0,
     1,
     {0.0, -0.0, inf, -inf, qnan, 1.5707963267948966, -1.5707963267948966, 3.141592653589793, 1e-8, -1e-8, dmax, tiny}},
    {"cos", -1e6, 1e6, false, 0.0, 0.0, 2, {0.0, -0.0, inf, -inf, qnan, 1.5707963267948966, 3.141592653589793, 1e-8, dmax, tiny}},
    {"tan", -1e6, 1e6, false, 0.0, 0.0, 2, {0.0, -0.0, inf, -inf, qnan, 1.5707963267948966, -1.5707963267948966, 1e-8, dmax, tiny}},
    {"asin", -1.0, 1.0, false, 0.0, 0.0, 6, {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, dmin, inf, -inf, qnan, 2.0, -2.0}},
    {"acos", -1.0, 1.0, false, 0.0, 0.0, 4, {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, dmin, inf, -inf, qnan, 2.0, -2.0}},
    {"atan", -1e6, 1e6, false, 0.0, 0.0, 1, {0.0, -0.0, 1.0, -1.0, inf, -inf, qnan, 1e-8, dmax, tiny}},
    {"sinh",
     -700.0,
     700.0,
     false,
     709.78,
     710.48,
     2,
     {0.0, -0.0, inf, -inf, qnan, 709.78, -709.78, 710.0, -710.0, 710.4, -710.4, 710.48, 1e-8, 700.0}},
    {"cosh",
     -700.0,
     700.0,
     false,
     709.78,
     710.48,
     2,
     {0.0, -0.0, inf, -inf, qnan, 709.78, -709.78, 710.0, -710.0, 710.4, -710.4, 710.48, 1e-8, 700.0}},
    {"tanh", -20.0, 20.0, false, 0.0, 0.0, 2, {0.0, -0.0, inf, -inf, qnan, 20.0, -20.0, 1.0, -1.0, 1e-8}},
    {"expm1", -745.0, 710.0, false, 0.0, 0.0, 1, {0.0, -0.0, inf, -inf, qnan, 710.0, -745.0, 1.0, -1.0, 1e-8, dmin}},
    {"log1p", -0.99, 1000.0, false, 0.0, 0.0, 1, {0.0, -0.0, -1.0, -0.5, 1.0, dmin, inf, -inf, qnan, -1.0000000000000002}},
    {"exp", -745.0, 710.0, false, 0.0, 0.0, 1, {0.0, -0.0, inf, -inf, qnan, 709.78, 710.0, -744.0, -745.0, 1.0, -1.0, 0.5, 2.0}},
    {"sqrt", 0.0, 1e300, false, 0.0, 0.0, 1, {0.0, -0.0, 1.0, 4.0, dmin, dmax, inf, -1.0, qnan}},
    {"floor", -1e6, 1e6, false, 0.0, 0.0, 1, {0.0, -0.0, 1.5, -1.5, 2.0, inf, -inf, qnan, dmin, dmax}},
    {"ceil", -1e6, 1e6, false, 0.0, 0.0, 1, {0.0, -0.0, 1.5, -1.5, 2.0, inf, -inf, qnan, dmin, dmax}},
    {"fabs", -1e300, 1e300, false, 0.0, 0.0, 1, {0.0, -0.0, 1.0, -1.0, inf, -inf, qnan, dmin, dmax, tiny}},
};

static const std::vector<BinaryConfig> binaryConfigs = {
    {"hypot",
     -1e300,
     1e300,
     -1e300,
     1e300,
     1,
     {{0.0, 0.0}, {-0.0, -0.0}, {inf, qnan}, {qnan, 1.0}, {dmax, dmax}, {3.0, 4.0}, {1e-300, 1e-300}, {-3.0, -4.0}}},
};

static int report(const char *name, std::size_t n, const std::vector<double> &out, const std::vector<double> &ref, int bound) {
  int failures = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (!close(out[i], ref[i], bound)) {
      printf("%s(case %zu): got %.17g ref %.17g ulp %lld\n", name, i, out[i], ref[i],
             std::isnan(out[i]) || std::isnan(ref[i]) ? 0 : ulpDistance(out[i], ref[i]));
      if (++failures >= 20) break;
    }
  printf(failures == 0 ? "f64 ok" : "f64 %s failures=%d", name, failures);
  fflush(stdout);
  return failures == 0 ? 0 : 1;
}

int main() {
#ifdef SIGNUM
  const std::vector<double> in{0.0, -0.0, -1.0, 1.0, -2.5, 9.0, inf, -inf, qnan, dmin};
  std::vector<double> out(in.size(), 0.0);
  std::vector<int> idx(in.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [a = in.data(), o = out.data()](int i) { o[i] = __polyregion_gpu_signum(a[i]); });
  int failures = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
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
  #ifdef BINARY
  BinaryConfig cfg{};
  for (const auto &c : binaryConfigs)
    if (std::strcmp(c.name, STR(FN)) == 0) cfg = c;
  if (cfg.name == nullptr) return 2;
  std::vector<double> in, in2;
  for (const auto &p : cfg.specials) {
    in.push_back(p.first);
    in2.push_back(p.second);
  }
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  constexpr std::size_t sweep = 8192;
  for (std::size_t i = 0; i < sweep; ++i) {
    const double u = nextU(state), v = nextU(state);
    in.push_back(cfg.lo + u * (cfg.hi - cfg.lo));
    in2.push_back(cfg.lo2 + v * (cfg.hi2 - cfg.lo2));
  }
  std::vector<double> out(in.size(), 0.0), ref(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
    ref[i] = REFB(in[i], in2[i]);
  std::vector<int> idx(in.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [a = in.data(), b = in2.data(), o = out.data()](int i) { o[i] = std::FN(a[i], b[i]); });
  return report(STR(FN), in.size(), out, ref, cfg.ulp);
  #else
  UnaryConfig cfg{};
  for (const auto &c : unaryConfigs)
    if (std::strcmp(c.name, STR(FN)) == 0) cfg = c;
  if (cfg.name == nullptr) return 2;
  std::vector<double> in = cfg.specials;
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  constexpr std::size_t sweep = 8192;
  for (std::size_t i = 0; i < sweep; ++i) {
    const double u = nextU(state);
    if (cfg.logUniform) {
      const int e = static_cast<int>(nextU(state) * 2046.0) - 1023;
      in.push_back(std::ldexp(1.0 + u, e));
    } else {
      in.push_back(cfg.lo + u * (cfg.hi - cfg.lo));
    }
  }
  if (cfg.edgeLo != 0.0 || cfg.edgeHi != 0.0) {
    for (std::size_t i = 0; i < sweep; ++i) {
      const double x = cfg.edgeLo + nextU(state) * (cfg.edgeHi - cfg.edgeLo);
      in.push_back(x);
      in.push_back(-x);
    }
  }
  std::vector<double> out(in.size(), 0.0), ref(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
    ref[i] = REFA(in[i]);
  std::vector<int> idx(in.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [a = in.data(), o = out.data()](int i) { o[i] = std::FN(a[i]); });
  return report(STR(FN), in.size(), out, ref, cfg.ulp);
  #endif
#endif
}
