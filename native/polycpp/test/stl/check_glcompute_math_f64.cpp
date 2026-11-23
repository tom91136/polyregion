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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  std::vector<double> in{0.25, 1.5, 2.0, 9.0}, out(in.size(), 0.0);
  std::vector<int> idx{0, 1, 2, 3};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [a = in.data(), b = in.data(), o = out.data()](int i) {
#ifdef BINARY
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
}
