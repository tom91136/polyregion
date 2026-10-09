#pragma region case: package-hip-wave-votes
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-wave-votes {output}.polyast

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
extern "C" int cudaLaunchKernel(const void *, dim3, dim3, void **, unsigned long, void *);
extern "C" int __cudaPushCallConfiguration(dim3, dim3, unsigned long = 0, void * = nullptr);

struct __attribute__((device_builtin)) uint3 {
  unsigned x, y, z;
};
#include <__clang_cuda_builtin_vars.h>

extern "C" __attribute__((device)) int __ockl_wfall_i32(int);
extern "C" __attribute__((device)) int __ockl_wfany_i32(int);

__attribute__((global)) void votes(int *out) {
  const int lane = int(threadIdx.x);
  int all = 0;
  while (!__ockl_wfall_i32(lane < 40 || all > 2))
    ++all;
  out[lane] = all + __ockl_wfany_i32(lane == 5);
}

POLYREGION_EXPORT_AS("hip_wave_votes.implementation.apply") void apply(int *out) {
#ifndef __CUDA_ARCH__
  votes<<<dim3(1), dim3(32)>>>(out);
#endif
}
