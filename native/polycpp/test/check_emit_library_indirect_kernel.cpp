#pragma region case: indirect-device-kernel-phase
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_35 -nocudainc -nocudalib -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-offload-i32-constant {output}.polyast 9

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
extern "C" int cudaLaunchKernel(const void *, dim3, dim3, void **, unsigned long, void *);
extern "C" int __cudaPushCallConfiguration(dim3, dim3, unsigned long = 0, void * = nullptr);

template <typename T> __attribute__((global)) void phase_kernel(T *out) {
#ifdef __CUDA_ARCH__
  out[0] = T(9);
#else
  out[0] = T(4);
#endif
}

template __attribute__((global)) void phase_kernel<int>(int *);

POLYREGION_EXPORT_AS("foo.implementation.apply") void apply(int *out) {
#ifndef __CUDA_ARCH__
  auto kernel = &phase_kernel<int>;
  kernel<<<dim3(1), dim3(1)>>>(out);
#endif
}
