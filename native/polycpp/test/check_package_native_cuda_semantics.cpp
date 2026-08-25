#pragma region case: package-native-cuda-semantics
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-native-cuda-semantics {output}.polyast

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

__attribute__((device)) int invoke_and_store(int *pointer) {
  *pointer += 1;
  return *pointer;
}

__attribute__((device)) void rebase(int *&pointer) { ++pointer; }

__attribute__((global)) void native_semantics(int *values) {
  __attribute__((shared)) int fixed[8];
  extern __attribute__((shared)) int dynamic[];
  const unsigned index = threadIdx.x + threadIdx.y + threadIdx.z + blockIdx.x + blockIdx.y + blockIdx.z + blockDim.x + blockDim.y
                         + blockDim.z + gridDim.x + gridDim.y + gridDim.z;
  int *pointer = values;
  const auto address = reinterpret_cast<unsigned long long>(pointer);
  pointer = reinterpret_cast<int *>(address);
  rebase(pointer);
  int registerArray[4]{};
  auto *bytes = reinterpret_cast<unsigned char *>(registerArray);
  __builtin_nontemporal_store(__builtin_nontemporal_load(pointer), pointer);
  __atomic_store_n(pointer, int(index), __ATOMIC_RELEASE);
  const int published = __atomic_load_n(pointer, __ATOMIC_ACQUIRE);
  int expected = 0;
  const bool exchanged = __atomic_compare_exchange_n(pointer, &expected, int(index), false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  static_cast<void>(invoke_and_store(pointer));
  fixed[threadIdx.x] = int(bytes[0]) + expected + int(exchanged) + published;
  dynamic[threadIdx.x] = fixed[threadIdx.x];
}

POLYREGION_EXPORT_AS("native_cuda.implementation.apply") void apply(int *values) {
#ifndef __CUDA_ARCH__
  native_semantics<<<dim3(1), dim3(8)>>>(values);
#endif
}
