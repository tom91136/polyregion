#pragma region case: package-native-cuda-semantics
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-native-cuda-semantics {output}.polyast

#pragma region case: package-native-cuda-non-default-stream-diagnostic
#pragma region offload-only
#pragma region compile-fails: Non-default CUDA/HIP launch streams are not supported in package code
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NON_DEFAULT_STREAM -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}

#pragma region case: package-native-cuda-aggregate-atomic-cas-diagnostic
#pragma region offload-only
#pragma region compile-fails: Aggregate atomic compare-exchange is not representable by PolyAST
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_AGGREGATE_ATOMIC_CAS -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}

#pragma region case: package-native-cuda-generic-scalar-atomic-cas
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_GENERIC_SCALAR_ATOMIC_CAS -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
extern "C" int cudaLaunchKernel(const void *, dim3, dim3, void **, unsigned long, void *);
extern "C" int __cudaPushCallConfiguration(dim3, dim3, unsigned long = 0, void * = nullptr);
extern "C" int cudaConfigureCall(dim3, dim3, unsigned long = 0, void * = nullptr);
extern "C" int cudaSetupArgument(const void *, unsigned long, unsigned long);
extern "C" int cudaLaunch(const void *);
extern "C" __attribute__((host, device)) int cudaMemcpy(void *, const void *, unsigned long, int);

struct AtomicPair {
  int first;
  int second;
};

struct __attribute__((device_builtin)) uint3 {
  unsigned x, y, z;
};
#include <__clang_cuda_builtin_vars.h>

__attribute__((device)) int invoke_and_store(int *pointer) {
  *pointer += 1;
  return *pointer;
}

__attribute__((device)) void rebase(int *&pointer) { ++pointer; }

__attribute__((device)) int private_pointer_offsets() {
  int values[4]{};
  *values = 9;
  int *pointer = (int *)values;
  *(pointer + 1) = 17;
  *(pointer + 3 - 1) = 25;
  return *values + values[1] + values[2];
}

__attribute__((device)) int dynamic_atomic_order(int *pointer, int order) {
  __atomic_store_n(pointer, 7, order);
  int desired = 11;
  int observed = 0;
  __atomic_exchange(pointer, &desired, &observed, order);
  return __atomic_load_n(pointer, order) + observed;
}

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
  fixed[threadIdx.x] =
      int(bytes[0]) + expected + int(exchanged) + published + private_pointer_offsets() + dynamic_atomic_order(pointer, __ATOMIC_SEQ_CST);
  dynamic[threadIdx.x] = fixed[threadIdx.x];
}

__attribute__((global)) void phase_context_semantics(int *values) {
#ifdef __CUDA_ARCH__
  values[0] = 1;
#else
  cudaMemcpy(values, values, sizeof(*values), 3);
#endif
}

POLYREGION_EXPORT_AS("native_cuda.implementation.apply") void apply(int *values) {
#ifndef __CUDA_ARCH__
  #ifdef CHECK_GENERIC_SCALAR_ATOMIC_CAS
  int expected = 0;
  int desired = 1;
  static_cast<void>(__atomic_compare_exchange(values, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  #elif defined(CHECK_AGGREGATE_ATOMIC_CAS)
  auto *pointer = reinterpret_cast<AtomicPair *>(values);
  AtomicPair expected{};
  AtomicPair desired{1, 2};
  static_cast<void>(__atomic_compare_exchange(pointer, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  #elif defined(CHECK_NON_DEFAULT_STREAM)
  native_semantics<<<dim3(1), dim3(8), 0, values>>>(values);
  #else
  native_semantics<<<dim3(1), dim3(8)>>>(values);
  phase_context_semantics<<<dim3(1), dim3(1)>>>(values);
  #endif
#endif
}
