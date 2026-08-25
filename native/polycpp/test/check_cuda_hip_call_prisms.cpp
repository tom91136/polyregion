#pragma region case: cuda-hip-call-prisms
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-cuda-hip-source-prisms {output}.polyast

#pragma region case: hip-mbcnt-mask-diagnostic
#pragma region offload-only
#pragma region compile-fails: HIP mbcnt currently requires a constant all-lanes mask
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_HIP_MBCNT_MASK -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: inline-asm-template-matching
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_INLINE_ASM_OPERAND_NAME -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: inline-asm-extra-instruction
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_INLINE_ASM_EXTRA_INSTRUCTION -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: inline-asm-near-match
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_INLINE_ASM_NEAR_MATCH -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: inline-asm-bfe-near-match
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_INLINE_ASM_BFE_NEAR_MATCH -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: inline-asm-vshr-near-match
#pragma region offload-only
#pragma region compile-fails: Unsupported inline asm at
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_INLINE_ASM_VSHR_NEAR_MATCH -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: cub-warpscan-operation-diagnostic
#pragma region offload-only
#pragma region compile-fails: CUB WarpScan currently supports only the standard additive operation
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CUB_WARPSCAN_OPERATION -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: cuda-occupancy-diagnostic
#pragma region offload-only
#pragma region compile-fails: cudaOccupancyMaxActiveBlocksPerMultiprocessor requires a target-specific occupancy query
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CUDA_OCCUPANCY -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: thrust-relocation-provenance-diagnostic
#pragma region offload-only
#pragma region compile-fails: thrust::uninitialized_copy_n raw destinations have ambiguous memory provenance
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_THRUST_RAW_RELOCATION -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#pragma region case: signed-bfe-diagnostic
#pragma region offload-only
#pragma region compile-fails: Signed PTX bit-field extraction is not supported
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_SIGNED_BFE -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

namespace cub {
enum CacheModifier { LOAD_DEFAULT, STORE_DEFAULT };
template <CacheModifier, class T> T ThreadLoad(T *pointer) { return *pointer; }
template <CacheModifier, class T> void ThreadStore(T *pointer, T value) { *pointer = value; }
template <unsigned Width> int ShuffleDown(int value, unsigned delta, unsigned lastLane, unsigned mask) {
  return value + int(Width + delta + lastLane + mask);
}
template <unsigned Width> int ShuffleUp(int value, unsigned delta, unsigned firstLane, unsigned mask) {
  return value + int(Width + delta + firstLane + mask);
}
template <unsigned Width> int ShuffleIndex(int value, unsigned sourceLane, unsigned mask) { return value + int(Width + sourceLane + mask); }
int WARP_BALLOT(int predicate, unsigned mask) { return predicate + int(mask); }
void WARP_SYNC(unsigned) {}
struct Sum {};
template <class T, unsigned Width> struct WarpScanShfl {
  unsigned member_mask;
  unsigned lane_id;
  template <class Op> T InclusiveScanStep(T input, Op, int firstLane, int offset) const { return input + T(Width + firstLane + offset); }
};
} // namespace cub

namespace application {
struct plus_like {};
struct WarpScanShfl {
  unsigned member_mask;
  unsigned lane_id;
  int InclusiveScanStep(int input, cub::Sum, int firstLane, int offset) const { return input + firstLane + offset; }
};
int basic_ostream_count(int value) { return value + 17; }
} // namespace application

namespace rocprim {
int warp_shuffle_up(int value, unsigned delta, unsigned width) { return value + int(delta + width); }
} // namespace rocprim

namespace thrust::cuda_cub {
struct execution_policy {};
template <class T> struct device_iterator {
  T *m_iterator;
};
template <class T> device_iterator<T> uninitialized_copy_n(execution_policy, T *source, long count, device_iterator<T> destination) {
  while (count-- > 0)
    *destination.m_iterator++ = *source++;
  return destination;
}
template <class T> T *uninitialized_copy_n(execution_policy, T *source, long count, T *destination) {
  while (count-- > 0)
    *destination++ = *source++;
  return destination;
}
} // namespace thrust::cuda_cub

int cudaMalloc(void **pointer, unsigned long) {
  *pointer = nullptr;
  return 1;
}
int cudaFree(void *) { return 1; }
int hipMalloc(void **pointer, unsigned long) {
  *pointer = nullptr;
  return 1;
}
int hipFree(void *) { return 1; }
enum cudaDeviceAttr {
  cudaDevAttrWarpSize,
  cudaDevAttrMultiProcessorCount,
  cudaDevAttrMaxSharedMemoryPerBlock,
  cudaDevAttrMaxGridDimX,
  cudaDevAttrComputeCapabilityMajor,
  cudaDevAttrComputeCapabilityMinor
};
enum hipDeviceAttribute_t { hipDeviceAttributeWarpSize };
int cudaDeviceGetAttribute(int *value, cudaDeviceAttr, int) {
  *value = 32;
  return 0;
}
int hipDeviceGetAttribute(int *value, hipDeviceAttribute_t, int) {
  *value = 32;
  return 0;
}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *, const void *, int, unsigned long) { return 0; }
int atomicAdd(int *pointer, int value) { return *pointer += value; }
int atomicCAS(int *pointer, int expected, int value) {
  const int previous = *pointer;
  if (previous == expected) *pointer = value;
  return previous;
}
unsigned __nv_brev(unsigned value) { return value; }
int __nv_clz(unsigned) { return 0; }
int __nv_popc(unsigned) { return 0; }
int __nv_popcll(unsigned long long) { return 0; }
unsigned __builtin_amdgcn_ballot_w32(bool predicate) { return predicate; }
unsigned __builtin_amdgcn_mbcnt_lo(unsigned mask, unsigned base) { return mask + base; }
int __shfl_xor_sync(unsigned, int value, unsigned laneMask, int width) { return value + int(laneMask) + width; }
int __nvvm_shfl_sync_bfly_i32(unsigned, int value, unsigned laneMask, int control) { return value + int(laneMask) + control; }
void __threadfence_block() {}
void __threadfence() {}
void __threadfence_system() {}

POLYREGION_EXPORT_AS("foo.implementation.apply") int apply(int value) {
  int *cuda = nullptr;
  int *hip = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&cuda), 16);
  hipMalloc(reinterpret_cast<void **>(&hip), 16);
  const auto relocated =
      thrust::cuda_cub::uninitialized_copy_n(thrust::cuda_cub::execution_policy{}, cuda, 1, thrust::cuda_cub::device_iterator<int>{hip});
  const int shuffled = cub::ShuffleDown<16>(value, 1, 14, ~0u) + cub::ShuffleUp<16>(value, 1, 1, ~0u) + cub::ShuffleIndex<16>(value, 3, ~0u)
                       + rocprim::warp_shuffle_up(value, 1, 16) + __shfl_xor_sync(~0u, value, 1, 16)
                       + __nvvm_shfl_sync_bfly_i32(~0u, value, 1, (16 << 8) | 15);
  const int voted = cub::WARP_BALLOT(value != 0, ~0u);
  int state = value;
  const int loaded = cub::ThreadLoad<cub::LOAD_DEFAULT>(&state);
  cub::ThreadStore<cub::STORE_DEFAULT>(&state, loaded + 1);
  const int hipLoaded = __hip_atomic_load(&state, __ATOMIC_RELAXED, 4);
  __hip_atomic_store(&state, hipLoaded + 1, __ATOMIC_RELAXED, 4);
  const int atomic = atomicAdd(&state, 2) + atomicCAS(&state, value + 3, value);
  const int bits = int(__nv_brev(unsigned(value))) + __nv_clz(unsigned(value)) + __nv_popc(unsigned(value))
                   + __nv_popcll(static_cast<unsigned long long>(unsigned(value))) + __builtin_ctz(unsigned(value) | 1u);
  const int hipBallot = int(__builtin_amdgcn_ballot_w32(value != 0));
  unsigned extracted[1] = {0};
  unsigned laneMask = 0;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(extracted[0]) : "r"(unsigned(value)), "r"(1u), "r"(3u));
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(laneMask));
  const cub::WarpScanShfl<int, 16> scan{~0u, 3};
  const int scanned = scan.InclusiveScanStep(value, cub::Sum{}, 0, 1);
  const application::WarpScanShfl unrelatedScan{~0u, 3};
  const int unrelatedScanned = unrelatedScan.InclusiveScanStep(value, cub::Sum{}, 0, 1);
  int cudaWarpSize = 0;
  int cudaComputeUnits = 0;
  int cudaLocalMemory = 0;
  int cudaMaxGridX = 0;
  int cudaComputeMajor = 0;
  int cudaComputeMinor = 0;
  int hipWarpSize = 0;
  cudaDeviceGetAttribute(&cudaWarpSize, cudaDevAttrWarpSize, 0);
  cudaDeviceGetAttribute(&cudaComputeUnits, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&cudaLocalMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  cudaDeviceGetAttribute(&cudaMaxGridX, cudaDevAttrMaxGridDimX, 0);
  cudaDeviceGetAttribute(&cudaComputeMajor, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&cudaComputeMinor, cudaDevAttrComputeCapabilityMinor, 0);
  hipDeviceGetAttribute(&hipWarpSize, hipDeviceAttributeWarpSize, 0);
  cub::WARP_SYNC(~0u);
  __threadfence_block();
  __threadfence();
  __threadfence_system();
  cudaFree(cuda);
  hipFree(hip);
  return shuffled + voted + atomic + bits + hipBallot + hipLoaded + int(extracted[0] + laneMask) + scanned + unrelatedScanned + cudaWarpSize
         + hipWarpSize + cudaComputeUnits + cudaLocalMemory + cudaMaxGridX + cudaComputeMajor + cudaComputeMinor
         + application::basic_ostream_count(value) + int(relocated.m_iterator != nullptr);
}

#ifdef CHECK_HIP_MBCNT_MASK
POLYREGION_EXPORT_AS("foo.implementation.reject_mbcnt") unsigned reject_mbcnt(unsigned mask) { return __builtin_amdgcn_mbcnt_lo(mask, 0); }
#endif

#ifdef CHECK_INLINE_ASM_OPERAND_NAME
POLYREGION_EXPORT_AS("foo.implementation.reject_asm") unsigned reject_asm(unsigned trap_value) {
  unsigned output = 0;
  asm("bra trap_handler; trap_handler:" : "=r"(output) : "r"(trap_value));
  return output;
}
#endif

#ifdef CHECK_INLINE_ASM_EXTRA_INSTRUCTION
POLYREGION_EXPORT_AS("foo.implementation.reject_extra_asm") unsigned reject_extra_asm(unsigned value) {
  unsigned output = 0;
  asm("{ .reg .pred p; and.b32 %0, %1, %2; setp.ne.u32 p, %0, 0; vote.ballot.sync.b32 %0, p, 0xffffffff; "
      "@!p not.b32 %0, %0; xor.b32 %0, %0, 1; }"
      : "=r"(output)
      : "r"(value), "r"(1u));
  return output;
}
#endif

#ifdef CHECK_INLINE_ASM_NEAR_MATCH
POLYREGION_EXPORT_AS("foo.implementation.reject_near_match_asm") unsigned reject_near_match_asm(unsigned value) {
  unsigned output = 0;
  asm("{ .reg .pred p; and.b32 %0, %1, %2; setp.eq.u32 p, %0, 0; vote.ballot.sync.b32 %0, p, 0xffffffff; "
      "@!p not.b32 %0, %0; }"
      : "=r"(output)
      : "r"(value), "r"(1u));
  return output;
}
#endif

#ifdef CHECK_INLINE_ASM_BFE_NEAR_MATCH
POLYREGION_EXPORT_AS("foo.implementation.reject_bfe_near_match") unsigned reject_bfe_near_match(unsigned value) {
  unsigned output = 0;
  asm("bfe.u32 %0, %2, %1, %3;" : "=r"(output) : "r"(value), "r"(1u), "r"(3u));
  return output;
}
#endif

#ifdef CHECK_INLINE_ASM_VSHR_NEAR_MATCH
POLYREGION_EXPORT_AS("foo.implementation.reject_vshr_near_match") unsigned reject_vshr_near_match(unsigned value) {
  unsigned output = 0;
  asm("vshr.u32.u32.u32.clamp.add %0, %2, %1, %3;" : "=r"(output) : "r"(value), "r"(1u), "r"(3u));
  return output;
}
#endif

#ifdef CHECK_CUB_WARPSCAN_OPERATION
POLYREGION_EXPORT_AS("foo.implementation.reject_scan") int reject_scan(int value) {
  const cub::WarpScanShfl<int, 16> scan{~0u, 3};
  return scan.InclusiveScanStep(value, application::plus_like{}, 0, 1);
}
#endif

#ifdef CHECK_CUDA_OCCUPANCY
POLYREGION_EXPORT_AS("foo.implementation.reject_occupancy") int reject_occupancy(const void *kernel) {
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, 0);
  return blocks;
}
#endif

#ifdef CHECK_THRUST_RAW_RELOCATION
POLYREGION_EXPORT_AS("foo.implementation.reject_raw_relocation") int *reject_raw_relocation(int *source, int *destination) {
  return thrust::cuda_cub::uninitialized_copy_n(thrust::cuda_cub::execution_policy{}, source, 1, destination);
}
#endif

#ifdef CHECK_SIGNED_BFE
POLYREGION_EXPORT_AS("foo.implementation.reject_signed_bfe") int reject_signed_bfe(int value, unsigned start, unsigned length) {
  int output = 0;
  asm("bfe.s32 %0, %1, %2, %3;" : "=r"(output) : "r"(value), "r"(start), "r"(length));
  return output;
}
#endif
