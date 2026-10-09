#pragma region case: package-cuda-ptx-version
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-i32-constant {output}.polyast 700

#pragma region case: package-cuda-occupancy-member
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-field-stored {output}.polyast sm_occupancy

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

namespace cub {
int PtxVersion(int &version) {
  version = 0;
  return 0;
}
template <typename KernelPtr> int MaxSmOccupancy(int &occupancy, KernelPtr, int) { return 0; }
} // namespace cub

__attribute__((global)) void kernel(int *) {}

struct KernelConfig {
  int sm_occupancy;
  int init() { return cub::MaxSmOccupancy(sm_occupancy, kernel, 128); }
};

POLYREGION_EXPORT_AS("ptx_version.implementation.query") void query(int *out) {
#ifndef __CUDA_ARCH__
  int version = 0;
  cub::PtxVersion(version);
  KernelConfig config;
  config.init();
  out[0] = version;
  out[1] = config.sm_occupancy;
#endif
}
