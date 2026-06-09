#if defined(__CUDA_ARCH__)

#define __global__ __attribute__((global))
extern "C" __global__ void vecadd(const float *a, const float *b, float *c) {
  unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  c[i] = a[i] + b[i];
}

#else

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" {
typedef int CUresult, CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;
CUresult cuInit(unsigned);
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
CUresult cuCtxSynchronize();
}

int main(int argc, char **argv) try {
  const std::string dir = argc > 1 ? argv[1] : ".";
  auto file = std::ifstream{dir + "/vecadd.ptx", std::ios::binary};
  const auto ptx = std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
  if (ptx.empty()) throw std::runtime_error("missing " + dir + "/vecadd.ptx");

  const auto check = [](CUresult r, std::string_view what) {
    if (r != 0) throw std::runtime_error(std::string{what} + " failed");
  };

  constexpr int N = 1024;
  check(cuInit(0), "cuInit");
  CUdevice dev{};
  check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  std::array<char, 128> name{};
  cuDeviceGetName(name.data(), name.size(), dev);
  std::cout << "  device '" << name.data() << "'\n";
  CUcontext ctx{};
  check(cuCtxCreate_v2(&ctx, 0, dev), "cuCtxCreate");
  CUmodule mod{};
  check(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData");
  CUfunction fn{};
  check(cuModuleGetFunction(&fn, mod, "vecadd"), "cuModuleGetFunction");

  std::array<float, N> a{}, b{}, c{};
  for (int i = 0; i < N; ++i) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(2 * i); }
  CUdeviceptr da{}, db{}, dc{};
  check(cuMemAlloc_v2(&da, sizeof a), "cuMemAlloc");
  check(cuMemAlloc_v2(&db, sizeof b), "cuMemAlloc");
  check(cuMemAlloc_v2(&dc, sizeof c), "cuMemAlloc");
  check(cuMemcpyHtoD_v2(da, a.data(), sizeof a), "cuMemcpyHtoD");
  check(cuMemcpyHtoD_v2(db, b.data(), sizeof b), "cuMemcpyHtoD");
  std::array<void *, 3> args{&da, &db, &dc};
  check(cuLaunchKernel(fn, N / 128, 1, 1, 128, 1, 1, 0, nullptr, args.data(), nullptr), "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  check(cuMemcpyDtoH_v2(c.data(), dc, sizeof c), "cuMemcpyDtoH");

  int bad = 0;
  for (int i = 0; i < N; ++i) bad += c[i] != 3.0f * static_cast<float>(i);
  std::cout << "  cuda " << (bad ? "FAIL" : "PASS") << " (c[1023]=" << c[1023] << ", mismatches=" << bad << ")\n";
  return bad == 0 ? 0 : 1;
} catch (const std::exception &e) {
  std::cerr << "  FAIL " << e.what() << "\n";
  return 1;
}

#endif
