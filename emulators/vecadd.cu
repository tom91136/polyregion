#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define STR_(x) #x
#define STR(x) STR_(x)
#define L0 abcdefghij
#define L1 CAT(L0, L0)
#define L2 CAT(L1, L1)
#define L3 CAT(L2, L2)
#define L4 CAT(L3, L3)
#define L5 CAT(L4, L4)
#define L6 CAT(L5, L5)
#define L7 CAT(L6, L6)
#define LONG_NAME CAT(L7, L4)

#if defined(__CUDA_ARCH__)

  #define __global__ __attribute__((global))
extern "C" __global__ void vecadd(const float *a, const float *b, float *c) {
  unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  c[i] = a[i] + b[i];
}

extern "C" __global__ void pick(const int *a, int *c) {
  unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  int scratch[8];
  for (int j = 0; j < 8; ++j)
    scratch[j] = -j;
  const int *p = i % 3 == 0 ? &scratch[i & 7] : &a[i];
  c[i] = *p;
}

extern "C" __global__ void LONG_NAME(const int *a, const int *b, int *c) {
  unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  c[i] = a[i] * 2 + b[i];
}

#else

  #include <array>
  #include <cstdio>
  #include <fstream>
  #include <iterator>
  #include <stdexcept>
  #include <string>
  #include <string_view>

typedef int CUresult, CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;

  #ifdef _WIN32
    #include <windows.h>
static CUresult (*cuInit)(unsigned);
static CUresult (*cuDeviceGet)(CUdevice *, int);
static CUresult (*cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*cuCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
static CUresult (*cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (*cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                                  void **);
static CUresult (*cuCtxSynchronize)();
static void loadCuda() {
  HMODULE h = LoadLibraryA("nvcuda.dll");
  if (!h) throw std::runtime_error("LoadLibrary nvcuda.dll failed");
    #define LD(f)                                                                                                                          \
      f = reinterpret_cast<decltype(f)>(GetProcAddress(h, #f));                                                                            \
      if (!f) throw std::runtime_error("missing export " #f)
  LD(cuInit);
  LD(cuDeviceGet);
  LD(cuDeviceGetName);
  LD(cuCtxCreate_v2);
  LD(cuModuleLoadData);
  LD(cuModuleGetFunction);
  LD(cuMemAlloc_v2);
  LD(cuMemcpyHtoD_v2);
  LD(cuMemcpyDtoH_v2);
  LD(cuLaunchKernel);
  LD(cuCtxSynchronize);
    #undef LD
}
  #else
extern "C" {
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
  #endif

int main(int argc, char **argv) try {
  #ifdef _WIN32
  loadCuda();
  #endif
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
  std::printf("  device '%s'\n", name.data());
  CUcontext ctx{};
  check(cuCtxCreate_v2(&ctx, 0, dev), "cuCtxCreate");
  CUmodule mod{};
  check(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData");
  CUfunction fn{};
  check(cuModuleGetFunction(&fn, mod, "vecadd"), "cuModuleGetFunction");

  std::array<float, N> a{}, b{}, c{};
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
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
  for (int i = 0; i < N; ++i)
    bad += c[i] != 3.0f * static_cast<float>(i);
  std::printf("  cuda %s (c[1023]=%g, mismatches=%d)\n", bad ? "FAIL" : "PASS", static_cast<double>(c[1023]), bad);

  CUfunction pick{};
  check(cuModuleGetFunction(&pick, mod, "pick"), "cuModuleGetFunction");
  std::array<int, N> in{}, out{};
  for (int i = 0; i < N; ++i)
    in[i] = i;
  check(cuMemcpyHtoD_v2(da, in.data(), sizeof in), "cuMemcpyHtoD");
  std::array<void *, 2> pickArgs{&da, &dc};
  check(cuLaunchKernel(pick, N / 128, 1, 1, 128, 1, 1, 0, nullptr, pickArgs.data(), nullptr), "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  check(cuMemcpyDtoH_v2(out.data(), dc, sizeof out), "cuMemcpyDtoH");
  int pickBad = 0;
  for (int i = 0; i < N; ++i)
    pickBad += out[i] != (i % 3 == 0 ? -(i & 7) : i);
  std::printf("  cuda local/global pointer select %s (mismatches=%d)\n", pickBad ? "FAIL" : "PASS", pickBad);

  CUfunction longName{};
  check(cuModuleGetFunction(&longName, mod, STR(LONG_NAME)), "cuModuleGetFunction");
  std::array<int, N> in2{};
  for (int i = 0; i < N; ++i)
    in2[i] = 3 * i;
  check(cuMemcpyHtoD_v2(db, in2.data(), sizeof in2), "cuMemcpyHtoD");
  std::array<void *, 3> longArgs{&da, &db, &dc};
  check(cuLaunchKernel(longName, N / 128, 1, 1, 128, 1, 1, 0, nullptr, longArgs.data(), nullptr), "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  check(cuMemcpyDtoH_v2(out.data(), dc, sizeof out), "cuMemcpyDtoH");
  int longBad = 0;
  for (int i = 0; i < N; ++i)
    longBad += out[i] != 2 * i + 3 * i;
  std::printf("  cuda long kernel and parameter names %s (mismatches=%d)\n", longBad ? "FAIL" : "PASS", longBad);
  return bad == 0 && pickBad == 0 && longBad == 0 ? 0 : 1;
} catch (const std::exception &e) {
  std::fprintf(stderr, "  FAIL %s\n", e.what());
  return 1;
}

#endif
