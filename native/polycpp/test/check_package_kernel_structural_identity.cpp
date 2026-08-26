#pragma region case: package-kernel-structural-identity
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -x cuda --cuda-gpu-arch=sm_35 -nocudainc -nocudalib -fsyntax-only {input}
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #kernel_kernel 2
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #kernel_overloaded 2
#pragma region do: {package_fixture} --assert-function-substring-count {output}.polyast #kernel_nested_kernel 2

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

template <class F> __attribute__((global)) void kernel(F function) { function(); }
template <class F> __attribute__((global)) void nested_kernel(F function) { function(); }
__attribute__((global)) void overloaded(int *) {}
__attribute__((global)) void overloaded(float *) {}

template <class T> auto inner(T value) {
  return [=] __attribute__((device)) { (void)value; };
}
template <class F> struct reordered {
  F function;
  __attribute__((device)) void operator()() const { function(); }
};
template <class T> void launch_nested(T value) {
  auto function = inner(value);
  nested_kernel<<<dim3(1), dim3(1)>>>(reordered<decltype(function)>{function});
}

#define LAUNCH_CAPTURE(value) kernel<<<dim3(1), dim3(1)>>>([=] { (void)value; })

#ifndef __CUDA_ARCH__
struct HostOnlyDeclarationNoise {
  int value;
};
#endif

POLYREGION_EXPORT_AS("kernel_identity.implementation.apply") void apply(int narrow, long long wide) {
  LAUNCH_CAPTURE(narrow);
  LAUNCH_CAPTURE(wide);
  overloaded<<<dim3(1), dim3(1)>>>(static_cast<int *>(nullptr));
  overloaded<<<dim3(1), dim3(1)>>>(static_cast<float *>(nullptr));
  launch_nested(narrow);
  launch_nested(wide);
}
