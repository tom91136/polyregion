#include <hip/hip_runtime.h>

namespace polyregion::spectra {
template <class Sig> struct function;
template <class R, class... A> struct [[clang::annotate("polyregion_callable_variable:Callable0")]] function<R(A...)> {
  __host__ __device__ function() {}
  template <class F> __host__ __device__ function(F) {}
  __host__ __device__ R operator()(A...) const { return R(); }
};
template <class Sig> struct function2;
template <class R, class... A> struct [[clang::annotate("polyregion_callable_variable:Callable1")]] function2<R(A...)> {
  __host__ __device__ function2() {}
  template <class F> __host__ __device__ function2(F) {}
  __host__ __device__ R operator()(A...) const { return R(); }
};
} // namespace polyregion::spectra

#include "common.inc"

#define SPECTRA_VARIANT "hip"
#include <polyregion/spectra/implementation/thrust.hpp>

#include "export_instantiations.inc"
