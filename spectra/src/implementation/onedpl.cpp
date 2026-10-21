// oneDPL Spectra implementation. Vendor-native and PolyAST compilation use the same algorithm templates;
// this translation unit only supplies erased variables and selects the concrete ABI width specialisations.

namespace polyregion::spectra {
template <class Sig> struct function;
template <class R, class... A> struct [[clang::annotate("polyregion_callable_variable:Callable0")]] function<R(A...)> {
  function() = default;
  template <class F> function(F) {}
  R operator()(A...) const { return R(); }
};
template <class Sig> struct function2;
template <class R, class... A> struct [[clang::annotate("polyregion_callable_variable:Callable1")]] function2<R(A...)> {
  function2() = default;
  template <class F> function2(F) {}
  R operator()(A...) const { return R(); }
};
} // namespace polyregion::spectra

#include "common.inc"

#define SPECTRA_VARIANT "dpl"
#include <polyregion/spectra/implementation/onedpl.hpp>

#include "export_instantiations.inc"
