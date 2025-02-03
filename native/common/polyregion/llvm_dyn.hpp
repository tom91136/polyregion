#pragma once

#include "llvm/Support/Casting.h"
#include <optional>

namespace polyregion::llvm_shared {

namespace details {
template <typename Ret, typename Arg, typename... Rest> Arg arg0_helper(Ret (*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest> Arg arg0_helper(Ret (F::*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest> Arg arg0_helper(Ret (F::*)(Arg, Rest...) const);
template <typename F> decltype(arg0_helper(&F::operator())) arg0_helper(F);
template <typename T> using arg0_type = decltype(arg0_helper(std::declval<T>()));
} // namespace details

template <typename T, typename U, typename... Fs> std::optional<T> visitDyn(U n, Fs... fs) {
  std::optional<T> result{};
  auto _ = {[&]() {
    if (!result) {
      if (auto x = llvm::dyn_cast<std::remove_pointer_t<details::arg0_type<Fs>>>(n)) {
        result = T(fs(x));
      }
    }
    return 0;
  }()...};
  return result;
}

template <typename U, typename... Fs> bool visitDyn0(U n, Fs... fs) {
  return (([&]() -> bool {
    if (auto x = llvm::dyn_cast<std::remove_pointer_t<details::arg0_type<Fs>>>(n)) {
      fs(x);
      return true;
    }
    return false;
  }()) || ...);
}

} // namespace polyregion::llvm_shared
