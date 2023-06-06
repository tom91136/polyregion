#pragma once

#include "concurrency_utils.hpp"
#include "runtime.h"

namespace polystl {

struct __generated__foo_cpp_34 {

  constexpr static const unsigned char __offloadImage[] = {0xFF};
  constexpr static const unsigned char __uniqueName[] = "theName";
  constexpr static const ArgBuffer __argBuffer = ArgBuffer{};


  int hello;
  __generated__foo_cpp_34(int & _hello)
      : hello{_hello}
  {}

  inline int operator()(int & x) const {

    return x * hello;
  }

}

template <class _ExecutionPolicy, //
          class _ForwardIterator1, class _ForwardIterator2,
          class _UnaryOperation>
_ForwardIterator2 transform(_ExecutionPolicy &&__exec, //
                            _ForwardIterator1 __first, //
                            _ForwardIterator1 __last,  //
                            _ForwardIterator2 __result, _UnaryOperation __op) {
  auto N = std::distance(__first, __last);

  using namespace polyregion::runtime;

  std::string &image = __op.__offloadImage;
  std::string &name = __op.__uniqueName;
  ArgBuffer &buffer = __op.__argBuffer;

  static auto platform = Platform::of(Backend::CUDA);
  static auto theDevice = std::move(platform->enumerate()[0]);
  static auto theQueue = theDevice->createQueue();

  if (!theDevice->moduleLoaded(name)) {
    theDevice->loadModule(name, image);
  }

  polyregion::concurrency_utils::waitAll([&](auto cb) {
    theQueue->enqueueInvokeAsync(name, "kernel", buffer, Policy{{N}, {}}, [&]() {
      fprintf(stderr, "Module %s completed\n", name.c_str());
      cb();
    });
  });

  //        [[lift]] std::vector<Type::> m = __op;
  // enqueueInvokeAsync("a", "b", {Type::Ptr, Type::Ptr, <captureTpes...>}, {&__first, &__result, <captures...>}, {N, 1, 1});
  //          void kernel(__first : Struct__First){   __result[i] = __op(__first[i])    }
  //        while (__first != __last) *__result++ = __op(*__first++);
  return __result;
}
} // namespace polystl
