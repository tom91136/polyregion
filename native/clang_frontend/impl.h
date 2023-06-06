#pragma once

#include "../runtime/runtime.h"
#include "concurrency_utils.hpp"

namespace polystl {

struct __generated__foo_cpp_34 {

  int32_t a;
  int32_t b;
  int32_t *c;

  constexpr static const unsigned char __offloadImage[] = {0xFF};
  constexpr static const unsigned char __uniqueName[] = "theName";
  const polyregion::runtime::ArgBuffer __argBuffer{
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &a},
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &b},
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Ptr, &c},
  };

  __generated__foo_cpp_34(int32_t a, int32_t b, int32_t *c) : a(a),b(b), c(c) {}

  //  inline int operator()(int & x) const {
  //    return x * hello;
  //  }
};

template <class _ExecutionPolicy, //
          class _ForwardIterator1,
          class _UnaryOperation>
void for_each(_ExecutionPolicy &&__exec, //
              _ForwardIterator1 __first, //
              _ForwardIterator1 __last,  //
              _UnaryOperation __op) {
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
  fprintf(stderr, "Done\n");
}
} // namespace polystl
