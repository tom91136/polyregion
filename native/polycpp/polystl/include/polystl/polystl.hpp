#pragma once

#include <cassert>
#include <cstdio>

#include "runtime.h"

template <typename F>
POLYREGION_EXPORT void __polyregion_offload_dispatch_impl__(size_t global,        //
                                                            size_t local,         //
                                                            size_t localMemBytes, //
                                                            F __f, const char *__kernelName, const unsigned char *__kernelImageBytes,
                                                            size_t __kernelImageSize) {

  //  fprintf(stderr, "__polyregion_offload_dispatch_impl__(%ld, %ld, %ld, sizeof=%lu, %s, %p, %ld)\n", global, local, localMemBytes,
  //  sizeof(__f),
  //          __kernelName, __kernelImageBytes, __kernelImageSize);
  //  fprintf(stderr, "\tImage=");
  //  for (size_t i = 0; i < __kernelImageSize; ++i) {
  //    fprintf(stderr, "0x%x ", __kernelImageBytes[i]);
  //  }
  polystl::__polyregion_offload_dispatch__(global, localMemBytes, localMemBytes, __f, __kernelName, __kernelImageBytes, __kernelImageSize);
}

template <typename F> POLYREGION_EXPORT void __polyregion_offload__(F __stub_polyregion__f__) {
  const unsigned char *__stub_kernelImageBytes__{};
  int __stub_kernelImageSize__{};
  const char *__stub_kernelName__{};
  __polyregion_offload_dispatch_impl__(1, 0, 0,                   //
                                       __stub_polyregion__f__,    //
                                       __stub_kernelName__,       //
                                       __stub_kernelImageBytes__, //
                                       __stub_kernelImageSize__);
  __stub_polyregion__f__();
}

template <typename F> POLYREGION_EXPORT std::invoke_result_t<F> __polyregion_offload_f1__(F __polyregion__f) {
  static bool offload = !std::getenv("POLYSTL_NO_OFFLOAD");
  std::invoke_result_t<F> __polyregion__v{};
  if (offload) {
    __polyregion_offload__([&__polyregion__v, &__polyregion__f]() { __polyregion__v = __polyregion__f(); });
  } else {
    [&__polyregion__v, &__polyregion__f]() { __polyregion__v = __polyregion__f(); }();
  }
  return __polyregion__v;
}

template <typename T, typename F> void assertOffload(F g) {
  T expected = g();
  T actual = __polyregion_offload_f1__<T>(g);
  //  assert(expected == actual);
}
