#pragma once

#include <cassert>
#include <cstdio>
// #include "../impl.h"
#include "../impl.h"
// #include "catch2/catch_all.hpp"

template <typename F>
void __polyregion_offload_dispatch__(size_t global,        //
                                     size_t local,         //
                                     size_t localMemBytes, //
                                     F __f, const char *__kernelName, const unsigned char *__kernelImageBytes, size_t __kernelImageSize) {

  fprintf(stderr, "__polyregion_offload_dispatch__(%d, %d, %d, sizeof=%d, %s, %p, %d)\n", global, local, localMemBytes, sizeof(__f),
          __kernelName, __kernelImageBytes, __kernelImageSize);
  fprintf(stderr, "\tImage=");
  for (size_t i = 0; i < __kernelImageSize; ++i) {
    fprintf(stderr, "0x%x ", __kernelImageBytes[i]);
  }
  fprintf(stderr, "\n");
}

template <typename F> void __polyregion_offload__(F __stub_polyregion__f__) {
  const unsigned char *__stub_kernelImageBytes__{};
  int __stub_kernelImageSize__{};
  const char *__stub_kernelName__{};
  __polyregion_offload_dispatch__(1, 0, 0,                   //
                                  __stub_polyregion__f__,    //
                                  __stub_kernelName__,       //
                                  __stub_kernelImageBytes__, //
                                  __stub_kernelImageSize__);
  __stub_polyregion__f__();
}


template <typename T, typename F> T __polyregion_offload_f1__(F __polyregion__f) {
  T __polyregion__v[1]{0};
  __polyregion_offload__([&]() { __polyregion__v[0] = __polyregion__f(); });
  return __polyregion__v[0];
}

template <typename T, typename F> void assertOffload(F g) {
  T expected = g();
  T actual = __polyregion_offload_f1__<T>(g);
  //  assert(expected == actual);
}
