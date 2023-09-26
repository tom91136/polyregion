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

  //  const static unsigned char data[] = {0XDE, 0xAD, 0xBE, 0xEF};
  //  __stub_kernelImageBytes__ = data;
  //  __stub_kernelImageSize__ = 42;
  //  __stub_kernelName__ = "<stub>";

  __polyregion_offload_dispatch__(1, 0, 0, __stub_polyregion__f__, __stub_kernelName__, __stub_kernelImageBytes__,
                                           __stub_kernelImageSize__);
  __stub_polyregion__f__();
}

template <typename F> struct __polyregion_offload_wrapper__ {};

template <typename T, typename F> T __polyregion_offload_f1__(F __polyregion__f) {
  T __polyregion__v[1]{0};

  //  struct __generated__foo_cpp_34 {
  //
  //    const std::string __kernelImage = {0x01};
  //    const std::string __uniqueName = "theName";
//    const polyregion::runtime::ArgBuffer __argBuffer{
//        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &a},
//        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &b},
//        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Ptr, &c},
//    };
//
//    int32_t a;
//    int32_t b;
//    int32_t *c;
//
//    __generated__foo_cpp_34(int32_t a, int32_t b, int32_t *c) : a(a), b(b), c(c) {}
//  };
//

  //  __wrapper.

  __polyregion_offload__([&]() {
    printf("lam=%lu\n", sizeof(F));
    __polyregion__v[0] = __polyregion__f();
  });

  //  struct __polyregion_offload_functor__ {
  //    T (&__v)[1];
  //    F (&__f);
  //    void operator()() { this->__v[0] = this->__f(); }
  //  };
  //  __polyregion_offload__(__polyregion_offload_functor__{__polyregion__v, __polyregion__f});
  return __polyregion__v[0];
}

namespace {

void m(){


  struct X{int a;};
  X x;

  struct __lam__{
    X x;
  };

  auto m = [&](){
    return x.a;
  };
  m();
}

}




/*

 // For every lambda expression, lift to struct
 //
 struct __lam2__{
   int aaa;
   int bbbb;
   int operator()(){
     return aaa + bbb;
   }
 }

 struct __lam2__{
   int u;
   __lam1__ g;
   int operator()(){
      int bbb = u;
      return g.operator();
   }
 }


*/

template <typename T, typename F> void assertOffload(F g) {
  T expected = g();
  T actual = __polyregion_offload_f1__<T>(g);
  //  assert(expected == actual);
}
