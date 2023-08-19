#pragma once

//#include "../impl.h"
#include "catch2/catch_all.hpp"

template <typename F> void __polyregion_offload__(F f) { return f(); }

template <typename T, typename F> T offload(F f) {
  T v[1]{0};

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
//  polystl::__polyregion_offload_dispatch__(1, 0, 0, __generated__foo_cpp_34(1, 2, nullptr), [&]() { v[0] = f(); });

  // f.
  __polyregion_offload__([&]() { v[0] = f(); });
  return v[0];
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
  T actual = offload<T>(g);
  CHECK(expected == actual);
}
