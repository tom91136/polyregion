#include "impl.h"
#include <cstdlib>
#include <iostream>

int main2() {

  struct __generated__foo_cpp_34 {

    const std::string __kernelImage = {0x01};
    const std::string __uniqueName = "theName";
    const polyregion::runtime::ArgBuffer __argBuffer{
        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &a},
        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &b},
        polyregion::runtime::TypedPointer{polyregion::runtime::Type::Ptr, &c},
    };

    int32_t a;
    int32_t b;
    int32_t *c;

    __generated__foo_cpp_34(int32_t a, int32_t b, int32_t *c) : a(a), b(b), c(c) {}
  };

  std::vector<int> indices{0, 1};
  std::vector<int> out{42, 43};
  int a = 1;
  int b = 2;

  setenv("POLY_PLATFORM", "cuda", true);
  setenv("POLY_DEVICE", "0", true);

//  polystl::for_each(
//      std::execution::par_unseq,                    //
//      indices.begin(), indices.end(),               //
//      [&](auto &x) { return out[x] += a * b + x; }, //
//      __generated__foo_cpp_34(a, b, out.data()));
  std::cout << out[0] << std::endl;
  return EXIT_SUCCESS;
}