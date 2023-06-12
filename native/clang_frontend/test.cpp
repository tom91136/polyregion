#include "impl.h"
#include <cstdlib>

struct __generated__foo_cpp_34 {

  int32_t a;
  int32_t b;
  int32_t *c;
  constexpr static std::string const __kernelImage = {0x01};
  constexpr static std::string const __uniqueName = "theName";
  const polyregion::runtime::ArgBuffer __argBuffer{
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &a},
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Int32, &b},
      polyregion::runtime::TypedPointer{polyregion::runtime::Type::Ptr, &c},
  };

  __generated__foo_cpp_34(int32_t a, int32_t b, int32_t *c) : a(a), b(b), c(c) {}

  inline int operator()(int &x) const { return c[0] = a * b + x; }
};

int main() {

  std::vector xs{1};
  std::vector out{1};

  setenv("POLY_PLATFORM", "cuda", true);
  setenv("POLY_DEVICE", "0", true);

  polystl::for_each(std::execution::par_unseq, //
                    xs.begin(), xs.end(),      //
                    __generated__foo_cpp_34(1, 2, out.data()));
  return EXIT_SUCCESS;
}