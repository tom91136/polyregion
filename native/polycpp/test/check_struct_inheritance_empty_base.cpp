#pragma region case: inheritance
#pragma region using: capture=&,=,value
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 33 44 1 1 1 1

#include <cstddef>
#include <cstdio>

#include "test_utils.h"

int main() {

  struct Base {};

  struct Derived : Base {
    int c, d;
  };
  struct A : Base {};
  struct B : Base {};
  struct Repeated : A, B {};
  Derived value{{}, 33, 44};

  static_assert(sizeof(Base) == 1);
  static_assert(sizeof(Derived) == sizeof(int) * 2);

  Derived result = __polyregion_offload_f1__([CHECK_CAPTURE]() { return value; });
  int base = __polyregion_offload_f1__([CHECK_CAPTURE]() { return static_cast<const Base *>(&value) != nullptr ? 1 : 0; });
  int repeated = __polyregion_offload_f1__([]() {
    Repeated value;
    return static_cast<const void *>(static_cast<const B *>(&value)) != static_cast<const void *>(&value) ? 1 : 0;
  });
  int nullPreserved = __polyregion_offload_f1__([]() {
    Repeated *value = nullptr;
    return static_cast<const B *>(value) == nullptr ? 1 : 0;
  });
  int reference = __polyregion_offload_f1__([]() {
    Repeated value;
    return static_cast<const void *>(&static_cast<const B &>(value)) != static_cast<const void *>(&value) ? 1 : 0;
  });
  printf("%d %d %d %d %d %d", result.c, result.d, base, repeated, nullPreserved, reference);
  return 0;
}
