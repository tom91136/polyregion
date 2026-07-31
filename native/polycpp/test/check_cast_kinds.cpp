#pragma region case: reinterpret
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 127 128 127 42

#pragma region case: dynamic
#pragma region offload-only
#pragma region compile-fails: Unsupported cast Dynamic
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}

#include <cstdio>

#include "test_utils.h"

struct Convertible {
  int v;
  operator int() const { return v + 1; }
};

struct Base {
  virtual ~Base() {}
  int b;
};

struct Derived : Base {
  int d;
};

int main() {
#if CHECK_KIND == 1
  const float f = 1.0f;
  const Convertible c{41};
  const int bits = __polyregion_offload_f1__([=]() { return __builtin_bit_cast(int, f) >> 23; });
  const int rvalueBits = __polyregion_offload_f1__([=]() { return __builtin_bit_cast(int, f + f) >> 23; });
  const int refBits = __polyregion_offload_f1__([=]() {
    float local = f;
    return static_cast<int>(reinterpret_cast<unsigned &>(local) >> 23u);
  });
  const int converted = __polyregion_offload_f1__([=]() { return static_cast<int>(c); });
  std::printf("%d %d %d %d", bits, rvalueBits, refBits, converted);
  return 0;
#elif CHECK_KIND == 2
  Derived derived{};
  derived.b = 1;
  derived.d = 2;
  Base *base = &derived;
  return __polyregion_offload_f1__([=]() {
    const Derived *d = dynamic_cast<const Derived *>(base);
    return d ? d->d : 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
}
