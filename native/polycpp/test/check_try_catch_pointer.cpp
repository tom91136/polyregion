#pragma region case: public-base-adjustment
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 7

#pragma region case: void-pointer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 11

#pragma region case: null-public-base
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 13

#pragma region case: private-base-does-not-match
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 17

#pragma region case: ambiguous-base-does-not-match
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 19

#pragma region case: null-pointer-matches-pointer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 23

#ifndef CHECK_KIND
  #define CHECK_KIND 0
#endif

#include <cstdio>

#include "test_utils.h"

struct PointerBase {
  int code;
};
struct Prefix {
  int prefix;
};
struct PointerDerived : Prefix, PointerBase {};
struct PrivatePointerDerived : private PointerBase {
  explicit PrivatePointerDerived(int value) { code = value; }
};
struct PointerLeft : PointerBase {};
struct PointerRight : PointerBase {};
struct AmbiguousPointerDerived : PointerLeft, PointerRight {};

int main() {
  int data[2] = {1, 7};
  int *p = data;
#if CHECK_KIND == 0
  const int r = __polyregion_offload_f1__([=]() {
    PointerDerived value{{3}, {p[1]}};
    try {
      if (p[0] == 1) throw &value;
    } catch (PointerBase *e) {
      return e->code;
    }
    return 0;
  });
#elif CHECK_KIND == 1
  const int r = __polyregion_offload_f1__([=]() {
    int value = p[1];
    try {
      if (p[0] == 1) throw &value;
    } catch (void *e) {
      return e == &value ? 11 : 0;
    }
    return 0;
  });
#elif CHECK_KIND == 2
  const int r = __polyregion_offload_f1__([=]() {
    PointerDerived *value = nullptr;
    try {
      if (p[0] == 1) throw value;
    } catch (PointerBase *e) {
      return e == nullptr ? 13 : 0;
    }
    return 0;
  });
#elif CHECK_KIND == 3
  const int r = __polyregion_offload_f1__([=]() {
    PrivatePointerDerived value{p[1]};
    try {
      if (p[0] == 1) throw &value;
    } catch (PointerBase *) {
      return 1;
    } catch (...) {
      return 17;
    }
    return 0;
  });
#elif CHECK_KIND == 4
  const int r = __polyregion_offload_f1__([=]() {
    AmbiguousPointerDerived value{};
    try {
      if (p[0] == 1) throw &value;
    } catch (PointerBase *) {
      return 1;
    } catch (...) {
      return 19;
    }
    return 0;
  });
#elif CHECK_KIND == 5
  const int r = __polyregion_offload_f1__([=]() {
    try {
      if (p[0] == 1) throw static_cast<int *>(nullptr);
    } catch (int *e) {
      return e == nullptr ? 23 : 0;
    }
    return 0;
  });
#else
  #error "CHECK_KIND undefined"
#endif
  std::printf("%d", r);
  return 0;
}
