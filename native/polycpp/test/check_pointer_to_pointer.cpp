#pragma region case: nested-pointer-load-store
#pragma region using: value_type=int,float,Record
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_VALUE_TYPE={value_type} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 9253741

#include <cstdio>

#include "test_utils.h"

struct Record {
  char tag;
  struct Payload {
    short code;
    int value;
    long long weight;
  } payload;
  int tail[3];

  explicit Record(int x)
      : tag(static_cast<char>(x)), payload{static_cast<short>(x + 1), x, static_cast<long long>(x) * 1000}, tail{x + 2, x + 3, x + 4} {}

  Record &operator+=(Record other) {
    payload.value += other.payload.value;
    return *this;
  }
};

#ifndef CHECK_VALUE_TYPE
  #error "CHECK_VALUE_TYPE undefined"
#endif

int value(int x) { return x; }
int value(float x) { return static_cast<int>(x); }
int value(Record x) { return x.payload.value; }

int main() {
  using T = CHECK_VALUE_TYPE;
  T captured{41};
  T *capturedInner = &captured;
  T **capturedOuter = &capturedInner;

  const int result = __polyregion_offload_f1__([=]() {
    T first{8};
    T second{20};
    T third{30};
    T *inner = &first;
    T **middle = &inner;
    T ***outer = &middle;

    **middle += T{1};
    **outer = &second;
    ***outer += T{5};
    middle[0] = &third;
    middle[0][0] += T{7};

    return ((value(first) * 100 + value(second)) * 100 + value(third)) * 100 + value(capturedOuter[0][0]);
  });
  std::printf("%d", result);
  return 0;
}
