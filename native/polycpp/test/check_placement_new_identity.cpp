#pragma region case: placement-new-identity
#pragma region using: capture=&,=
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>
#include <new>

#include "test_utils.h"

struct SelfPointer {
  SelfPointer *self;
  int value;

  explicit SelfPointer(int value) : self(this), value(value) {}
};

struct AggregateSelf {
  AggregateSelf *self = this;
  int value = 20;
};

struct BaseSelf {
  BaseSelf *self = this;
};

struct DerivedSelf : BaseSelf {
  DerivedSelf() : BaseSelf{} {}
};

int main() {
  const int result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
    alignas(SelfPointer) unsigned char storage[sizeof(SelfPointer)];
    alignas(AggregateSelf) unsigned char aggregateStorage[sizeof(AggregateSelf)];
    alignas(int) unsigned char scalarStorage[sizeof(int)];
    auto *value = ::new (static_cast<void *>(storage)) SelfPointer(10);
    auto *aggregate = ::new (static_cast<void *>(aggregateStorage)) AggregateSelf{};
    auto *scalar = ::new (static_cast<void *>(scalarStorage)) int(12);
    DerivedSelf derived;
    const bool identities = value->self == value && aggregate->self == aggregate && derived.self == static_cast<BaseSelf *>(&derived);
    return identities ? value->value + aggregate->value + *scalar : 0;
  });
  std::printf("%d", result);
  return 0;
}
