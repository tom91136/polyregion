#pragma region case: addressof
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 23

#include <cstdio>
#include <memory>

#include "test_utils.h"

struct Value {
  int value;
  Value *operator&() { return nullptr; }
};

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    Value value{20};
    Value *standard = std::addressof(value);
    Value *builtin = __builtin_addressof(value);
    standard->value += 2;
    Value *pointer = builtin;
    return pointer != reinterpret_cast<Value *>(std::addressof(pointer)) ? 23 : 22;
  });
  std::printf("%d", result);
  return 0;
}
