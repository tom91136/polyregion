#pragma region case: this-capture
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

struct Holder {
  int value;

  auto reader() const {
    return [this]() { return value; };
  }
};

int main() {
  Holder holder{42};
  const int result = __polyregion_offload_f1__([=]() {
    auto reader = holder.reader();
    return reader();
  });
  std::printf("%d", result);
  return 0;
}
