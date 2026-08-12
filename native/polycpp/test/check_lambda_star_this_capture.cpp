#pragma region case: star-this-capture
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdio>

#include "test_utils.h"

struct Holder {
  int value;

  auto reader() const {
    return [*this]() { return value; };
  }
};

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    Holder holder{1};
    auto reader = holder.reader();
    return reader();
  });
  std::printf("%d", result);
  return 0;
}
