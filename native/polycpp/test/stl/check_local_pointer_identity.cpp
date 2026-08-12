#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: identity=127

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>

struct PointerHolder {
  int *ptr;
};

struct OtherPointerHolder {
  int *ptr;
};

struct NestedPointerHolder {
  PointerHolder holder;
};

struct NullPointerHolder {
  int *ptr;
};

int main() {
  std::array<int, 1> idx{0};
  std::array<int, 1> out{-1};

  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = out.data()](int) {
    int values[2]{11, 22};
    int *first = &values[0];
    int *firstAlias = values;
    int *firstCopy = first;
    int *second = &values[1];

    PointerHolder a{first};
    PointerHolder b{firstAlias};
    PointerHolder c{firstCopy};
    PointerHolder different{second};
    PointerHolder copiedField{a.ptr};
    OtherPointerHolder other{&values[0]};
    NestedPointerHolder nested{{&values[1]}};
    NullPointerHolder empty{nullptr};

    d[0] = (a.ptr == b.ptr) + 2 * (a.ptr == c.ptr) + 4 * (a.ptr != different.ptr) + 8 * (other.ptr == &values[0])
           + 16 * (nested.holder.ptr == &values[1]) + 32 * (empty.ptr == nullptr) + 64 * (copiedField.ptr == &values[0]);
  });

  printf("identity=%d", out[0]);
  fflush(stdout);
  return 0;
}
