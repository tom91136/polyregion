#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: zero4=0 partial=10 fullnested=21 defaultctor=33

#include <algorithm>
#include <array>
#include <cstdio>
#include <execution>

struct DefaultConstructed {
  DefaultConstructed *self;
  int value;
  DefaultConstructed() : self(this), value(7) {}
};

struct HoldsDefaultConstructed {
  DefaultConstructed value;
};

int main() {
  std::array<int, 1> idx{0};

  std::array<int, 1> outZero{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outZero.data()](int) {
    std::array<int, 4> acc = {};
    d[0] = acc[0] + acc[1] + acc[2] + acc[3];
  });

  std::array<int, 1> outFlat{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outFlat.data()](int) {
    std::array<int, 4> acc = {};
    acc[0] = 1;
    acc[1] = 2;
    acc[2] = 3;
    acc[3] = 4;
    d[0] = acc[0] + acc[1] + acc[2] + acc[3];
  });

  std::array<int, 1> outNested{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outNested.data()](int) {
    std::array<std::array<int, 3>, 2> grid = {};
    grid[0][0] = 1;
    grid[0][1] = 2;
    grid[0][2] = 3;
    grid[1][0] = 4;
    grid[1][1] = 5;
    grid[1][2] = 6;
    d[0] = grid[0][0] + grid[0][1] + grid[0][2] + grid[1][0] + grid[1][1] + grid[1][2];
  });

  std::array<int, 1> outDefaultConstructed{-1};
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [d = outDefaultConstructed.data()](int) {
    std::array<DefaultConstructed, 2> values;
    HoldsDefaultConstructed holder;
    DefaultConstructed local;
    const DefaultConstructed constLocal;
    d[0] = values[0].value + values[1].value + (values[0].self == &values[0]) + (values[1].self == &values[1])
           + (holder.value.self == &holder.value) + local.value + (local.self == &local) + constLocal.value
           + (constLocal.self == &constLocal);
  });

  printf("zero4=%d partial=%d fullnested=%d defaultctor=%d", outZero[0], outFlat[0], outNested[0], outDefaultConstructed[0]);
  fflush(stdout);
  return 0;
}
